"""Offline tests for fallback safety, citations, and memory trimming."""

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from invertis_bot.config import Settings
from invertis_bot.rag import ConversationMemory, RAGEngine


class EmptyVectorStore:
    """Retriever test double that provides no evidence."""

    def confidence(self, query: str, *, category: str | None = None) -> float:
        return 0.0

    def search(self, query: str, *, top_k: int | None = None, category: str | None = None) -> list[Document]:
        return []


class RelevantVectorStore(EmptyVectorStore):
    """Retriever test double with one verified passage."""

    def confidence(self, query: str, *, category: str | None = None) -> float:
        return 0.9

    def search(self, query: str, *, top_k: int | None = None, category: str | None = None) -> list[Document]:
        return [Document(page_content="The documented programme duration is four years.", metadata={"url": "https://www.invertisuniversity.ac.in/ug-programmes/sample", "title": "Sample programme", "category": "courses"})]


def fake_model() -> RunnableLambda:
    """Create a deterministic LCEL-compatible model double with no API calls."""
    return RunnableLambda(lambda _: "The documented programme duration is four years.")


def test_empty_retrieval_returns_official_site_fallback() -> None:
    settings = Settings(retrieval_score_threshold=0.3)
    result = RAGEngine(EmptyVectorStore(), settings=settings).answer("What is the exact fee?")
    assert result.found is False
    assert "couldn’t find verified information" in result.answer
    assert settings.website_url in result.answer
    assert not result.sources


def test_low_confidence_returns_fallback_without_calling_llm() -> None:
    result = RAGEngine(EmptyVectorStore(), model=fake_model()).answer("Unknown question")
    assert result.found is False
    assert "not present" not in result.answer.lower()
    assert "official website" in result.answer.lower()


def test_answer_includes_source_citation_and_uses_fake_llm() -> None:
    result = RAGEngine(RelevantVectorStore(), model=fake_model()).answer("How long is the programme?")
    assert result.found is True
    assert "four years" in result.answer
    assert result.sources[0].location.endswith("/sample")
    assert result.sources[0].category == "courses"


def test_conversation_memory_keeps_last_n_exchanges() -> None:
    memory = ConversationMemory(max_turns=2)
    memory.add("one", "a")
    memory.add("two", "b")
    memory.add("three", "c")
    assert len(memory.messages) == 2
    assert "one" not in memory.as_text()
    assert "three" in memory.as_text()


def test_fallback_is_streamed_and_saved_to_memory() -> None:
    memory = ConversationMemory(max_turns=3)
    prepared = RAGEngine(EmptyVectorStore()).prepare("Hostel details?", memory=memory)
    output = "".join(prepared.tokens)
    assert output
    assert memory.messages[-1][0] == "Hostel details?"


def test_hindi_fallback_uses_hindi_and_official_contact_route() -> None:
    result = RAGEngine(EmptyVectorStore()).answer("छात्रावास की फीस क्या है?")
    assert "मुझे उपलब्ध सत्यापित जानकारी" in result.answer
    assert result.sources == ()


def test_source_only_mode_answers_from_retrieved_passage() -> None:
    settings = Settings(enable_gemini=False)
    result = RAGEngine(RelevantVectorStore(), settings=settings).answer("How long is the programme?")
    assert result.found is True
    assert "official university sources" in result.answer
    assert "four years" in result.answer
    assert result.sources


def test_generation_failure_returns_extractive_source_answer() -> None:
    def fail_generation(_: Any) -> str:
        raise RuntimeError("rate limited")

    rate_limited_model = RunnableLambda(fail_generation)

    result = RAGEngine(
        RelevantVectorStore(),
        model=rate_limited_model,
        settings=Settings(enable_gemini=True),
    ).answer("How long is the programme?")
    assert result.found is True
    assert "relevant text from the university sources" in result.answer
    assert "four years" in result.answer
    assert result.sources
