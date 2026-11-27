from langchain_core.documents import Document

from invertis_bot.config import Settings
from invertis_bot.vectorstore import VectorStoreManager, _bm25_rank, _reciprocal_rank_fusion


class FakeStore:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents

    def get(self, *, include: list[str]) -> dict[str, list]:
        assert include == ["documents", "metadatas"]
        return {
            "documents": [document.page_content for document in self.documents],
            "metadatas": [document.metadata for document in self.documents],
        }

    def max_marginal_relevance_search(
        self, query: str, *, k: int, fetch_k: int, filter: dict | None
    ) -> list[Document]:
        return self.documents[:k]


def test_bm25_ranks_exact_query_terms_above_unrelated_passages() -> None:
    exact = Document(page_content="Application form and admission portal")
    unrelated = Document(page_content="Campus sports and student events")
    ranked = _bm25_rank({"application", "admission"}, [unrelated, exact])
    assert ranked[0][0] == exact
    assert ranked[0][1] > 0


def test_reciprocal_rank_fusion_merges_and_deduplicates_candidates() -> None:
    shared = Document(page_content="Admission application", metadata={"url": "/admission"})
    semantic_only = Document(page_content="General campus information")
    lexical_only = Document(page_content="Fee structure", metadata={"url": "/fees"})
    result = _reciprocal_rank_fusion(
        ([semantic_only, shared], [shared, lexical_only]),
        limit=3,
        weights=(1.0, 1.25),
    )
    assert len(result) == 3
    assert result[0] == shared
    assert {document.page_content for document in result} == {
        "Admission application",
        "General campus information",
        "Fee structure",
    }


def test_hybrid_search_uses_keyword_candidates_and_category_filter(tmp_path) -> None:
    matching = Document(
        page_content="The application form is available on the admission portal.",
        metadata={"category": "admission", "url": "/admission"},
    )
    unrelated = Document(
        page_content="Campus sports and student clubs.",
        metadata={"category": "general", "url": "/campus"},
    )
    manager = VectorStoreManager(Settings(index_dir=tmp_path / "index"))
    manager._store = FakeStore([unrelated, matching])
    manager._embedding_backend = "local"

    result = manager.search("application form admission portal", category="admission")

    assert result
    assert result[0].metadata["category"] == "admission"
    assert manager.confidence("application form admission portal", category="admission") > 0
