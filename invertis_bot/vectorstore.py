"""Persistent Chroma vector-index lifecycle and category-aware retrieval."""

from __future__ import annotations

import logging
import hashlib
import math
import re
import shutil
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from invertis_bot.config import Settings, get_settings
from invertis_bot.exceptions import IndexNotFoundError, LLMError

logger = logging.getLogger(__name__)


class UnicodeHashEmbeddings(Embeddings):
    """Deterministic local multilingual feature embeddings for quota-safe fallback.

    This lightweight hashed word/character representation requires no model download.
    Gemini remains preferred; the local representation keeps retrieval available when
    hosted embedding quotas are unavailable.
    """

    dimensions = 2048

    @classmethod
    def _embed(cls, text: str) -> list[float]:
        tokens = _tokens(text)
        stop_words = {
            "a", "an", "and", "are", "as", "at", "be", "can", "do", "does", "for", "from",
            "how", "i", "in", "is", "it", "me", "my", "of", "on", "or", "the", "there",
            "this", "to", "was", "what", "when", "where", "which", "who", "why", "with",
            "you", "your", "के", "लिए", "कैसे", "करें", "में", "का", "की", "को", "है", "हैं", "क्या",
        }
        tokens = [token for token in tokens if token not in stop_words]
        aliases = {
            "admissions": "admission", "apply": "application", "applying": "application",
            "applications": "application", "fees": "fee", "cost": "fee", "costs": "fee",
            "programmes": "programme", "programs": "programme", "courses": "course",
            "placements": "placement", "jobs": "placement", "results": "result",
            "प्रवेश": "admission", "दाखिला": "admission", "एडमिशन": "admission",
            "आवेदन": "application", "अप्लाई": "application",
            "फीस": "fee", "शुल्क": "fee", "पैसे": "fee",
            "छात्रावास": "hostel", "हॉस्टल": "hostel", "कोर्स": "course",
            "पाठ्यक्रम": "course", "प्लेसमेंट": "placement", "नौकरी": "placement",
            "संपर्क": "contact", "फोन": "contact", "परिणाम": "result", "रिजल्ट": "result",
            "पात्रता": "eligibility", "योग्यता": "eligibility", "विभाग": "department",
            "कब": "date", "कहां": "location", "कहाँ": "location",
        }
        tokens = [aliases.get(token, token) for token in tokens]
        features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        vector = [0.0] * cls.dimensions
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            index = value % cls.dimensions
            vector[index] += 1.0 if value & (1 << 63) else -1.0
        norm = math.sqrt(sum(component * component for component in vector))
        return [component / norm for component in vector] if norm else vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return normalized feature vectors for source passages."""
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Return the same deterministic representation used for indexed passages."""
        return self._embed(text)


class VectorStoreManager:
    """Create, reload, and search a persistent Chroma collection."""

    COLLECTION_NAME = "invertis_university"
    HOSTED_EMBEDDING_CHUNK_LIMIT = 75

    def __init__(self, settings: Settings | None = None, embeddings: Any | None = None) -> None:
        self.settings = settings or get_settings()
        self._embeddings = embeddings
        self._embedding_backend: str | None = None
        self._store: Any | None = None

    def _get_embeddings(self, backend: str | None = None) -> Any:
        """Construct Gemini embeddings on demand, avoiding API setup in unit tests."""
        selected = backend or self._embedding_backend or "gemini"
        if selected == "local":
            self._embedding_backend = "local"
            self._embeddings = self._embeddings or UnicodeHashEmbeddings()
            return self._embeddings
        if self._embeddings is not None:
            return self._embeddings
        if not self.settings.api_key:
            raise LLMError("GOOGLE_API_KEY is required to build or query the hosted embedding model.")
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            self._embeddings = GoogleGenerativeAIEmbeddings(
                model=self.settings.embedding_model,
                google_api_key=self.settings.api_key,
            )
            return self._embeddings
        except Exception as exc:
            raise LLMError(f"Could not initialize Gemini embeddings: {exc}") from exc

    def build(self, documents: list[Document], *, rebuild: bool = False) -> int:
        """Persist embedded chunks, retrying locally when hosted embeddings are throttled."""
        if not documents:
            raise ValueError("Cannot build a vector index with no document chunks")
        try:
            from langchain_chroma import Chroma
            self.settings.index_dir.parent.mkdir(parents=True, exist_ok=True)
            if self.settings.index_dir.exists() and not rebuild:
                self._store = self.load()
                self._store.add_documents(documents)
                logger.info("Added %d chunks to %s", len(documents), self.settings.index_dir)
                return len(documents)

            selected_backend = (
                "gemini"
                if self.settings.api_key and len(documents) <= self.HOSTED_EMBEDDING_CHUNK_LIMIT
                else "local"
            )
            if selected_backend == "local" and self.settings.api_key:
                logger.info(
                    "Using local embeddings for %d chunks to avoid hosted embedding request quotas",
                    len(documents),
                )
            if rebuild:
                # Chroma can retain SQLite file handles on Windows; avoid renaming open DB folders.
                self._store = None
                shutil.rmtree(self.settings.index_dir, ignore_errors=True)
            self.settings.index_dir.mkdir(parents=True, exist_ok=True)
            try:
                self._store = Chroma.from_documents(
                    documents=documents,
                    embedding=self._get_embeddings(selected_backend),
                    collection_name=self.COLLECTION_NAME,
                    persist_directory=str(self.settings.index_dir),
                )
            except Exception as hosted_error:
                if selected_backend == "local":
                    raise
                logger.warning(
                    "Gemini embedding failed; using local hashed embeddings for this index: %s",
                    hosted_error,
                )
                self._store = None
                self._embeddings = None
                self._embedding_backend = "local"
                shutil.rmtree(self.settings.index_dir, ignore_errors=True)
                self.settings.index_dir.mkdir(parents=True, exist_ok=True)
                self._store = Chroma.from_documents(
                    documents=documents,
                    embedding=self._get_embeddings("local"),
                    collection_name=self.COLLECTION_NAME,
                    persist_directory=str(self.settings.index_dir),
                )
                selected_backend = "local"

            (self.settings.index_dir / ".embedding_backend").write_text(selected_backend, encoding="utf-8")
            self._embedding_backend = selected_backend
            logger.info("Indexed %d chunks in %s", len(documents), self.settings.index_dir)
            return len(documents)
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"Could not build Chroma index: {exc}") from exc

    def load(self) -> Any:
        """Load the persisted collection, raising a useful message if absent."""
        if self._store is not None:
            return self._store
        if not (self.settings.index_dir / "chroma.sqlite3").exists():
            raise IndexNotFoundError("The search index is empty. Run `python -m invertis_bot.ingest --rebuild` first.")
        try:
            from langchain_chroma import Chroma
            marker = self.settings.index_dir / ".embedding_backend"
            backend = marker.read_text(encoding="utf-8").strip() if marker.exists() else "gemini"
            self._embedding_backend = backend
            self._store = Chroma(
                collection_name=self.COLLECTION_NAME,
                embedding_function=self._get_embeddings(backend),
                persist_directory=str(self.settings.index_dir),
            )
            return self._store
        except IndexNotFoundError:
            raise
        except Exception as exc:
            raise LLMError(f"Could not load Chroma index: {exc}") from exc

    def search(self, query: str, *, top_k: int | None = None, category: str | None = None) -> list[Document]:
        """Retrieve relevant chunks with maximal marginal relevance and optional filter."""
        store = self.load()
        filter_query = {"category": category} if category and category != "all" else None
        return store.max_marginal_relevance_search(
            query,
            k=top_k or self.settings.retrieval_top_k,
            fetch_k=max((top_k or self.settings.retrieval_top_k) * 3, 10),
            filter=filter_query,
        )

    def confidence(self, query: str, *, category: str | None = None) -> float:
        """Estimate best-match relevance in the [0, 1] scale used by Chroma."""
        store = self.load()
        filter_query = {"category": category} if category and category != "all" else None
        # Hashed bag-of-words vectors are intentionally less semantic. Combine their
        # similarity with lexical coverage so directly relevant passages pass the
        # threshold while unrelated top-k hits still fall back.
        if self._embedding_backend == "local":
            documents = store.similarity_search(query, k=3, filter=filter_query)
            return _lexical_coverage(query, documents)

        scored = store.similarity_search_with_relevance_scores(query, k=3, filter=filter_query)
        return max((max(0.0, min(1.0, float(item[1]))) for item in scored), default=0.0)

    def get_store(self) -> Any:
        """Expose the loaded underlying vector store for advanced integrations."""
        return self.load()


def _lexical_coverage(query: str, documents: list[Document]) -> float:
    """Measure overlap between query terms and candidate passage terms."""
    query_terms = _canonical_terms(query)
    if not query_terms:
        return 0.0
    return max(
        (
            len(query_terms & _canonical_terms(document.page_content))
            / len(query_terms)
            for document in documents
        ),
        default=0.0,
    )


def _canonical_terms(text: str) -> set[str]:
    """Map common Hindi/Hinglish and English variants to shared retrieval terms."""
    stop_words = {"के", "लिए", "कैसे", "करें", "में", "का", "की", "को", "है", "हैं", "क्या"}
    aliases = {
        "admissions": "admission", "apply": "application", "applying": "application",
        "applications": "application", "fees": "fee", "cost": "fee", "costs": "fee",
        "programmes": "programme", "programs": "programme", "courses": "course",
        "placements": "placement", "jobs": "placement", "results": "result",
        "प्रवेश": "admission", "दाखिला": "admission", "एडमिशन": "admission",
        "आवेदन": "application", "अप्लाई": "application",
        "फीस": "fee", "शुल्क": "fee", "पैसे": "fee",
        "छात्रावास": "hostel", "हॉस्टल": "hostel", "कोर्स": "course",
        "पाठ्यक्रम": "course", "प्लेसमेंट": "placement", "नौकरी": "placement",
        "संपर्क": "contact", "फोन": "contact", "परिणाम": "result", "रिजल्ट": "result",
        "पात्रता": "eligibility", "योग्यता": "eligibility", "विभाग": "department",
        "कब": "date", "कहां": "location", "कहाँ": "location",
    }
    return {
        aliases.get(token, token)
        for token in _tokens(text)
        if token not in stop_words
    }


def _tokens(text: str) -> list[str]:
    """Tokenize Latin and Devanagari words without dropping Hindi vowel marks."""
    return re.findall(r"[\w\u0900-\u097f]+", text.lower(), flags=re.UNICODE)
