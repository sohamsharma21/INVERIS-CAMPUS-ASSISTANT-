"""Tests for the quota-safe local embedding fallback."""

from invertis_bot.vectorstore import UnicodeHashEmbeddings


def test_local_embeddings_are_deterministic_and_normalized() -> None:
    embeddings = UnicodeHashEmbeddings()
    first = embeddings.embed_query("Invertis University admissions")
    second = embeddings.embed_query("Invertis University admissions")
    assert first == second
    assert len(first) == embeddings.dimensions
    assert abs(sum(value * value for value in first) - 1.0) < 1e-9


def test_local_embeddings_support_hindi_and_text_batches() -> None:
    embeddings = UnicodeHashEmbeddings()
    vectors = embeddings.embed_documents(["hostel fees", "छात्रावास की फीस"])
    assert len(vectors) == 2
    assert all(len(vector) == embeddings.dimensions for vector in vectors)
    assert vectors[0] == vectors[1]
