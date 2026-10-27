"""Tests for cleanup, category tagging, and lazy chunk metadata."""

from langchain_core.documents import Document

from invertis_bot.processing import DocumentProcessor


def test_clean_text_collapses_whitespace_and_preserves_paragraphs() -> None:
    assert DocumentProcessor.clean_text("  Hello\tworld\r\n\n\nNext  ") == "Hello world\n\nNext"


def test_empty_document_is_skipped() -> None:
    processor = DocumentProcessor(chunk_size=100, chunk_overlap=10)
    assert list(processor.chunks([Document(page_content="  ", metadata={})])) == []


def test_chunking_preserves_url_category_and_title() -> None:
    processor = DocumentProcessor(chunk_size=30, chunk_overlap=5)
    source = Document(page_content="A long course description. " * 8, metadata={"url": "https://example.test/ug-programmes", "title": "Courses"})
    chunks = list(processor.chunks([source]))
    assert len(chunks) > 1
    assert all(chunk.metadata["category"] == "courses" for chunk in chunks)
    assert all(chunk.metadata["title"] == "Courses" for chunk in chunks)
    assert all(chunk.metadata["url"] == "https://example.test/ug-programmes" for chunk in chunks)


def test_local_file_defaults_to_general_category() -> None:
    processor = DocumentProcessor(chunk_size=200, chunk_overlap=20)
    chunks = list(processor.chunks([Document(page_content="Campus overview", metadata={"url": "overview.txt"})]))
    assert chunks[0].metadata["category"] == "general"
