"""Text cleanup and lazy document chunking with source metadata preservation."""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from invertis_bot.loaders import category_for_path, category_for_url


@dataclass(slots=True)
class DocumentProcessor:
    """Normalize source text and produce metadata-rich overlapping chunks."""

    chunk_size: int = 1000
    chunk_overlap: int = 160
    splitter: RecursiveCharacterTextSplitter = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            add_start_index=True,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    @staticmethod
    def clean_text(text: str) -> str:
        """Remove control noise and collapse excessive whitespace without losing paragraphs."""
        text = text.replace("\xa0", " ").replace("\r", "\n")
        text = re.sub(r"[\t\f\v ]+", " ", text)
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
        return text.strip()

    def chunks(self, documents: Iterable[Document]) -> Iterator[Document]:
        """Yield chunks lazily, preserving and normalizing useful source metadata."""
        for document in documents:
            clean = self.clean_text(document.page_content)
            if not clean:
                continue
            # Chroma only accepts scalar metadata and rejects None values (e.g. web page numbers).
            metadata: dict[str, Any] = {key: value for key, value in document.metadata.items() if value is not None}
            source = str(metadata.get("url") or metadata.get("source") or "")
            metadata.setdefault("url", source)
            metadata.setdefault("title", source.rsplit("/", 1)[-1] or "University document")
            metadata.setdefault("category", category_for_url(source) if source.startswith("http") else category_for_path(Path(source)))
            for chunk in self.splitter.create_documents([clean], metadatas=[metadata]):
                yield chunk
