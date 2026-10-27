"""Pluggable loaders for scraped web pages and user-provided documents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from urllib.parse import urlparse

import requests
from langchain_core.documents import Document
from pypdf import PdfReader

from invertis_bot.exceptions import DocumentLoadError
from invertis_bot.utils import DocumentRecord


class BaseLoader(ABC):
    """Abstract source loader; subclasses normalize data to LangChain Documents."""

    @abstractmethod
    def load(self) -> list[Document]:
        """Load documents from the configured source."""


class WebLoader(BaseLoader):
    """Fetch and clean one allowed public HTML page."""

    def __init__(self, url: str, *, timeout: float = 20.0) -> None:
        self.url = url
        self.timeout = timeout

    def load(self) -> list[Document]:
        """Fetch a public web page and preserve URL/title/category metadata."""
        try:
            from invertis_bot.config import get_settings
            from invertis_bot.scraper import Scraper

            settings = get_settings().model_copy(update={"request_timeout_seconds": self.timeout})
            scraper = Scraper(settings)
            if not scraper._allowed(self.url):
                raise DocumentLoadError(f"Robots policy or host restrictions disallow {self.url}")
            html = scraper._get_html(self.url)
            if not html:
                return []
            content, title, _ = scraper._extract(html, self.url)
            category = category_for_url(self.url)
            metadata = {"url": self.url, "title": title, "category": category, "scraped_at": DocumentRecord(content, self.url, title, category).scraped_at}
            return [Document(page_content=content, metadata=metadata)] if content else []
        except DocumentLoadError:
            raise
        except requests.RequestException as exc:
            raise DocumentLoadError(f"Could not load web page {self.url}: {exc}") from exc


class PDFLoader(BaseLoader):
    """Extract text page-by-page from a local PDF while keeping page citations."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> list[Document]:
        """Return one LangChain Document per PDF page."""
        try:
            reader = PdfReader(str(self.path))
            documents: list[Document] = []
            for page_number, page in enumerate(reader.pages, start=1):
                content = (page.extract_text() or "").strip()
                if content:
                    documents.append(Document(
                        page_content=content,
                        metadata={"url": str(self.path.resolve()), "title": self.path.name, "category": category_for_path(self.path), "page": page_number},
                    ))
            if not documents:
                raise DocumentLoadError(f"No extractable text found in {self.path}")
            return documents
        except Exception as exc:
            if isinstance(exc, DocumentLoadError):
                raise
            raise DocumentLoadError(f"Could not parse PDF {self.path}: {exc}") from exc


def category_for_url(url: str) -> str:
    """Infer a broad user-facing category from an official URL path."""
    path = urlparse(url).path.lower()
    if "fee" in path or "transport" in path:
        return "fees"
    if "admission" in path or "scholarship" in path:
        return "admission"
    if "hostel" in path or "facility" in path:
        return "hostel"
    if "placement" in path or "recruit" in path:
        return "placement"
    if any(term in path for term in ("exam", "result", "announcement", "academic-calendar", "happenings")):
        return "exams"
    if any(term in path for term in ("program", "course", "department", "syllabus")):
        return "courses"
    return "general"


def category_for_path(path: Path) -> str:
    """Infer a category from a user filename or parent folder."""
    text = " ".join(part.lower() for part in path.parts)
    for category in ("admission", "courses", "fees", "hostel", "placement", "exams"):
        if category in text or (category == "placement" and "placements" in text):
            return category
    return "general"
