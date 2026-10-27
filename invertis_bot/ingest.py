"""Command-line ingestion: crawl/load sources, chunk them, and build Chroma."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Sequence

from langchain_core.documents import Document

from invertis_bot.config import Settings, get_settings
from invertis_bot.exceptions import DocumentLoadError, ScrapingError
from invertis_bot.loaders import PDFLoader
from invertis_bot.processing import DocumentProcessor
from invertis_bot.scraper import Scraper
from invertis_bot.utils import timed_block
from invertis_bot.vectorstore import VectorStoreManager

logger = logging.getLogger(__name__)


def load_scraped_json(path: Path) -> list[Document]:
    """Convert the scraper's JSON records into LangChain source documents."""
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [Document(page_content=item["page_content"], metadata={key: value for key, value in item.items() if key != "page_content"}) for item in payload if item.get("page_content", "").strip()]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise DocumentLoadError(f"Could not read scraped data file {path}: {exc}") from exc


def load_local_documents(directory: Path) -> list[Document]:
    """Load supported PDF and text files from the user's documents directory."""
    documents: list[Document] = []
    directory.mkdir(parents=True, exist_ok=True)
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        try:
            if path.suffix.lower() == ".pdf":
                documents.extend(PDFLoader(path).load())
            elif path.suffix.lower() in {".txt", ".md"}:
                text = path.read_text(encoding="utf-8")
                if text.strip():
                    from invertis_bot.loaders import category_for_path
                    documents.append(Document(page_content=text, metadata={"url": str(path.resolve()), "title": path.name, "category": category_for_path(path)}))
            else:
                logger.debug("Skipping unsupported document %s", path)
        except DocumentLoadError as exc:
            logger.warning("Skipping unreadable document %s: %s", path, exc)
    return documents


def build_index(*, rebuild: bool = True, scrape: bool = True, refresh: bool = False, settings: Settings | None = None) -> int:
    """Collect sources and persist all chunks in the configured Chroma index."""
    config = settings or get_settings()
    config.raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = config.raw_dir / "university_pages.json"
    if refresh:
        shutil.rmtree(config.raw_dir / ".cache", ignore_errors=True)
    if scrape:
        try:
            with timed_block("Official site crawl"):
                Scraper(config).scrape_to_json(raw_path)
        except ScrapingError as exc:
            logger.warning("Website crawl failed; will use existing/raw local sources: %s", exc)
    sources = load_scraped_json(raw_path) + load_local_documents(config.documents_dir)
    if not sources:
        raise DocumentLoadError("No source content found. Check network access or add files under data/documents/.")
    processor = DocumentProcessor(config.chunk_size, config.chunk_overlap)
    chunks = list(processor.chunks(sources))
    if not chunks:
        raise DocumentLoadError("All source files were empty after text cleanup.")
    with timed_block("Embedding and Chroma indexing"):
        return VectorStoreManager(config).build(chunks, rebuild=rebuild)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; supports the requested ``--rebuild`` command."""
    parser = argparse.ArgumentParser(description="Build the Invertis Campus Assistant knowledge index")
    parser.add_argument("--rebuild", action="store_true", help="Replace the existing persistent index")
    parser.add_argument("--no-scrape", action="store_true", help="Build only from already-scraped JSON and local documents")
    parser.add_argument("--refresh-cache", action="store_true", help="Clear cached HTML before crawling the website")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        count = build_index(rebuild=args.rebuild, scrape=not args.no_scrape, refresh=args.refresh_cache)
        print(f"Indexed {count} chunks successfully.")
        return 0
    except Exception as exc:
        logger.exception("Index build failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
