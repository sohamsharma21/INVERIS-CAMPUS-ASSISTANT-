"""Command-line ingestion: crawl/load sources, chunk them, and build Chroma."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from datetime import datetime, timezone
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


def schedule_status_path(settings: Settings) -> Path:
    return settings.raw_dir / "schedule_status.json"


def read_schedule_status(settings: Settings) -> dict[str, object]:
    path = schedule_status_path(settings)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def write_schedule_status(settings: Settings, **values: object) -> None:
    status = read_schedule_status(settings)
    status.update(values)
    status["updated_at"] = datetime.now(timezone.utc).isoformat()
    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    path = schedule_status_path(settings)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


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
    parser.add_argument("--scheduled", action="store_true", help="Run once only if the configured crawl interval has elapsed")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    if args.scheduled:
        if not settings.enable_crawl_schedule:
            print("Scheduled crawling is disabled. Set ENABLE_CRAWL_SCHEDULE=true.")
            return 0
        status = read_schedule_status(settings)
        last_run = status.get("last_success_at")
        if isinstance(last_run, str):
            try:
                elapsed = datetime.now(timezone.utc) - datetime.fromisoformat(last_run)
                if elapsed.total_seconds() < settings.crawl_schedule_interval_hours * 3600:
                    print("Scheduled crawl skipped: configured interval has not elapsed.")
                    return 0
            except ValueError:
                pass
    try:
        count = build_index(rebuild=args.rebuild, scrape=not args.no_scrape, refresh=args.refresh_cache, settings=settings)
        if args.scheduled:
            write_schedule_status(
                settings,
                last_success_at=datetime.now(timezone.utc).isoformat(),
                chunks=count,
                status="success",
            )
        print(f"Indexed {count} chunks successfully.")
        return 0
    except Exception as exc:
        if args.scheduled:
            write_schedule_status(settings, status="failed", error=str(exc))
        logger.exception("Index build failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
