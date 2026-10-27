"""Shared data types, safe I/O helpers, and small context managers."""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, TextIO

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DocumentRecord:
    """Normalized source content before chunking."""

    page_content: str
    url: str
    title: str
    category: str = "general"
    scraped_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    page: int | None = None


@dataclass(slots=True, frozen=True)
class Source:
    """A human-readable citation for a retrieved passage."""

    title: str
    location: str
    category: str
    page: int | None = None


@contextmanager
def timed_block(name: str) -> Iterator[None]:
    """Context manager that logs duration even if its block raises."""
    started = time.perf_counter()
    try:
        yield
    finally:
        logger.info("%s took %.3fs", name, time.perf_counter() - started)


@contextmanager
def safe_text_file(path: Path, mode: str = "r", *, encoding: str = "utf-8") -> Iterator[TextIO]:
    """Open a UTF-8 text file and ensure parent directories exist for writes."""
    if any(flag in mode for flag in ("w", "a", "x", "+")):
        path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open(mode, encoding=encoding)
    try:
        yield handle
    finally:
        handle.close()


def write_json(path: Path, payload: Any) -> None:
    """Atomically serialize JSON by writing a temporary sibling first."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with safe_text_file(temporary, "w") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def record_to_dict(record: DocumentRecord) -> dict[str, Any]:
    """Convert a dataclass record into a JSON-ready mapping."""
    return asdict(record)
