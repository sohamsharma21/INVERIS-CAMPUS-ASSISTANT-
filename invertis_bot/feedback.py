"""SQLite-backed user feedback and privacy-conscious query analytics."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(slots=True, frozen=True)
class FeedbackEntry:
    """One rating attached to a generated answer."""

    question: str
    rating: str
    created_at: str
    answer_excerpt: str


class FeedbackStore:
    """Persist feedback and aggregate popular questions with parameterized SQL."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    answer_excerpt TEXT NOT NULL,
                    rating TEXT NOT NULL CHECK (rating IN ('up', 'down')),
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS query_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
            """)

    def add_feedback(self, question: str, answer: str, rating: str) -> None:
        """Record thumbs-up/down feedback, storing only a short answer excerpt."""
        if rating not in {"up", "down"}:
            raise ValueError("rating must be 'up' or 'down'")
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO feedback(question, answer_excerpt, rating, created_at) VALUES (?, ?, ?, ?)",
                (question[:2000], answer[:500], rating, datetime.now(timezone.utc).isoformat()),
            )

    def log_question(self, question: str) -> None:
        """Record a question for aggregated popularity reporting."""
        with self._connect() as connection:
            connection.execute("INSERT INTO query_log(question, created_at) VALUES (?, ?)", (question[:2000], datetime.now(timezone.utc).isoformat()))

    def recent_feedback(self, limit: int = 100) -> list[FeedbackEntry]:
        """Return recent feedback without exposing internal database objects."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT question, rating, created_at, answer_excerpt FROM feedback ORDER BY id DESC LIMIT ?",
                (max(1, min(limit, 500)),),
            ).fetchall()
        return [FeedbackEntry(row["question"], row["rating"], row["created_at"], row["answer_excerpt"]) for row in rows]

    def most_asked(self, limit: int = 10) -> list[tuple[str, int]]:
        """Return frequently repeated normalized question strings."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT question, COUNT(*) AS count FROM query_log GROUP BY lower(trim(question)) ORDER BY count DESC LIMIT ?",
                (max(1, min(limit, 100)),),
            ).fetchall()
        return [(row["question"], int(row["count"])) for row in rows]
