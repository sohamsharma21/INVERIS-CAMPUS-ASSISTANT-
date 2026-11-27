"""Robots-aware, rate-limited crawler limited to public Invertis HTML pages."""

from __future__ import annotations

import hashlib
import logging
import time
from collections import deque
from collections.abc import Iterator
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

from invertis_bot.config import Settings, get_settings
from invertis_bot.decorators import retry, timer
from invertis_bot.exceptions import ScrapingError
from invertis_bot.utils import DocumentRecord, write_json

logger = logging.getLogger(__name__)
USER_AGENT = "InvertisCampusAssistant/1.0 (educational RAG; respects robots.txt)"
SEED_PATHS = (
    "", "admission", "admission/fee-structure", "admission/scholarship", "admission/entrance-examinations",
    "admission/faq", "ug-programmes", "pg-programmes", "doctoral-programmes", "diploma-programmes",
    "placements/overview", "placements/placement-records", "placements/placement-policy",
    "happenings/announcements", "happenings/news", "academic/academic-calendar", "contact-us",
    "facilities/campus-facilities", "facilities/life-invertis",
)


class Scraper:
    """Lazily crawl public HTML pages on the university's own host."""

    def __init__(self, settings: Settings | None = None, session: requests.Session | None = None) -> None:
        self.settings = settings or get_settings()
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.base_url = self.settings.website_url.rstrip("/") + "/"
        self.allowed_host = urlparse(self.base_url).hostname
        self.cache_dir = self.settings.raw_dir / ".cache"
        self._last_request_at = 0.0
        self._robots = self._load_robots()

    def _load_robots(self) -> RobotFileParser:
        """Fetch robots.txt; fail closed if its policy cannot be verified."""
        robots_url = urljoin(self.base_url, "/robots.txt")
        parser = RobotFileParser(robots_url)
        try:
            response = self.session.get(
                robots_url,
                timeout=self.settings.request_timeout_seconds,
                allow_redirects=False,
            )
            response.raise_for_status()
            if response.is_redirect:
                raise ScrapingError("robots.txt redirected; refusing to crawl without a verified policy")
            parser.parse(response.text.splitlines())
        except ScrapingError:
            raise
        except requests.RequestException as exc:
            raise ScrapingError(f"Could not verify robots.txt; refusing to crawl: {exc}") from exc
        return parser

    def _allowed(self, url: str) -> bool:
        parsed = urlparse(url)
        return parsed.scheme in {"http", "https"} and parsed.hostname == self.allowed_host and self._robots.can_fetch(USER_AGENT, url)

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        wait_for = self.settings.request_delay_seconds - elapsed
        if wait_for > 0:
            time.sleep(wait_for)
        self._last_request_at = time.monotonic()

    def _cached_path(self, url: str) -> Path:
        return self.cache_dir / f"{hashlib.sha256(url.encode()).hexdigest()}.html"

    @retry(attempts=3, initial_delay=0.5, exceptions=(requests.RequestException,))
    def _get_html(self, url: str) -> str:
        """Return a cached page or fetch it with retries and respectful delay."""
        cached = self._cached_path(url)
        if cached.exists():
            try:
                return cached.read_text(encoding="utf-8")
            except OSError:
                logger.warning("Could not read crawler cache %s", cached)
        self._throttle()
        response = self.session.get(
            url,
            timeout=self.settings.request_timeout_seconds,
            allow_redirects=False,
        )
        response.raise_for_status()
        if response.is_redirect:
            logger.info("Skipping redirect instead of following it: %s", url)
            return ""
        if "text/html" not in response.headers.get("Content-Type", "text/html").lower():
            return ""
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(response.text, encoding="utf-8")
        return response.text

    @staticmethod
    def _extract(html: str, url: str) -> tuple[str, str, list[str]]:
        """Extract readable content and same-page outgoing links."""
        soup = BeautifulSoup(html, "html.parser")
        title = soup.title.get_text(" ", strip=True) if soup.title else url
        for element in soup.select("script, style, nav, footer, header, noscript, iframe, svg"):
            element.decompose()
        main = soup.find("main") or soup.find("article") or soup.body or soup
        text = main.get_text("\n", strip=True)
        links = [urljoin(url, anchor.get("href", "")) for anchor in soup.find_all("a", href=True)]
        return text, title, links

    def crawl(self) -> Iterator[DocumentRecord]:
        """Yield normalized page records lazily, bounded by ``max_pages``."""
        queue = deque(urljoin(self.base_url, path) for path in SEED_PATHS)
        visited: set[str] = set()
        requests_made = 0
        while queue and requests_made < self.settings.max_pages:
            candidate = urldefrag(queue.popleft())[0]
            parsed = urlparse(candidate)
            normalized = parsed._replace(query="", fragment="").geturl()
            if normalized in visited or not self._allowed(normalized):
                continue
            visited.add(normalized)
            try:
                requests_made += 1
                html = self._get_html(normalized)
                if not html:
                    continue
                text, title, links = self._extract(html, normalized)
                if text:
                    category = _category_for_url(normalized)
                    yield DocumentRecord(text, normalized, title, category, datetime.now(timezone.utc).isoformat())
                for link in links:
                    if len(visited) + len(queue) >= self.settings.max_pages * 3:
                        break
                    link_url = urldefrag(link)[0]
                    if urlparse(link_url).hostname == self.allowed_host:
                        queue.append(link_url)
            except requests.RequestException as exc:
                logger.warning("Skipping %s after request failure: %s", normalized, exc)
            except Exception as exc:
                logger.exception("Skipping malformed page %s: %s", normalized, exc)

    @timer
    def scrape_to_json(self, output_path: Path | None = None) -> list[DocumentRecord]:
        """Crawl eligible pages and persist clean text and source metadata to JSON."""
        records = list(self.crawl())
        target = output_path or self.settings.raw_dir / "university_pages.json"
        write_json(target, [asdict(item) for item in records])
        logger.info("Saved %d crawled pages to %s", len(records), target)
        return records


def _category_for_url(url: str) -> str:
    """Avoid an import cycle while categorizing scraped routes."""
    from invertis_bot.loaders import category_for_url
    return category_for_url(url)
