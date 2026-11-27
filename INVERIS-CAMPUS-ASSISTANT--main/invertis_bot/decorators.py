"""Reusable decorators for timing, retries, logging, and bounded caching."""

from __future__ import annotations

import functools
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar, cast

P = ParamSpec("P")
R = TypeVar("R")
logger = logging.getLogger(__name__)


def timer(function: Callable[P, R]) -> Callable[P, R]:
    """Log elapsed execution time while preserving the wrapped callable."""
    @functools.wraps(function)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        started = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            logger.info("%s completed in %.3fs", function.__qualname__, time.perf_counter() - started)
    return wrapper


def log_calls(function: Callable[P, R]) -> Callable[P, R]:
    """Log a function's name and elapsed duration without logging argument values."""
    @functools.wraps(function)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        logger.debug("Calling %s", function.__qualname__)
        result = function(*args, **kwargs)
        logger.debug("%s returned successfully", function.__qualname__)
        return result
    return wrapper


def retry(
    attempts: int = 3,
    initial_delay: float = 0.5,
    backoff: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Retry transient failures with exponential backoff; use narrowly in network calls."""
    if attempts < 1 or initial_delay < 0 or backoff < 1:
        raise ValueError("attempts must be positive, delay non-negative, and backoff at least one")

    def decorate(function: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(function)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            delay = initial_delay
            for attempt in range(1, attempts + 1):
                try:
                    return function(*args, **kwargs)
                except exceptions:
                    if attempt == attempts:
                        raise
                    logger.warning("%s failed (attempt %d/%d); retrying", function.__qualname__, attempt, attempts)
                    time.sleep(delay)
                    delay *= backoff
            raise RuntimeError("unreachable retry state")
        return wrapper
    return decorate


def cache(maxsize: int = 128) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Memoize calls in a bounded, lock-protected LRU cache (hashable args required)."""
    if maxsize < 1:
        raise ValueError("maxsize must be at least one")

    def decorate(function: Callable[P, R]) -> Callable[P, R]:
        values: OrderedDict[Any, R] = OrderedDict()
        lock = threading.RLock()

        @functools.wraps(function)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            key = (args, tuple(sorted(kwargs.items())))
            with lock:
                if key in values:
                    values.move_to_end(key)
                    return values[key]
            result = function(*args, **kwargs)
            with lock:
                # Another thread may have populated the same key while this call ran.
                if key not in values:
                    values[key] = result
                if len(values) > maxsize:
                    values.popitem(last=False)
                else:
                    values.move_to_end(key)
                return values[key]

        def clear() -> None:
            with lock:
                values.clear()

        setattr(wrapper, "cache_clear", clear)
        return cast(Callable[P, R], wrapper)
    return decorate
