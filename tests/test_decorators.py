"""Tests for advanced function decorators without timing/network dependencies."""

import pytest

from invertis_bot.decorators import cache, log_calls, retry


def test_cache_reuses_result_and_supports_clear() -> None:
    calls = 0

    @cache(maxsize=2)
    def compute(value: int) -> int:
        nonlocal calls
        calls += 1
        return value * 2

    assert compute(3) == compute(3) == 6
    assert calls == 1
    compute.cache_clear()  # type: ignore[attr-defined]
    assert compute(3) == 6
    assert calls == 2


def test_retry_returns_after_transient_error(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = 0
    monkeypatch.setattr("invertis_bot.decorators.time.sleep", lambda _: None)

    @retry(attempts=3, initial_delay=0, exceptions=(ValueError,))
    def flaky() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ValueError("temporary")
        return "ok"

    assert flaky() == "ok"
    assert attempts == 3


def test_retry_reraises_after_last_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("invertis_bot.decorators.time.sleep", lambda _: None)

    @retry(attempts=2, initial_delay=0, exceptions=(ValueError,))
    def always_fails() -> None:
        raise ValueError("failed")

    with pytest.raises(ValueError, match="failed"):
        always_fails()


def test_log_calls_preserves_metadata() -> None:
    @log_calls
    def greet(name: str) -> str:
        """Say hello."""
        return f"hello {name}"

    assert greet.__name__ == "greet"
    assert greet.__doc__ == "Say hello."
    assert greet("Ada") == "hello Ada"
