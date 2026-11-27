from datetime import datetime, timedelta, timezone

from invertis_bot.config import Settings
from invertis_bot.ingest import read_schedule_status, write_schedule_status


def test_schedule_status_round_trip(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path, raw_dir=tmp_path / "raw")
    write_schedule_status(
        settings,
        last_success_at=datetime.now(timezone.utc).isoformat(),
        chunks=42,
        status="success",
    )
    status = read_schedule_status(settings)
    assert status["chunks"] == 42
    assert status["status"] == "success"
    assert "updated_at" in status


def test_schedule_status_handles_missing_file(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path, raw_dir=tmp_path / "missing")
    assert read_schedule_status(settings) == {}
