"""Typed application settings loaded from environment variables and ``.env``."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Environment-backed configuration; safe defaults support offline tests."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    google_api_key: SecretStr | None = Field(default=None, alias="GOOGLE_API_KEY")
    enable_gemini: bool = Field(default=True, alias="ENABLE_GEMINI")
    chat_model: str = Field(default="gemini-3.8-flash", alias="CHAT_MODEL")
    embedding_model: str = Field(default="gemini-embedding-001", alias="EMBEDDING_MODEL")
    admin_password: SecretStr | None = Field(default=None, alias="ADMIN_PASSWORD")
    website_url: str = Field(default="https://www.invertisuniversity.ac.in/", alias="WEBSITE_URL")
    data_dir: Path = Field(default=PROJECT_ROOT / "data", alias="DATA_DIR")
    index_dir: Path = Field(default=PROJECT_ROOT / "data" / "index", alias="INDEX_DIR")
    raw_dir: Path = Field(default=PROJECT_ROOT / "data" / "raw", alias="RAW_DIR")
    documents_dir: Path = Field(default=PROJECT_ROOT / "data" / "documents", alias="DOCUMENTS_DIR")
    request_delay_seconds: float = Field(default=1.0, ge=0.0, alias="REQUEST_DELAY_SECONDS")
    request_timeout_seconds: float = Field(default=20.0, gt=0.0, alias="REQUEST_TIMEOUT_SECONDS")
    max_pages: int = Field(default=120, ge=1, alias="MAX_PAGES")
    chunk_size: int = Field(default=1000, ge=100, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=160, ge=0, alias="CHUNK_OVERLAP")
    retrieval_top_k: int = Field(default=5, ge=1, alias="RETRIEVAL_TOP_K")
    retrieval_score_threshold: float = Field(default=0.22, ge=0.0, le=1.0, alias="RETRIEVAL_SCORE_THRESHOLD")
    memory_turns: int = Field(default=5, ge=1, alias="MEMORY_TURNS")
    feedback_db: Path = Field(default=PROJECT_ROOT / "data" / "feedback.sqlite3", alias="FEEDBACK_DB")

    @property
    def api_key(self) -> str | None:
        """Return the API key without exposing it through repr output."""
        return self.google_api_key.get_secret_value() if self.google_api_key else None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a singleton settings instance for the current process."""
    return Settings()
