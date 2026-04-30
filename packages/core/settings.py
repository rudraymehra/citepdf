"""Centralized settings loaded from .env via pydantic-settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    anthropic_api_key: str = Field(..., alias="ANTHROPIC_API_KEY")
    # Default to Sonnet for generation: 3-5x faster than Opus, marginally lower
    # quality. Override with ANTHROPIC_GENERATION_MODEL=claude-opus-4-7 for max
    # quality at the cost of ~3x latency.
    anthropic_generation_model: str = Field(
        "claude-sonnet-4-6", alias="ANTHROPIC_GENERATION_MODEL"
    )
    # Default to Haiku for the many-call paths (rewrite, judge, contextual headers).
    anthropic_judge_model: str = Field(
        "claude-haiku-4-5", alias="ANTHROPIC_JUDGE_MODEL"
    )

    qdrant_url: str = Field("http://localhost:6333", alias="QDRANT_URL")
    qdrant_api_key: str | None = Field(None, alias="QDRANT_API_KEY")

    redis_url: str = Field("redis://localhost:6379", alias="REDIS_URL")

    hf_home: str = Field("./.cache/huggingface", alias="HF_HOME")

    api_host: str = Field("0.0.0.0", alias="API_HOST")
    api_port: int = Field(8000, alias="API_PORT")
    api_base_url: str = Field("http://localhost:8000", alias="API_BASE_URL")
    streamlit_port: int = Field(8501, alias="STREAMLIT_PORT")

    data_dir: str = Field("./eval/data", alias="DATA_DIR")

    topk_prefetch: int = Field(50, alias="TOPK_PREFETCH")
    topk_final: int = Field(8, alias="TOPK_FINAL")
    oos_centroid_threshold: float = Field(0.20, alias="OOS_CENTROID_THRESHOLD")
    oos_top1_threshold: float = Field(0.10, alias="OOS_TOP1_THRESHOLD")
    faithfulness_threshold: float = Field(0.85, alias="FAITHFULNESS_THRESHOLD")

    refusal_string: str = Field(
        "I cannot answer this from the provided document.", alias="REFUSAL_STRING"
    )

    enable_contextual_headers: bool = Field(True, alias="ENABLE_CONTEXTUAL_HEADERS")
    enable_raptor: bool = Field(True, alias="ENABLE_RAPTOR")
    raptor_max_levels: int = Field(3, alias="RAPTOR_MAX_LEVELS")
    deep_mode_multiquery_n: int = Field(3, alias="DEEP_MODE_MULTIQUERY_N")

    # Chainlit persistence (ChatGPT-style sidebar with past threads).
    # Empty string disables persistence (in-memory only).
    chainlit_db_url: str = Field(
        "sqlite+aiosqlite:///./data/chainlit.db", alias="CHAINLIT_DB_URL"
    )
    chainlit_auth_secret: str = Field(
        "", alias="CHAINLIT_AUTH_SECRET"
    )

    @property
    def data_path(self) -> Path:
        p = Path(self.data_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
