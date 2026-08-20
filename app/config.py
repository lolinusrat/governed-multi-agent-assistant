"""Application settings, read once from the environment.

This is the only module that knows a provider API key exists. Everything else
takes an already-configured client.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.3-70b-versatile", alias="GROQ_MODEL")
    llm_provider: Literal["groq", "nebius", "stub"] = Field(default="stub", alias="LLM_PROVIDER")

    nebius_api_key: str = Field(default="", alias="NEBIUS_API_KEY")
    nebius_model: str = Field(default="meta-llama/Llama-3.3-70B-Instruct", alias="NEBIUS_MODEL")
    nebius_base_url: str = Field(
        default="https://api.studio.nebius.com/v1", alias="NEBIUS_BASE_URL"
    )

    retrieval_min_score: float = Field(default=0.15, ge=0.0, le=1.0, alias="RETRIEVAL_MIN_SCORE")
    retrieval_max_results: int = Field(default=6, ge=1, le=20, alias="RETRIEVAL_MAX_RESULTS")
    policy_dir: Path = Field(default=Path("data"), alias="POLICY_DIR")

    api_base_url: str = Field(default="http://127.0.0.1:8000", alias="API_BASE_URL")

    observability_enabled: bool = Field(default=True, alias="OBSERVABILITY_ENABLED")
    observability_file: Path = Field(default=Path("logs/events.jsonl"), alias="OBSERVABILITY_FILE")

    @property
    def observability_path(self) -> Path:
        """Absolute path to the JSONL event log."""
        return (
            self.observability_file
            if self.observability_file.is_absolute()
            else PROJECT_ROOT / self.observability_file
        )

    @property
    def policy_path(self) -> Path:
        """Absolute path to the policy corpus, resolved against the project root."""
        return self.policy_dir if self.policy_dir.is_absolute() else PROJECT_ROOT / self.policy_dir


@lru_cache
def get_settings() -> Settings:
    return Settings()
