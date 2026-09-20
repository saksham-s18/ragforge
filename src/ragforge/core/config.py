from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_prefix="RAGFORGE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Core Application Settings
    app_name: str = "ragforge"
    env: Literal["development", "staging", "production", "test"] = "development"
    debug: bool = False
    log_level: str = "INFO"
    api_v1_prefix: str = "/api/v1"

    # Server Settings
    host: str = "0.0.0.0"
    port: int = 8000

    # Placeholders for future services (optional defaults, no hardcoded secrets)
    database_url: str | None = Field(
        default=None,
        description="PostgreSQL async connection string",
    )
    qdrant_url: str | None = Field(
        default=None,
        description="Qdrant vector database endpoint URL",
    )
    qdrant_api_key: str | None = Field(
        default=None,
        description="Optional API key for Qdrant Cloud or protected instances",
    )
    embedding_provider: str = Field(
        default="openai",
        description="Embedding provider identifier (e.g. openai, fastembed)",
    )
    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key placeholder",
    )
    llm_provider: str = Field(
        default="openai",
        description="Primary LLM provider identifier (e.g. openai, anthropic, ollama)",
    )


@lru_cache
def get_settings() -> Settings:
    """Returns a cached Settings instance."""
    return Settings()
