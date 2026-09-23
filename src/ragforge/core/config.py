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
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Qdrant vector database endpoint URL",
    )
    qdrant_api_key: str | None = Field(
        default=None,
        description="Optional API key for Qdrant Cloud or protected instances",
    )
    qdrant_collection: str = Field(
        default="ragforge_chunks",
        description="Default Qdrant collection name for chunk storage",
    )
    qdrant_vector_dimension: int = Field(
        default=64,
        description="Configured vector dimensionality for Qdrant collection",
    )
    embedding_provider: str = Field(
        default="deterministic",
        description="Embedding provider identifier (e.g. deterministic, fastembed)",
    )
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="Embedding model name or identifier (e.g. BAAI/bge-small-en-v1.5)",
    )
    embedding_cache_dir: str | None = Field(
        default=None,
        description="Optional local cache directory for embedding models",
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
