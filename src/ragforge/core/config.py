from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field
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
    groq_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("GROQ_API_KEY", "RAGFORGE_GROQ_API_KEY"),
        description="Groq API key for primary LLM generation",
    )
    openai_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_API_KEY", "RAGFORGE_OPENAI_API_KEY"),
        description="OpenAI API key for fallback LLM generation",
    )
    llm_provider: str = Field(
        default="groq",
        description="Primary LLM provider identifier (e.g. groq, openai)",
    )
    llm_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Primary LLM model identifier",
    )
    llm_fallback_provider: str | None = Field(
        default="openai",
        description="Optional fallback LLM provider identifier invoked on transient errors",
    )
    llm_fallback_model: str = Field(
        default="gpt-4o-mini",
        description="Fallback LLM model identifier",
    )
    llm_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Default sampling temperature for LLM generation",
    )
    llm_max_tokens: int = Field(
        default=1024,
        gt=0,
        description="Default maximum token count for generated completions",
    )
    llm_timeout: float = Field(
        default=30.0,
        gt=0.0,
        description="HTTP client timeout in seconds for LLM generation requests",
    )
    indexing_batch_size: int = Field(
        default=32,
        description="Default batch size for chunk embedding and upserting",
    )
    index_state_file: str = Field(
        default=".ragforge/index_state.json",
        description="Path to JSON file tracking document indexing state and content hashes",
    )


@lru_cache
def get_settings() -> Settings:
    """Returns a cached Settings instance."""
    return Settings()
