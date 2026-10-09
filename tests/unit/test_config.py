import pytest

from ragforge.core.config import Settings, get_settings


def test_settings_load_defaults() -> None:
    """Verify that default settings instantiate cleanly with standard defaults."""
    settings = Settings()
    assert settings.app_name == "ragforge"
    assert settings.env == "development"
    assert settings.debug is False
    assert settings.log_level == "INFO"
    assert settings.host == "0.0.0.0"
    assert settings.port == 8000
    assert settings.api_v1_prefix == "/api/v1"
    assert settings.embedding_provider == "deterministic"
    assert settings.embedding_model == "BAAI/bge-small-en-v1.5"
    assert settings.llm_provider == "groq"
    assert settings.database_url is None
    assert settings.qdrant_url == "http://localhost:6333"
    assert settings.qdrant_api_key is None
    assert settings.qdrant_collection == "ragforge_chunks"
    assert settings.qdrant_vector_dimension == 64
    assert settings.reranking_enabled is False
    assert settings.reranker_provider == "deterministic"
    assert settings.rerank_candidate_k == 20
    assert settings.rerank_top_k == 5


def test_get_settings_is_cached() -> None:
    """Verify that get_settings uses lru_cache."""
    settings_first = get_settings()
    settings_second = get_settings()
    assert settings_first is settings_second


def test_settings_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that RAGFORGE_ prefixed environment variables override defaults."""
    monkeypatch.setenv("RAGFORGE_APP_NAME", "custom_ragforge")
    monkeypatch.setenv("RAGFORGE_ENV", "test")
    monkeypatch.setenv("RAGFORGE_DEBUG", "true")
    monkeypatch.setenv("RAGFORGE_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("RAGFORGE_QDRANT_URL", "http://qdrant.internal:6333")
    monkeypatch.setenv("RAGFORGE_QDRANT_COLLECTION", "custom_collection")
    monkeypatch.setenv("RAGFORGE_QDRANT_VECTOR_DIMENSION", "128")
    monkeypatch.setenv("RAGFORGE_EMBEDDING_PROVIDER", "fastembed")
    monkeypatch.setenv("RAGFORGE_EMBEDDING_MODEL", "custom-model")
    monkeypatch.setenv("RAGFORGE_EMBEDDING_CACHE_DIR", "/tmp/models")

    settings = Settings()
    assert settings.app_name == "custom_ragforge"
    assert settings.env == "test"
    assert settings.debug is True
    assert settings.log_level == "DEBUG"
    assert settings.qdrant_url == "http://qdrant.internal:6333"
    assert settings.qdrant_collection == "custom_collection"
    assert settings.qdrant_vector_dimension == 128
    assert settings.embedding_provider == "fastembed"
    assert settings.embedding_model == "custom-model"
    assert settings.embedding_cache_dir == "/tmp/models"


def test_settings_reranking_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that reranking configuration can be overridden via environment variables."""
    monkeypatch.setenv("RAGFORGE_RERANKING_ENABLED", "true")
    monkeypatch.setenv("RAGFORGE_RERANKER_PROVIDER", "custom_provider")
    monkeypatch.setenv("RAGFORGE_RERANK_CANDIDATE_K", "35")
    monkeypatch.setenv("RAGFORGE_RERANK_TOP_K", "8")

    settings = Settings()
    assert settings.reranking_enabled is True
    assert settings.reranker_provider == "custom_provider"
    assert settings.rerank_candidate_k == 35
    assert settings.rerank_top_k == 8
