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
    assert settings.hybrid_enabled is False
    assert settings.fusion_strategy == "rrf"
    assert settings.rrf_k == 60
    assert settings.rrf_dense_weight == 1.0
    assert settings.rrf_lexical_weight == 1.0
    assert settings.hybrid_candidate_k == 20


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
    monkeypatch.setenv("RAGFORGE_HYBRID_ENABLED", "true")
    monkeypatch.setenv("RAGFORGE_FUSION_STRATEGY", "rrf")
    monkeypatch.setenv("RAGFORGE_RRF_K", "45")
    monkeypatch.setenv("RAGFORGE_RRF_DENSE_WEIGHT", "1.5")
    monkeypatch.setenv("RAGFORGE_RRF_LEXICAL_WEIGHT", "0.8")
    monkeypatch.setenv("RAGFORGE_HYBRID_CANDIDATE_K", "30")

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
    assert settings.hybrid_enabled is True
    assert settings.fusion_strategy == "rrf"
    assert settings.rrf_k == 45
    assert settings.rrf_dense_weight == 1.5
    assert settings.rrf_lexical_weight == 0.8
    assert settings.hybrid_candidate_k == 30


@pytest.mark.parametrize("invalid_k", [0, -1, -60])
def test_settings_invalid_rrf_k(invalid_k: int) -> None:
    """Verify that rrf_k must be strictly positive."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Settings(rrf_k=invalid_k)


@pytest.mark.parametrize("invalid_weight", [-0.1, -1.0])
def test_settings_invalid_dense_weight(invalid_weight: float) -> None:
    """Verify that rrf_dense_weight must be non-negative."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Settings(rrf_dense_weight=invalid_weight)


@pytest.mark.parametrize("invalid_weight", [-0.1, -1.0])
def test_settings_invalid_lexical_weight(invalid_weight: float) -> None:
    """Verify that rrf_lexical_weight must be non-negative."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Settings(rrf_lexical_weight=invalid_weight)


@pytest.mark.parametrize("invalid_candidate_k", [0, -1, -20])
def test_settings_invalid_hybrid_candidate_k(invalid_candidate_k: int) -> None:
    """Verify that hybrid_candidate_k must be strictly positive."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Settings(hybrid_candidate_k=invalid_candidate_k)


@pytest.mark.parametrize("invalid_strategy", ["weighted", "linear", "custom", ""])
def test_settings_unsupported_fusion_strategy(invalid_strategy: str) -> None:
    """Verify that unsupported fusion strategies fail validation clearly."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as exc_info:
        Settings(fusion_strategy=invalid_strategy)
    assert "unsupported fusion strategy" in str(exc_info.value).lower()
