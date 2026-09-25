import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip tests marked with 'integration' or 'benchmark' unless explicitly supplied."""
    markexpr = config.getoption("markexpr", default="")
    if "integration" not in markexpr:
        skip_integration = pytest.mark.skip(
            reason="Integration test skipped. Run with 'uv run pytest -m integration' to execute."
        )
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)

    if "benchmark" not in markexpr:
        skip_benchmark = pytest.mark.skip(
            reason="Benchmark test skipped. Run with 'uv run pytest -m benchmark' to execute."
        )
        for item in items:
            if "benchmark" in item.keywords:
                item.add_marker(skip_benchmark)
