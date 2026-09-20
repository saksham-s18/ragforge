import pytest
from httpx import ASGITransport, AsyncClient

from ragforge.api.app import create_app
from ragforge.core.config import Settings


def test_create_app() -> None:
    """Verify application factory creates a FastAPI instance."""
    settings = Settings(env="test", debug=True)
    app = create_app(settings=settings)
    assert app.title == "ragforge"


@pytest.mark.asyncio
async def test_health_endpoint() -> None:
    """Verify GET /health returns HTTP 200 and expected payload."""
    settings = Settings(env="test", debug=True)
    app = create_app(settings=settings)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "service": "ragforge",
    }
