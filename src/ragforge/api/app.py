from fastapi import FastAPI

from ragforge.api.routes.health import router as health_router
from ragforge.core.config import Settings, get_settings
from ragforge.core.logging import setup_logging


def create_app(settings: Settings | None = None) -> FastAPI:
    """Application factory that configures and returns a FastAPI instance."""
    app_settings = settings or get_settings()

    setup_logging(app_settings)

    app = FastAPI(
        title=app_settings.app_name,
        version="0.1.0",
        description="Production-oriented Agentic RAG Platform",
        docs_url="/docs" if app_settings.debug or app_settings.env != "production" else None,
        redoc_url="/redoc" if app_settings.debug or app_settings.env != "production" else None,
    )

    # Register routes
    app.include_router(health_router)

    return app
