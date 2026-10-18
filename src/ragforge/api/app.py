import inspect
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from ragforge.api.routes.health import router as health_router
from ragforge.api.routes.query import router as query_router
from ragforge.core.config import Settings, get_settings
from ragforge.core.logging import setup_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context manager handling startup and shutdown cleanup."""
    yield
    # Shutdown: Release any application-scoped closables
    closables = getattr(app.state, "_closables", [])
    for resource in closables:
        if hasattr(resource, "close"):
            try:
                res = resource.close()
                if inspect.isawaitable(res):
                    await res
            except Exception as exc:
                logger.warning("Error closing resource during application shutdown: %s", exc)

    vector_store = getattr(app.state, "vector_store", None)
    if (
        vector_store is not None
        and vector_store not in closables
        and hasattr(vector_store, "close")
    ):
        try:
            res = vector_store.close()
            if inspect.isawaitable(res):
                await res
        except Exception as exc:
            logger.warning("Error closing vector store during application shutdown: %s", exc)


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
        lifespan=lifespan,
    )

    app.state.settings = app_settings

    if settings is not None:
        app.dependency_overrides[get_settings] = lambda: app_settings

    # Register routes
    app.include_router(health_router)
    app.include_router(query_router, prefix=app_settings.api_v1_prefix)

    return app
