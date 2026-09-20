from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["Health"])


class HealthResponse(BaseModel):
    """Response payload for health check endpoint."""

    status: str = "healthy"
    service: str = "ragforge"


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Returns the operational status of the service."""
    return HealthResponse(status="healthy", service="ragforge")
