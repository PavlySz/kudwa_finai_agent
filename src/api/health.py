"""
Health check endpoints for system monitoring
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from datetime import datetime

from src.config.database import get_db
from src.config.settings import settings
from src.models.base import HealthCheck

router = APIRouter(prefix="/api/health")


@router.get("/", response_model=HealthCheck)
async def health_check(db: AsyncSession = Depends(get_db)):
    """
    System health check endpoint

    Returns system status including database connectivity
    """
    # Check database connection
    db_status = False
    try:
        await db.execute(text("SELECT 1"))
        db_status = True
    except Exception as e:
        print(f"Database health check failed: {e}")

    return HealthCheck(
        status="healthy" if db_status else "unhealthy",
        timestamp=datetime.utcnow(),
        version=settings.APP_VERSION,
        database=db_status,
        message=(
            "All systems operational" if db_status else "Database connection failed"
        ),
    )


@router.get("/ping")
async def ping():
    """
    Simple ping endpoint for basic connectivity check
    """
    return {"ping": "pong", "timestamp": datetime.utcnow()}
