"""
Financial AI Backend - Main Application Entry Point
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from src.config.settings import settings
from src.config.database import init_db
from src.api import health, data_import, ai_queries, financial, evaluation


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events
    """
    # Startup
    print("Starting Financial AI Backend...")
    await init_db()
    yield
    # Shutdown
    print("Shutting down Financial AI Backend...")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI-powered financial data processing and analysis system",
    lifespan=lifespan,
)

# Configure CORS (restrictive for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.DEBUG else [],  # Restrict in production
    allow_credentials=False,  # Disabled for security
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["health"])
app.include_router(data_import.router, tags=["data"])
app.include_router(ai_queries.router, tags=["ai"])
app.include_router(financial.router, tags=["financial"])
app.include_router(evaluation.router, tags=["evaluation"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Financial AI Backend API",
        "version": settings.APP_VERSION,
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG
    )
