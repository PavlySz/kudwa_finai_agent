"""
Database configuration and connection management
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.ext.asyncio import AsyncEngine

from src.config.settings import settings

# Create async engine
engine: AsyncEngine = create_async_engine(
    settings.DATABASE_URL, echo=settings.DEBUG, future=True
)

# Create async session factory
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# Create declarative base for models
Base = declarative_base()


async def init_db():
    """
    Initialize database tables
    """
    async with engine.begin() as conn:
        # Import all models here to ensure they're registered
        from src.models import base

        # Create all tables
        await conn.run_sync(Base.metadata.create_all)
        print("Database initialized successfully")


async def get_db():
    """
    Dependency to get database session
    """
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()
