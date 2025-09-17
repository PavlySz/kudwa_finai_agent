"""
Application configuration and settings

Exact model names:
```
Claude-4-Sonnet: claude-sonnet-4-20250514
Claude-4.1-Opus: claude-opus-4-1-20250805
GPT-5: gpt-5
Gemini-2.5-Pro: gemini-2.5-pro
```

API keys are stored in a file called `keys.env`
"""

from pydantic_settings import BaseSettings
from typing import Optional, Literal
from enum import Enum
from pydantic import computed_field


class ModelName(str, Enum):
    """Available LLM models"""

    GPT5 = "gpt-5"
    CLAUDE_SONNET = "claude-sonnet-4-20250514"
    CLAUDE_OPUS = "claude-opus-4-1-20250805"


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables
    """

    # Application
    APP_NAME: str = "Financial AI Backend"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 5000

    # Deployment Configuration
    USE_REPLIT: bool = False  # Set to True when deploying on Replit

    @computed_field
    @property
    def API_URL(self) -> str:
        return (
            "https://pavly-kudwa-finai-agent.replit.app"
            if self.USE_REPLIT
            else "http://localhost:8000"
        )

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./financial_data.db"

    # AI/LLM Configuration
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None  # Just in case

    # Model Selection (easily switchable!)
    DEFAULT_MODEL: ModelName = ModelName.GPT5
    SIMPLE_QUERY_MODEL: ModelName = ModelName.GPT5
    COMPLEX_QUERY_MODEL: ModelName = ModelName.CLAUDE_SONNET
    VERIFICATION_MODEL: ModelName = ModelName.CLAUDE_OPUS  # For evaluation/verification

    # Model routing thresholds
    COMPLEXITY_THRESHOLD_SIMPLE: int = 3
    COMPLEXITY_THRESHOLD_COMPLEX: int = 7

    class Config:
        env_file = "keys.env"  # Load from keys.env instead of .env
        case_sensitive = True


# Create settings instance
settings = Settings()
