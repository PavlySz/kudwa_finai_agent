"""
Base models and schemas for data validation
"""

from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class DataSource(str, Enum):
    """Enum for data sources"""

    QUICKBOOKS = "quickbooks"
    ROOTFI = "rootfi"


class BaseResponse(BaseModel):
    """Base response model for API responses"""

    success: bool
    message: Optional[str] = None
    data: Optional[Any] = None


class FinancialPeriod(BaseModel):
    """Model for financial period information"""

    start_date: datetime
    end_date: datetime
    period_type: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class FinancialLineItem(BaseModel):
    """Model for individual financial line items"""

    name: str
    value: float
    account_id: Optional[str] = None
    category: Optional[str] = None
    sub_items: Optional[List["FinancialLineItem"]] = None

    model_config = ConfigDict(from_attributes=True)


class FinancialRecord(BaseModel):
    """Base model for financial records"""

    id: Optional[int] = None
    source: DataSource
    company_id: Optional[str] = None
    period: FinancialPeriod
    currency: str = "USD"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class HealthCheck(BaseModel):
    """Health check response model"""

    status: str
    timestamp: datetime
    version: str
    database: bool
    message: str
