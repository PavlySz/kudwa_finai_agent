"""
Database models for financial data
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, Enum
from sqlalchemy.orm import relationship
from datetime import datetime

from src.config.database import Base
from src.models.base import DataSource


class Company(Base):
    """Company/Organization model"""

    __tablename__ = "companies"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    external_id = Column(String(255), index=True)  # Platform-specific ID
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    financial_records = relationship("FinancialRecord", back_populates="company")


class FinancialRecord(Base):
    """Unified financial record model"""

    __tablename__ = "financial_records"

    id = Column(Integer, primary_key=True, index=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=False)
    source = Column(Enum(DataSource), nullable=False, index=True)

    # Period information
    period_start = Column(DateTime, nullable=False, index=True)
    period_end = Column(DateTime, nullable=False, index=True)
    period_type = Column(String(50))  # monthly, quarterly, yearly

    # Financial data
    currency = Column(String(10), default="USD")

    # Metadata
    platform_id = Column(String(255))  # Original platform record ID
    raw_data = Column(Text)  # Store original JSON for reference
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    company = relationship("Company", back_populates="financial_records")
    line_items = relationship(
        "FinancialLineItem",
        back_populates="financial_record",
        cascade="all, delete-orphan",
    )


class FinancialLineItem(Base):
    """Individual financial line items (revenue, expenses, etc.)"""

    __tablename__ = "financial_line_items"

    id = Column(Integer, primary_key=True, index=True)
    financial_record_id = Column(
        Integer, ForeignKey("financial_records.id"), nullable=False
    )

    # Line item details
    category = Column(String(100), nullable=False, index=True)  # revenue, expense, etc.
    name = Column(String(255), nullable=False)
    value = Column(Float, nullable=False)
    account_id = Column(String(255))  # Original account ID from source

    # Hierarchy
    parent_id = Column(Integer, ForeignKey("financial_line_items.id"))

    # Relationships
    financial_record = relationship("FinancialRecord", back_populates="line_items")
    parent = relationship("FinancialLineItem", remote_side=[id])
    children = relationship(
        "FinancialLineItem", back_populates="parent", cascade="all, delete-orphan"
    )
