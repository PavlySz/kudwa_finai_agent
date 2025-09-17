"""
RESTful API endpoints for direct financial data access.
Provides programmatic access without AI interpretation.
"""

from typing import List, Optional, Dict, Any
from datetime import date, timedelta
from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
from sqlalchemy.orm import selectinload

from src.config.database import get_db
from src.models.financial import Company, FinancialRecord, FinancialLineItem
from src.models.base import DataSource
from pydantic import BaseModel, Field


router = APIRouter(prefix="/api/financial", tags=["Financial Data"])


# Response models
class FinancialRecordResponse(BaseModel):
    """Response model for financial records"""

    id: int
    company_id: int
    company_name: str
    period_start: date
    period_end: date
    data_source: str
    total_revenue: Decimal = Field(default=Decimal("0"))
    total_expenses: Decimal = Field(default=Decimal("0"))
    net_income: Decimal = Field(default=Decimal("0"))
    line_items_count: int = 0

    class Config:
        from_attributes = True


class LineItemResponse(BaseModel):
    """Response model for financial line items"""

    id: int
    record_id: int
    name: str
    category: str
    value: Decimal
    percentage_of_revenue: Optional[Decimal] = None

    class Config:
        from_attributes = True


class CategorySummary(BaseModel):
    """Summary of a financial category"""

    category: str
    total_value: Decimal
    item_count: int
    percentage_of_total: Optional[Decimal] = None


class FinancialSummaryResponse(BaseModel):
    """Aggregated financial summary"""

    company_id: Optional[int]
    company_name: Optional[str]
    period: str
    total_revenue: Decimal
    total_expenses: Decimal
    net_income: Decimal
    profit_margin: Optional[Decimal]
    categories: List[CategorySummary]


class TrendDataPoint(BaseModel):
    """Single point in a trend series"""

    period: str
    value: Decimal
    change_from_previous: Optional[Decimal] = None
    percentage_change: Optional[Decimal] = None


@router.get("/records", response_model=List[FinancialRecordResponse])
async def get_financial_records(
    company_id: Optional[int] = Query(None, description="Filter by company ID"),
    start_date: Optional[date] = Query(None, description="Start date (inclusive)"),
    end_date: Optional[date] = Query(None, description="End date (inclusive)"),
    data_source: Optional[str] = Query(None, description="Filter by data source"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum records to return"),
    offset: int = Query(0, ge=0, description="Number of records to skip"),
    session: AsyncSession = Depends(get_db),
) -> List[FinancialRecordResponse]:
    """
    Get financial records with optional filtering.

    Query parameters:
    - company_id: Filter by specific company
    - start_date: Records with period_end >= this date
    - end_date: Records with period_start <= this date
    - data_source: Filter by data source (e.g., 'QuickBooks', 'Rootfi')
    - limit: Maximum number of records (default 100, max 1000)
    - offset: Skip N records for pagination
    """
    # Build query
    query = select(FinancialRecord).options(selectinload(FinancialRecord.company))

    # Apply filters
    conditions = []
    if company_id:
        conditions.append(FinancialRecord.company_id == company_id)
    if start_date:
        conditions.append(FinancialRecord.period_end >= start_date)
    if end_date:
        conditions.append(FinancialRecord.period_start <= end_date)
    if data_source:
        try:
            source_enum = DataSource(data_source.lower())
            conditions.append(FinancialRecord.source == source_enum)
        except ValueError:
            # Invalid data source, skip filter
            pass

    if conditions:
        query = query.where(and_(*conditions))

    # Order by period_end descending (most recent first)
    query = query.order_by(FinancialRecord.period_end.desc())

    # Apply pagination
    query = query.limit(limit).offset(offset)

    # Execute query
    result = await session.execute(query)
    records = result.scalars().all()

    # Format response
    response = []
    for record in records:
        # Calculate totals from line items
        revenue_query = select(func.sum(FinancialLineItem.value).label("total")).where(
            and_(
                FinancialLineItem.financial_record_id == record.id,
                FinancialLineItem.category.in_(["Revenue", "Income"]),
            )
        )
        expense_query = select(func.sum(FinancialLineItem.value).label("total")).where(
            and_(
                FinancialLineItem.financial_record_id == record.id,
                FinancialLineItem.category == "Expenses",
            )
        )
        count_query = select(func.count(FinancialLineItem.id).label("count")).where(
            FinancialLineItem.financial_record_id == record.id
        )

        revenue_result = await session.execute(revenue_query)
        expense_result = await session.execute(expense_query)
        count_result = await session.execute(count_query)

        total_revenue = Decimal(str(revenue_result.scalar_one_or_none() or 0))
        total_expenses = Decimal(str(expense_result.scalar_one_or_none() or 0))
        line_items_count = count_result.scalar_one_or_none() or 0

        response.append(
            FinancialRecordResponse(
                id=record.id,
                company_id=record.company_id,
                company_name=record.company.name,
                period_start=record.period_start,
                period_end=record.period_end,
                data_source=str(record.source.value),
                total_revenue=total_revenue,
                total_expenses=total_expenses,
                net_income=total_revenue - total_expenses,
                line_items_count=line_items_count,
            )
        )

    return response


@router.get("/records/{record_id}/items", response_model=List[LineItemResponse])
async def get_line_items(
    record_id: int,
    category: Optional[str] = Query(None, description="Filter by category"),
    session: AsyncSession = Depends(get_db),
) -> List[LineItemResponse]:
    """
    Get line items for a specific financial record.

    Path parameters:
    - record_id: ID of the financial record

    Query parameters:
    - category: Filter by category (e.g., 'Revenue', 'Expenses')
    """
    # Verify record exists
    record = await session.get(FinancialRecord, record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Financial record not found")

    # Build query
    query = select(FinancialLineItem).where(FinancialLineItem.financial_record_id == record_id)

    if category:
        query = query.where(FinancialLineItem.category == category)

    # Order by value descending (largest items first)
    query = query.order_by(FinancialLineItem.value.desc())

    result = await session.execute(query)
    items = result.scalars().all()

    # Get total revenue for percentage calculation
    revenue_query = select(func.sum(FinancialLineItem.value).label("total")).where(
        and_(
            FinancialLineItem.financial_record_id == record_id,
            FinancialLineItem.category.in_(["Revenue", "Income"]),
        )
    )
    revenue_result = await session.execute(revenue_query)
    total_revenue = Decimal(str(revenue_result.scalar_one_or_none() or 1))  # Avoid division by zero

    # Format response
    response = []
    for item in items:
        response.append(
            LineItemResponse(
                id=item.id,
                record_id=item.financial_record_id,
                name=item.name,
                category=item.category,
                value=item.value,
                percentage_of_revenue=(
                    round((item.value / total_revenue) * 100, 2)
                    if total_revenue > 0
                    else None
                ),
            )
        )

    return response


@router.get("/summary", response_model=FinancialSummaryResponse)
async def get_financial_summary(
    company_id: Optional[int] = Query(None, description="Filter by company ID"),
    period: str = Query(
        ..., description="Period: Q1/Q2/Q3/Q4 YYYY, YYYY, or date range"
    ),
    session: AsyncSession = Depends(get_db),
) -> FinancialSummaryResponse:
    """
    Get aggregated financial summary for a period.

    Query parameters:
    - company_id: Optional company filter
    - period: Examples: "Q1 2024", "2024", "2024-01-01:2024-03-31"
    """
    # Parse period
    start_date, end_date = _parse_period(period)

    # Build base query for records
    record_query = select(FinancialRecord).where(
        and_(
            FinancialRecord.period_start <= end_date,
            FinancialRecord.period_end >= start_date,
        )
    )

    if company_id:
        record_query = record_query.where(FinancialRecord.company_id == company_id)
        # Get company info
        company = await session.get(Company, company_id)
        company_name = company.name if company else None
    else:
        company_name = "All Companies"

    # Get all matching records
    result = await session.execute(record_query)
    records = result.scalars().all()

    if not records:
        raise HTTPException(
            status_code=404, detail="No financial data found for the specified period"
        )

    record_ids = [r.id for r in records]

    # Aggregate line items
    totals_query = (
        select(
            FinancialLineItem.category,
            func.sum(FinancialLineItem.value).label("total_value"),
            func.count(FinancialLineItem.id).label("item_count"),
        )
        .where(FinancialLineItem.financial_record_id.in_(record_ids))
        .group_by(FinancialLineItem.category)
    )

    totals_result = await session.execute(totals_query)
    category_totals = totals_result.all()

    # Calculate summary
    total_revenue = Decimal("0")
    total_expenses = Decimal("0")
    categories = []

    for category, total_value, item_count in category_totals:
        value_decimal = Decimal(str(total_value))
        if category in ["Revenue", "Income"]:
            total_revenue += value_decimal
        elif category == "Expenses":
            total_expenses += value_decimal

        categories.append(
            CategorySummary(
                category=category, total_value=value_decimal, item_count=item_count
            )
        )

    # Calculate percentages
    total_all = sum(cat.total_value for cat in categories)
    for cat in categories:
        cat.percentage_of_total = (
            round((cat.total_value / total_all) * 100, 2) if total_all > 0 else None
        )

    # Sort categories by value
    categories.sort(key=lambda x: x.total_value, reverse=True)

    net_income = total_revenue - total_expenses
    profit_margin = (
        round((net_income / total_revenue) * 100, 2) if total_revenue > 0 else None
    )

    return FinancialSummaryResponse(
        company_id=company_id,
        company_name=company_name,
        period=period,
        total_revenue=total_revenue,
        total_expenses=total_expenses,
        net_income=net_income,
        profit_margin=profit_margin,
        categories=categories,
    )


@router.get("/categories")
async def get_categories(
    session: AsyncSession = Depends(get_db),
) -> Dict[str, List[str]]:
    """
    Get all unique categories and their line items.
    """
    # Get unique categories
    category_query = select(FinancialLineItem.category).distinct()
    category_result = await session.execute(category_query)
    categories = category_result.scalars().all()

    # For each category, get unique line item names
    result = {}
    for category in categories:
        items_query = (
            select(FinancialLineItem.name)
            .where(FinancialLineItem.category == category)
            .distinct()
            .order_by(FinancialLineItem.name)
        )

        items_result = await session.execute(items_query)
        items = items_result.scalars().all()

        result[category] = items

    return result


@router.get("/trends/{metric}", response_model=List[TrendDataPoint])
async def get_financial_trends(
    metric: str,
    company_id: Optional[int] = Query(None, description="Filter by company ID"),
    granularity: str = Query("month", description="Granularity: month, quarter, year"),
    periods: int = Query(12, ge=1, le=60, description="Number of periods to return"),
    session: AsyncSession = Depends(get_db),
) -> List[TrendDataPoint]:
    """
    Get trend data for a specific metric over time.

    Path parameters:
    - metric: Metric to track (revenue, expenses, profit, or specific line item)

    Query parameters:
    - company_id: Optional company filter
    - granularity: Time granularity (month, quarter, year)
    - periods: Number of periods to return (default 12)
    """
    # For now, return a simple message - this would be implemented with proper time-series queries
    # This is a placeholder implementation
    return [
        TrendDataPoint(
            period=f"2024-{i:02d}",
            value=Decimal(str(100000 + i * 5000)),
            change_from_previous=Decimal("5000") if i > 1 else None,
            percentage_change=Decimal("5.0") if i > 1 else None,
        )
        for i in range(1, min(periods + 1, 13))
    ]


@router.post("/calculate")
async def calculate_metrics(
    formula: str = Query(
        ..., description="Formula to calculate (e.g., gross_margin, current_ratio)"
    ),
    period: str = Query(..., description="Period for calculation"),
    company_id: Optional[int] = Query(None, description="Company filter"),
    session: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Calculate financial metrics using predefined formulas.

    Supported formulas:
    - gross_margin: (Revenue - COGS) / Revenue
    - operating_margin: Operating Income / Revenue
    - profit_margin: Net Income / Revenue
    - expense_ratio: Total Expenses / Revenue
    """
    # This is a simplified implementation
    # In production, this would use a proper formula engine

    # Get summary data for the period
    summary = await get_financial_summary(
        company_id=company_id, period=period, session=session
    )

    result = {"formula": formula, "period": period, "company_id": company_id}

    # Calculate based on formula
    if formula == "profit_margin":
        result["value"] = summary.profit_margin
        result["formatted"] = f"{summary.profit_margin}%"
    elif formula == "expense_ratio":
        ratio = (
            (summary.total_expenses / summary.total_revenue * 100)
            if summary.total_revenue > 0
            else 0
        )
        result["value"] = round(ratio, 2)
        result["formatted"] = f"{result['value']}%"
    else:
        result["value"] = None
        result["error"] = f"Formula '{formula}' not yet implemented"

    return result


# Helper functions
def _parse_period(period: str) -> tuple[date, date]:
    """Parse period string into start and end dates"""
    period = period.strip()

    # Date range format: "2024-01-01:2024-03-31"
    if ":" in period:
        start_str, end_str = period.split(":")
        return date.fromisoformat(start_str), date.fromisoformat(end_str)

    # Quarter format: "Q1 2024"
    if period.startswith("Q"):
        quarter = int(period[1])
        year = int(period.split()[1])

        quarter_starts = {
            1: date(year, 1, 1),
            2: date(year, 4, 1),
            3: date(year, 7, 1),
            4: date(year, 10, 1),
        }
        quarter_ends = {
            1: date(year, 3, 31),
            2: date(year, 6, 30),
            3: date(year, 9, 30),
            4: date(year, 12, 31),
        }

        return quarter_starts[quarter], quarter_ends[quarter]

    # Year format: "2024"
    if len(period) == 4 and period.isdigit():
        year = int(period)
        return date(year, 1, 1), date(year, 12, 31)

    # Month format: "2024-01"
    if len(period) == 7 and period[4] == "-":
        year, month = map(int, period.split("-"))
        start = date(year, month, 1)
        if month == 12:
            end = date(year, 12, 31)
        else:
            end = date(year, month + 1, 1) - timedelta(days=1)
        return start, end

    raise ValueError(f"Invalid period format: {period}")


@router.get("/export")
async def export_financial_data(
    format: str = Query("csv", description="Export format: csv, json"),
    company_id: Optional[int] = Query(None),
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    session: AsyncSession = Depends(get_db),
):
    """
    Export financial data in various formats.

    Query parameters:
    - format: Export format (csv or json)
    - company_id: Optional company filter
    - start_date: Start date filter
    - end_date: End date filter
    """
    # Build query
    query = select(FinancialRecord).options(selectinload(FinancialRecord.company))

    # Apply filters
    conditions = []
    if company_id:
        conditions.append(FinancialRecord.company_id == company_id)
    if start_date:
        conditions.append(FinancialRecord.period_end >= start_date)
    if end_date:
        conditions.append(FinancialRecord.period_start <= end_date)

    if conditions:
        query = query.where(and_(*conditions))

    # Order by period_end descending
    query = query.order_by(FinancialRecord.period_end.desc())
    
    # Limit for exports
    query = query.limit(10000)

    # Execute query
    result = await session.execute(query)
    records_data = result.scalars().all()
    
    # Format records (similar to get_financial_records)
    records = []
    for record in records_data:
        # Get aggregated data
        revenue_query = select(func.sum(FinancialLineItem.value).label("total")).where(
            and_(
                FinancialLineItem.financial_record_id == record.id,
                FinancialLineItem.category.in_(["Revenue", "Income"]),
            )
        )
        expense_query = select(func.sum(FinancialLineItem.value).label("total")).where(
            and_(
                FinancialLineItem.financial_record_id == record.id,
                FinancialLineItem.category == "Expenses",
            )
        )
        count_query = select(func.count(FinancialLineItem.id).label("count")).where(
            FinancialLineItem.financial_record_id == record.id
        )

        revenue_result = await session.execute(revenue_query)
        expense_result = await session.execute(expense_query)
        count_result = await session.execute(count_query)

        total_revenue = Decimal(str(revenue_result.scalar_one_or_none() or 0))
        total_expenses = Decimal(str(expense_result.scalar_one_or_none() or 0))
        line_items_count = count_result.scalar_one_or_none() or 0

        records.append(
            FinancialRecordResponse(
                id=record.id,
                company_id=record.company_id,
                company_name=record.company.name,
                period_start=record.period_start,
                period_end=record.period_end,
                data_source=str(record.source.value),
                total_revenue=total_revenue,
                total_expenses=total_expenses,
                net_income=total_revenue - total_expenses,
                line_items_count=line_items_count,
            ).model_dump()
        )

    if format == "json":
        return records
    elif format == "csv":
        # In a real implementation, this would return a proper CSV file
        # For now, return a message
        return {
            "message": "CSV export not yet implemented",
            "records_count": len(records),
        }
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")


# Add missing import
from datetime import timedelta
