"""
Data import service for processing and storing financial data
"""

import json
from typing import Dict, Any, List
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from src.models.financial import Company, FinancialRecord, FinancialLineItem
from src.models.base import DataSource
from src.parsers.quickbooks import QuickBooksParser
from src.parsers.rootfi import RootfiParser


class DataImportService:
    """Service for importing financial data from various sources"""

    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.parsers = {
            DataSource.QUICKBOOKS: QuickBooksParser(),
            DataSource.ROOTFI: RootfiParser(),
        }

    async def import_file(self, file_path: str, source: DataSource) -> Dict[str, Any]:
        """
        Import financial data from a file

        Args:
            file_path: Path to the data file
            source: Data source type

        Returns:
            Import summary
        """
        # Get appropriate parser
        parser = self.parsers.get(source)
        if not parser:
            raise ValueError(f"No parser available for source: {source}")

        # Parse the file
        parsed_data = parser.parse_file(file_path)

        # Import to database
        result = await self._import_parsed_data(parsed_data, source)

        return result

    async def _import_parsed_data(
        self, data: Dict[str, Any], source: DataSource
    ) -> Dict[str, Any]:
        """Import parsed data into the database"""
        company_name = data.get("company_name", "Unknown Company")
        currency = data.get("currency", "USD")
        records_data = data.get("records", {})

        # Get or create company
        company = await self._get_or_create_company(company_name)

        # Import financial records
        imported_count = 0
        errors = []

        for period_key, record_data in records_data.items():
            try:
                await self._import_record(company, record_data, source, currency)
                imported_count += 1
            except Exception as e:
                errors.append(f"Error importing {period_key}: {str(e)}")
                print(f"Import error for {period_key}: {e}")  # Debug logging

        # Commit all changes
        await self.db.commit()

        return {
            "success": True,
            "company_id": company.id,
            "imported_records": imported_count,
            "errors": errors,
        }

    async def _get_or_create_company(self, name: str) -> Company:
        """Get existing company or create new one"""
        result = await self.db.execute(select(Company).where(Company.name == name))
        company = result.scalar_one_or_none()

        if not company:
            company = Company(name=name)
            self.db.add(company)
            await self.db.flush()

        return company

    async def _import_record(
        self,
        company: Company,
        record_data: Dict[str, Any],
        source: DataSource,
        currency: str,
    ):
        """Import a single financial record"""
        period = record_data["period"]
        line_items_data = record_data.get("line_items", [])

        # Create financial record
        record = FinancialRecord(
            company_id=company.id,
            source=source,
            period_start=period["start_date"],
            period_end=period["end_date"],
            period_type="monthly",  # Detect from date range
            currency=currency,
            raw_data=json.dumps(
                record_data, default=str
            ),  # Store original data with datetime serialization
        )

        self.db.add(record)
        await self.db.flush()

        # Import line items
        for item_data in line_items_data:
            line_item = FinancialLineItem(
                financial_record_id=record.id,
                category=item_data["category"],
                name=item_data["name"],
                value=item_data["value"],
                account_id=item_data.get("account_id", ""),
            )
            self.db.add(line_item)

        # Flush to ensure all line items are saved
        await self.db.flush()


class DataQueryService:
    """Service for querying financial data"""

    def __init__(self, db_session: AsyncSession):
        self.db = db_session

    async def get_companies(self) -> List[Company]:
        """Get all companies"""
        result = await self.db.execute(select(Company))
        return result.scalars().all()

    async def get_financial_records(
        self,
        company_id: int = None,
        start_date: datetime = None,
        end_date: datetime = None,
    ) -> List[FinancialRecord]:
        """Get financial records with optional filters"""
        query = select(FinancialRecord).options(
            # Eager load relationships
            selectinload(FinancialRecord.company),
            selectinload(FinancialRecord.line_items),
        )

        if company_id:
            query = query.where(FinancialRecord.company_id == company_id)

        if start_date:
            query = query.where(FinancialRecord.period_start >= start_date)

        if end_date:
            query = query.where(FinancialRecord.period_end <= end_date)

        result = await self.db.execute(query)
        return result.scalars().all()

    async def get_summary_stats(self, company_id: int = None) -> Dict[str, Any]:
        """Get summary statistics for financial data"""
        # Get all records
        records = await self.get_financial_records(company_id=company_id)

        if not records:
            return {"total_records": 0, "date_range": None, "sources": []}

        # Calculate stats
        start_dates = [r.period_start for r in records]
        end_dates = [r.period_end for r in records]
        sources = list(set(r.source.value for r in records))

        return {
            "total_records": len(records),
            "date_range": {"start": min(start_dates), "end": max(end_dates)},
            "sources": sources,
            "companies": list(set(r.company.name for r in records)),
        }
