"""
Parser for Rootfi financial data format
"""

import json
from datetime import datetime
from typing import List, Dict, Any
from src.models.base import DataSource


class RootfiParser:
    """Parser for Rootfi financial data format"""

    def __init__(self):
        self.source = DataSource.ROOTFI

    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """
        Parse Rootfi JSON file and return structured data

        Args:
            file_path: Path to the JSON file

        Returns:
            Dictionary with parsed financial data
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return self.parse_data(data)

    def parse_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse Rootfi JSON data structure

        Args:
            data: Raw JSON data

        Returns:
            Structured financial data
        """
        records = data.get("data", [])

        # Group records by company
        companies_data = {}

        for record in records:
            company_id = record.get("rootfi_company_id", "unknown")

            if company_id not in companies_data:
                companies_data[company_id] = {
                    "company_id": str(company_id),
                    "records": {},
                }

            # Create unique period key
            period_key = f"{record.get('period_start')}_{record.get('period_end')}"

            # Parse the record
            parsed_record = self._parse_record(record)
            companies_data[company_id]["records"][period_key] = parsed_record

        # For now, return data for the first company (or handle multiple companies)
        if companies_data:
            first_company = list(companies_data.values())[0]
            return {
                "company_name": f"Company_{first_company['company_id']}",
                "currency": "USD",  # Default, could be extracted from records
                "report_basis": "Accrual",  # Default
                "records": first_company["records"],
            }

        return {
            "company_name": "Unknown Company",
            "currency": "USD",
            "report_basis": "Accrual",
            "records": {},
        }

    def _parse_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Parse individual Rootfi record"""
        # Parse dates
        period_start = datetime.fromisoformat(
            record["period_start"].replace("Z", "+00:00")
        )
        period_end = datetime.fromisoformat(record["period_end"].replace("Z", "+00:00"))

        # Collect all line items
        line_items = []

        # Process revenue items
        for revenue_section in record.get("revenue", []):
            self._extract_line_items(revenue_section, "Revenue", line_items)

        # Process cost of goods sold
        for cogs_section in record.get("cost_of_goods_sold", []):
            self._extract_line_items(cogs_section, "Cost of Goods Sold", line_items)

        # Process expenses
        for expense_section in record.get("expenses", []):
            self._extract_line_items(expense_section, "Expenses", line_items)

        # Process other income/expenses
        for other_section in record.get("other_income_and_expenses", []):
            self._extract_line_items(other_section, "Other Income/Expenses", line_items)

        return {
            "period": {
                "start_date": period_start,
                "end_date": period_end,
                "col_key": f"{period_start.strftime('%b %Y')}",
            },
            "line_items": line_items,
            "metadata": {
                "rootfi_id": record.get("rootfi_id"),
                "platform_id": record.get("platform_id"),
                "net_income": record.get("net_income", 0),
                "gross_profit": record.get("gross_profit", 0),
                "total_revenue": record.get("total_revenue", 0),
                "total_expenses": record.get("total_expenses", 0),
            },
        }

    def _extract_line_items(
        self,
        section: Dict[str, Any],
        category: str,
        line_items: List[Dict],
        parent_name: str = None,
    ):
        """Recursively extract line items from a section"""
        name = section.get("name", "")
        value = section.get("value", 0)
        account_id = section.get("account_id", "")

        # Add the current item
        if name and value is not None:
            full_name = f"{parent_name} > {name}" if parent_name else name
            line_items.append(
                {
                    "category": category,
                    "name": full_name,
                    "value": float(value),
                    "account_id": account_id,
                }
            )

        # Process sub-items
        for sub_item in section.get("line_items", []):
            self._extract_line_items(sub_item, category, line_items, name)
