"""
Parser for QuickBooks Profit & Loss JSON format
"""

import json
from datetime import datetime
from typing import List, Dict, Any

from src.models.base import DataSource


class QuickBooksParser:
    """Parser for QuickBooks P&L JSON format"""

    def __init__(self):
        self.source = DataSource.QUICKBOOKS

    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """Parse QuickBooks JSON file and return structured data"""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return self.parse_data(data)

    def parse_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse QuickBooks JSON data structure"""
        report_data = data.get("data", {})
        header = report_data.get("Header", {})
        columns = report_data.get("Columns", {}).get("Column", [])
        rows = report_data.get("Rows", {}).get("Row", [])

        # Extract periods from columns
        periods = self._extract_periods(columns)

        # Parse financial data by period
        records_by_period = {}
        for period in periods:
            if period["col_title"] != "":  # Skip the account column
                records_by_period[period["col_key"]] = {
                    "period": period,
                    "line_items": [],
                }

        # Parse rows to extract line items
        self._parse_main_rows(rows, columns, records_by_period)

        return {
            "company_name": header.get("CompanyName", "Unknown Company"),
            "currency": header.get("Currency", "USD"),
            "report_basis": header.get("ReportBasis", "Accrual"),
            "records": records_by_period,
        }

    def _extract_periods(self, columns: List[Dict]) -> List[Dict]:
        """Extract period information from columns"""
        periods = []
        for col in columns:
            if col.get("ColType") == "Money":
                metadata = {
                    item["Name"]: item["Value"] for item in col.get("MetaData", [])
                }
                start_date_str = metadata.get("StartDate", "")
                end_date_str = metadata.get("EndDate", "")
                if start_date_str and end_date_str:
                    periods.append(
                        {
                            "col_title": col.get("ColTitle", ""),
                            "col_key": metadata.get("ColKey", ""),
                            "start_date": datetime.strptime(start_date_str, "%Y-%m-%d"),
                            "end_date": datetime.strptime(end_date_str, "%Y-%m-%d"),
                        }
                    )
        return periods

    def _parse_main_rows(
        self, rows: List[Dict], columns: List[Dict], records_by_period: Dict
    ):
        """Parse main row structure"""
        for row in rows:
            row_type = row.get("type", "")

            if row_type == "Section":
                # Section with group (e.g., Income, COGS, Expenses)
                section_name = row.get("group", "")
                section_rows = row.get("Rows", {}).get("Row", [])

                # Process all rows in this section
                for section_row in section_rows:
                    self._process_section_row(
                        section_row, columns, records_by_period, section_name
                    )

            elif "Header" in row:
                # Row with header (main category like Income, Expenses)
                header_text = row["Header"]["ColData"][0]["value"]
                sub_rows = row.get("Rows", {}).get("Row", [])

                for sub_row in sub_rows:
                    self._process_section_row(
                        sub_row, columns, records_by_period, header_text
                    )

    def _process_section_row(
        self, row: Dict, columns: List[Dict], records_by_period: Dict, category: str
    ):
        """Process a row within a section"""
        if "Header" in row:
            # This row has a header with account name
            account_name = row["Header"]["ColData"][0].get("value", "")
            account_id = row["Header"]["ColData"][0].get("id", "")

            # Check if this row has sub-rows (nested accounts)
            if "Rows" in row and row["Rows"].get("Row"):
                # Process nested rows
                for sub_row in row["Rows"]["Row"]:
                    self._process_section_row(
                        sub_row, columns, records_by_period, category
                    )
            else:
                # No sub-rows, this header row contains the data
                # In QuickBooks format, the data is in the header row itself
                col_data = row["Header"]["ColData"]
                self._extract_line_items(
                    col_data,
                    columns,
                    records_by_period,
                    category,
                    account_name,
                    account_id,
                )

        elif "ColData" in row:
            # Direct data row
            col_data = row["ColData"]
            account_name = col_data[0].get("value", "") if col_data else ""
            account_id = col_data[0].get("id", "") if col_data else ""

            if account_name:
                self._extract_line_items(
                    col_data,
                    columns,
                    records_by_period,
                    category,
                    account_name,
                    account_id,
                )

    def _extract_line_items(
        self,
        col_data: List[Dict],
        columns: List[Dict],
        records_by_period: Dict,
        category: str,
        account_name: str,
        account_id: str,
    ):
        """Extract line items from column data"""
        # Skip the first column (account name) and process money columns
        for i, value_data in enumerate(col_data[1:], 1):
            if i < len(columns):
                col_info = columns[i]
                if col_info.get("ColType") == "Money":
                    value_str = value_data.get("value", "")
                    if value_str and value_str != "":
                        try:
                            value = float(value_str)
                            # Get the period key
                            metadata = {
                                item["Name"]: item["Value"]
                                for item in col_info.get("MetaData", [])
                            }
                            col_key = metadata.get("ColKey", "")

                            if col_key in records_by_period and value != 0:
                                records_by_period[col_key]["line_items"].append(
                                    {
                                        "category": category,
                                        "name": account_name,
                                        "value": value,
                                        "account_id": account_id,
                                    }
                                )
                        except (ValueError, KeyError):
                            pass
