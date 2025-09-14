"""Test script to view database records"""

import sqlite3
from datetime import datetime


def format_datetime(dt_str):
    """Format datetime string for display"""
    if dt_str:
        return datetime.fromisoformat(dt_str).strftime("%Y-%m-%d")
    return "N/A"


def show_table_data(cursor, table_name, limit=10):
    """Show top and bottom records from a table"""
    print(f"\n{'='*60}")
    print(f"{table_name.upper()} TABLE")
    print(f"{'='*60}")

    # Get total count
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    total = cursor.fetchone()[0]
    print(f"Total records: {total}")

    if total == 0:
        print("No records found!")
        return

    # Get column names
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [col[1] for col in cursor.fetchall()]

    # Top records
    print(f"\nTop {min(limit, total)} records:")
    print("-" * 60)
    cursor.execute(f"SELECT * FROM {table_name} LIMIT {limit}")

    for row in cursor.fetchall():
        if table_name == "companies":
            print(f"ID: {row[0]}, Name: {row[1]}, External ID: {row[2]}")
        elif table_name == "financial_records":
            print(
                f"ID: {row[0]}, Company: {row[1]}, Source: {row[2]}, "
                f"Period: {format_datetime(row[3])} to {format_datetime(row[4])}, "
                f"Type: {row[5]}, Currency: {row[6]}"
            )
        elif table_name == "financial_line_items":
            print(
                f"ID: {row[0]}, Record: {row[1]}, Category: {row[2]}, "
                f"Name: {row[3][:30]}..., Value: ${row[4]:,.2f}"
            )

    # Bottom records (if more than limit)
    if total > limit:
        print(f"\nBottom {min(limit, total)} records:")
        print("-" * 60)
        cursor.execute(f"SELECT * FROM {table_name} ORDER BY id DESC LIMIT {limit}")

        for row in cursor.fetchall():
            if table_name == "companies":
                print(f"ID: {row[0]}, Name: {row[1]}, External ID: {row[2]}")
            elif table_name == "financial_records":
                print(
                    f"ID: {row[0]}, Company: {row[1]}, Source: {row[2]}, "
                    f"Period: {format_datetime(row[3])} to {format_datetime(row[4])}, "
                    f"Type: {row[5]}, Currency: {row[6]}"
                )
            elif table_name == "financial_line_items":
                print(
                    f"ID: {row[0]}, Record: {row[1]}, Category: {row[2]}, "
                    f"Name: {row[3][:30]}..., Value: ${row[4]:,.2f}"
                )


def main():
    """Main function to display database contents"""
    print("Financial Database Contents")
    print("=" * 60)

    # Connect to database
    try:
        conn = sqlite3.connect("financial_data.db")
        cursor = conn.cursor()

        # Show data from each table
        for table in ["companies", "financial_records", "financial_line_items"]:
            show_table_data(cursor, table)

        # Show some aggregated stats
        print(f"\n{'='*60}")
        print("SUMMARY STATISTICS")
        print(f"{'='*60}")

        # Stats by company
        cursor.execute(
            """
            SELECT c.name, COUNT(DISTINCT fr.id) as record_count, 
                   COUNT(DISTINCT fli.id) as line_item_count,
                   SUM(CASE WHEN fli.category IN ('Revenue', 'Income') THEN fli.value ELSE 0 END) as total_revenue,
                   SUM(CASE WHEN fli.category IN ('Expenses', 'COGS') THEN fli.value ELSE 0 END) as total_expenses
            FROM companies c
            LEFT JOIN financial_records fr ON c.id = fr.company_id
            LEFT JOIN financial_line_items fli ON fr.id = fli.financial_record_id
            GROUP BY c.id, c.name
        """
        )

        print("\nBy Company:")
        for row in cursor.fetchall():
            print(f"- {row[0]}: {row[1]} periods, {row[2]} line items")
            print(f"  Total Revenue: ${row[3]:,.2f}")
            print(f"  Total Expenses: ${row[4]:,.2f}")

        # Date range
        cursor.execute(
            """
            SELECT MIN(period_start), MAX(period_end) 
            FROM financial_records
        """
        )
        date_range = cursor.fetchone()
        if date_range[0]:
            print(
                f"\nDate Range: {format_datetime(date_range[0])} to {format_datetime(date_range[1])}"
            )

        conn.close()

    except sqlite3.Error as e:
        print(f"Database error: {e}")
        print(
            "\nMake sure the server has been run at least once to create the database."
        )


if __name__ == "__main__":
    main()
