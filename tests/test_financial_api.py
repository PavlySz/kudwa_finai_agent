"""
Tests for RESTful financial data APIs.
"""

import sys
import os
import asyncio
from datetime import date

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from main import app


# Create test client
client = TestClient(app)


def test_get_financial_records():
    """Test getting financial records endpoint"""
    print("\n=== Testing GET /api/financial/records ===")

    # Test without filters
    response = client.get("/api/financial/records")
    assert response.status_code == 200
    data = response.json()
    print(f"✓ Retrieved {len(data)} financial records")

    # Test with company filter
    response = client.get("/api/financial/records?company_id=1")
    assert response.status_code == 200
    data = response.json()
    print(f"✓ Retrieved {len(data)} records for company 1")

    # Test with date filter
    response = client.get(
        "/api/financial/records?start_date=2024-01-01&end_date=2024-03-31"
    )
    assert response.status_code == 200
    data = response.json()
    print(f"✓ Retrieved {len(data)} records for Q1 2024")

    # Test pagination
    response = client.get("/api/financial/records?limit=5&offset=0")
    assert response.status_code == 200
    data = response.json()
    assert len(data) <= 5
    print(f"✓ Pagination works (got {len(data)} records with limit=5)")


def test_get_line_items():
    """Test getting line items for a record"""
    print("\n=== Testing GET /api/financial/records/{record_id}/items ===")

    # First get a record
    response = client.get("/api/financial/records?limit=1")
    if response.status_code == 200 and response.json():
        record_id = response.json()[0]["id"]

        # Get line items
        response = client.get(f"/api/financial/records/{record_id}/items")
        assert response.status_code == 200
        items = response.json()
        print(f"✓ Retrieved {len(items)} line items for record {record_id}")

        # Test with category filter
        response = client.get(
            f"/api/financial/records/{record_id}/items?category=Revenue"
        )
        assert response.status_code == 200
        revenue_items = response.json()
        print(f"✓ Retrieved {len(revenue_items)} revenue items")
    else:
        print("⚠ No records found to test line items")


def test_get_financial_summary():
    """Test financial summary endpoint"""
    print("\n=== Testing GET /api/financial/summary ===")

    # Test quarter summary
    response = client.get("/api/financial/summary?period=Q1 2024")
    assert response.status_code in [200, 404]  # 404 if no data for period
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Q1 2024 Summary:")
        print(f"  - Revenue: ${float(data['total_revenue']):,.2f}")
        print(f"  - Expenses: ${float(data['total_expenses']):,.2f}")
        print(f"  - Net Income: ${float(data['net_income']):,.2f}")
        if data.get("profit_margin"):
            print(f"  - Profit Margin: {data['profit_margin']}%")

    # Test year summary
    response = client.get("/api/financial/summary?period=2024")
    assert response.status_code in [200, 404]
    if response.status_code == 200:
        print("✓ Retrieved 2024 annual summary")

    # Test date range
    response = client.get("/api/financial/summary?period=2024-01-01:2024-03-31")
    assert response.status_code in [200, 404]
    if response.status_code == 200:
        print("✓ Retrieved custom date range summary")


def test_get_categories():
    """Test categories endpoint"""
    print("\n=== Testing GET /api/financial/categories ===")

    response = client.get("/api/financial/categories")
    assert response.status_code == 200
    categories = response.json()

    print(f"✓ Found {len(categories)} categories:")
    for category, items in categories.items():
        print(f"  - {category}: {len(items)} unique items")


def test_get_trends():
    """Test trends endpoint"""
    print("\n=== Testing GET /api/financial/trends/{metric} ===")

    # Test revenue trend
    response = client.get("/api/financial/trends/revenue?periods=6")
    assert response.status_code == 200
    trend_data = response.json()
    print(f"✓ Retrieved {len(trend_data)} data points for revenue trend")

    # Test with company filter
    response = client.get(
        "/api/financial/trends/expenses?company_id=1&granularity=quarter"
    )
    assert response.status_code == 200
    print("✓ Retrieved quarterly expense trend for company 1")


def test_calculate_metrics():
    """Test metric calculation endpoint"""
    print("\n=== Testing POST /api/financial/calculate ===")

    # Test profit margin calculation
    response = client.post(
        "/api/financial/calculate?formula=profit_margin&period=Q1 2024"
    )
    assert response.status_code in [200, 404]
    if response.status_code == 200:
        data = response.json()
        if data.get("value"):
            print(f"✓ Profit margin for Q1 2024: {data['formatted']}")
        else:
            print(f"✓ {data.get('error', 'Calculation completed')}")

    # Test expense ratio
    response = client.post("/api/financial/calculate?formula=expense_ratio&period=2024")
    assert response.status_code in [200, 404]
    if response.status_code == 200:
        data = response.json()
        if data.get("value"):
            print(f"✓ Expense ratio for 2024: {data['formatted']}")


def test_export_data():
    """Test data export endpoint"""
    print("\n=== Testing GET /api/financial/export ===")

    # Test JSON export
    response = client.get("/api/financial/export?format=json&limit=10")
    assert response.status_code == 200
    print("✓ JSON export works")

    # Test CSV export (not implemented yet)
    response = client.get("/api/financial/export?format=csv")
    assert response.status_code == 200
    data = response.json()
    print(f"✓ CSV export placeholder: {data.get('message', 'Success')}")


def run_all_tests():
    """Run all financial API tests"""
    print("\n" + "=" * 60)
    print("FINANCIAL API TESTS")
    print("=" * 60)

    test_get_financial_records()
    test_get_line_items()
    test_get_financial_summary()
    test_get_categories()
    test_get_trends()
    test_calculate_metrics()
    test_export_data()

    print("\n" + "=" * 60)
    print("✅ ALL FINANCIAL API TESTS COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
