"""
Test script for forecasting and anomaly detection features
"""

import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:8000"


def test_analytics_query(query: str, feature_type: str):
    """Test an analytics query"""
    print(f"\n{'='*60}")
    print(f"{feature_type.upper()} TEST")
    print(f"Query: {query}")
    print(f"{'='*60}")

    response = requests.post(
        f"{BASE_URL}/api/queries/natural",
        json={
            "query": query,
            "session_id": f"test-analytics-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        },
    )

    if response.status_code == 200:
        data = response.json()
        print(f"✓ Success!")
        print(f"\nAnswer: {data.get('answer', 'No answer')}")

        # Check if forecast/anomaly data was processed
        metadata = data.get("metadata", {})
        print(f"\nDetected Intent: {metadata.get('intent')}")

        if data.get("key_insights"):
            print("\nKey Insights:")
            for insight in data["key_insights"]:
                print(f"  - {insight}")

        if data.get("supporting_data"):
            print("\nSupporting Data Available:")
            for key in data["supporting_data"]:
                print(f"  - {key}")
    else:
        print(f"✗ Error {response.status_code}: {response.text}")


def main():
    print("Testing AI Analytics Features")
    print("=" * 60)

    # Test Forecasting
    forecast_queries = [
        "Forecast revenue for the next 3 months",
        "Predict expenses for next quarter",
        "What will our profit be next month?",
        "Project cash flow for Q2 2025",
    ]

    print("\n🔮 TESTING FORECASTING")
    for query in forecast_queries:
        try:
            test_analytics_query(query, "Forecast")
        except Exception as e:
            print(f"Error: {e}")

    # Test Anomaly Detection
    anomaly_queries = [
        "Are there any anomalies in our financial data?",
        "Show me unusual expense patterns",
        "Detect outliers in revenue this year",
        "Find any strange patterns in our finances",
    ]

    print("\n\n🚨 TESTING ANOMALY DETECTION")
    for query in anomaly_queries:
        try:
            test_analytics_query(query, "Anomaly")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
