"""
Test analytics features (forecasting and anomaly detection).
Can be used both locally and on Replit deployment.
"""

import sys

sys.path.append(".")

import requests
import json
from src.config.settings import settings

# Use the configured API URL (local or Replit)
API_URL = settings.API_URL


def test_analytics():
    """Test forecasting and anomaly detection features"""
    print(f"\n{'='*60}")
    print("ANALYTICS FEATURES TEST")
    print(f"API URL: {API_URL}")
    print(f"{'='*60}\n")

    # Test forecasting queries
    print("\n--- FORECASTING TESTS ---")
    forecast_queries = [
        "Forecast revenue for the next 3 months",
        "Predict expenses for Q2 2025",
        "What will our cash flow be next quarter?",
        "Project profit margins for the next 6 months",
    ]

    session_id = "test-analytics-session"

    for query in forecast_queries:
        print(f"\n🔮 {query}")

        try:
            response = requests.post(
                f"{API_URL}/api/queries/natural",
                json={"query": query, "session_id": session_id},
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("intent") == "forecast":
                    print("✓ Forecast intent recognized")
                    print(f"Response: {data.get('narrative', '')[:200]}...")
                else:
                    print(f"Intent: {data.get('intent')} (expected 'forecast')")
            else:
                print(f"✗ Error {response.status_code}")

        except Exception as e:
            print(f"✗ Request failed: {e}")

    # Test anomaly detection queries
    print("\n\n--- ANOMALY DETECTION TESTS ---")
    anomaly_queries = [
        "Are there any anomalies in our expenses?",
        "Show me unusual revenue patterns",
        "Detect outliers in financial data for 2024",
        "Find any strange expense patterns this quarter",
    ]

    for query in anomaly_queries:
        print(f"\n🔍 {query}")

        try:
            response = requests.post(
                f"{API_URL}/api/queries/natural",
                json={"query": query, "session_id": session_id},
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("intent") == "anomaly":
                    print("✓ Anomaly intent recognized")
                    print(f"Response: {data.get('narrative', '')[:200]}...")
                else:
                    print(f"Intent: {data.get('intent')} (expected 'anomaly')")
            else:
                print(f"✗ Error {response.status_code}")

        except Exception as e:
            print(f"✗ Request failed: {e}")

    print(f"\n{'='*60}")
    print("ANALYTICS TEST COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    test_analytics()
