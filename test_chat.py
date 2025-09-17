"""
Simple test script to verify the natural language query API is working
"""

import requests
import json
from datetime import datetime

# API base URL
BASE_URL = "http://localhost:8000"


def test_natural_query(query: str, session_id: str = "test-user"):
    """Test a natural language query"""
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")

    response = requests.post(
        f"{BASE_URL}/api/queries/natural",
        json={"query": query, "session_id": session_id},
    )

    if response.status_code == 200:
        data = response.json()
        print(f"✓ Success!")
        print(f"\nAnswer: {data.get('answer', 'No answer')}")

        if data.get("key_insights"):
            print("\nKey Insights:")
            for insight in data["key_insights"]:
                print(f"  - {insight}")

        if data.get("metadata"):
            print(f"\nIntent: {data['metadata'].get('intent')}")
            print(f"Metrics: {data['metadata'].get('metrics')}")
    else:
        print(f"✗ Error {response.status_code}: {response.text}")


def main():
    print("AI-Powered Financial Intelligence System - Test Chat")
    print("=" * 60)

    # Test queries
    queries = [
        "What was the total revenue?",
        "Show me expenses for Q1 2024",
        "Compare revenue between Q1 and Q2 2024",
        "Forecast revenue for the next 3 months",
        "Are there any unusual patterns in expenses?",
        "What about profit margins?",  # Context-aware follow-up
    ]

    session_id = f"test-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    for query in queries:
        try:
            test_natural_query(query, session_id)
        except Exception as e:
            print(f"Error testing query: {e}")


if __name__ == "__main__":
    main()
