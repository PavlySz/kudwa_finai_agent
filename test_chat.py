"""
Simple chat test for the AI Financial Agent.
Can be used both locally and on Replit deployment.
"""

import sys

sys.path.append(".")

import requests
import json
from src.config.settings import settings

# Use the configured API URL (local or Replit)
API_URL = settings.API_URL


def test_chat():
    """Test basic chat functionality"""
    print(f"\n{'='*60}")
    print("AI FINANCIAL AGENT CHAT TEST")
    print(f"API URL: {API_URL}")
    print(f"{'='*60}\n")

    # Test health check first
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            print("✓ API is healthy")
        else:
            print(f"✗ API health check failed: {response.status_code}")
            return
    except Exception as e:
        print(f"✗ Cannot connect to API: {e}")
        print(f"Make sure the server is running at {API_URL}")
        return

    # Test queries
    test_queries = [
        "What was the total revenue in Q1 2024?",
        "Show me expenses for March 2024",
        "Compare revenue between Q1 and Q2 2024",
        "Forecast revenue for the next 3 months",
        "Are there any anomalies in our expenses?",
    ]

    session_id = "test-chat-session"

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'─'*60}")
        print(f"Query {i}: {query}")
        print(f"{'─'*60}")

        try:
            response = requests.post(
                f"{API_URL}/api/queries/natural",
                json={"query": query, "session_id": session_id},
            )

            if response.status_code == 200:
                data = response.json()
                print(f"✓ Success")
                print(f"Intent: {data.get('intent', 'Unknown')}")
                print(f"Model used: {data.get('model_used', 'Unknown')}")
                print(f"\nResponse:\n{data.get('narrative', 'No response')}")

                if data.get("data"):
                    print(f"\nData points: {len(data['data'])}")
            else:
                print(f"✗ Error {response.status_code}: {response.text}")

        except Exception as e:
            print(f"✗ Request failed: {e}")

    print(f"\n{'='*60}")
    print("CHAT TEST COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    test_chat()
