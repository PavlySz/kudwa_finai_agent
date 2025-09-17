"""
Tests for AI Evaluation Framework APIs.
"""

import sys
import os
import asyncio
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from main import app


# Create test client
client = TestClient(app)


def test_run_test_suite():
    """Test running the automated test suite"""
    print("\n=== Testing POST /api/eval/test-suite/run ===")

    # Test running all tests
    response = client.post("/api/eval/test-suite/run", json={})
    if response.status_code != 200:
        print(f"Error: Status {response.status_code}")
        print(f"Detail: {response.json().get('detail', 'Unknown error')}")
    assert response.status_code == 200
    data = response.json()
    print(f"✓ Ran {data['summary']['total_tests']} tests")
    print(f"  - Passed: {data['summary']['passed']}")
    print(f"  - Failed: {data['summary']['failed']}")
    print(f"  - Pass rate: {data['summary']['pass_rate']:.1f}%")

    # Test running specific category
    print("\n" + "-"*60)
    print("Testing category-specific run (basic category only)...")
    print("-"*60)
    response = client.post("/api/eval/test-suite/run", json={"category": "basic"})
    assert response.status_code == 200
    data = response.json()
    print(f"✓ Ran basic category tests: {data['summary']['total_tests']} tests")
    print(f"  - Passed: {data['summary']['passed']}")
    print(f"  - Failed: {data['summary']['failed']}")

    # Test running specific test IDs
    print("\n" + "-"*60)
    print("Testing specific test IDs run...")
    print("-"*60)
    response = client.post(
        "/api/eval/test-suite/run", json={"test_ids": ["basic_001", "basic_002"]}
    )
    assert response.status_code == 200
    data = response.json()
    print(f"✓ Ran specific tests: {data['summary']['total_tests']} tests")
    print(f"  - Passed: {data['summary']['passed']}")
    print(f"  - Failed: {data['summary']['failed']}")


def test_llm_judge():
    """Test LLM-as-Judge evaluation"""
    print("\n=== Testing POST /api/eval/judge ===")

    # Test single evaluation
    response = client.post(
        "/api/eval/judge",
        json={
            "query": "What was the total revenue in Q1 2024?",
            "response": "Based on the financial data, the total revenue for Q1 2024 was $5,234,567.89, representing a 15% increase from Q1 2023.",
            "ground_truth": None,
            "context": {"period": "Q1 2024"},
        },
    )

    # Note: This might fail if API keys aren't configured
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Evaluation completed:")
        print(f"  - Overall score: {data['overall_score']}/10")
        print(f"  - Evaluator model: {data['evaluator_model']}")

        # Show individual criterion scores
        for score in data["scores"]:
            print(f"  - {score['criterion']}: {score['score']}/10")
    else:
        print(f"⚠ Judge evaluation skipped (status: {response.status_code})")
        print(f"  Reason: {response.json().get('detail', 'Unknown error')}")


def test_batch_judge():
    """Test batch LLM-as-Judge evaluation"""
    print("\n=== Testing POST /api/eval/judge/batch ===")

    # Test batch evaluation
    response = client.post(
        "/api/eval/judge/batch",
        json={
            "evaluations": [
                {
                    "query": "What was the revenue?",
                    "response": "The revenue was $5M",
                    "ground_truth": "$5,234,567",
                    "context": None,
                },
                {
                    "query": "Show me expenses",
                    "response": "Total expenses: $2.1M",
                    "ground_truth": None,
                    "context": {"period": "2024"},
                },
            ]
        },
    )

    if response.status_code == 200:
        data = response.json()
        print(f"✓ Batch evaluation completed:")
        print(f"  - Total evaluations: {len(data['individual_results'])}")
        print(f"  - Overall average: {data['aggregate_scores']['overall_average']}/10")
    else:
        print(f"⚠ Batch evaluation skipped (status: {response.status_code})")


def test_metrics_dashboard():
    """Test metrics dashboard endpoint"""
    print("\n=== Testing GET /api/eval/metrics/dashboard ===")

    response = client.get("/api/eval/metrics/dashboard")
    assert response.status_code == 200
    data = response.json()

    print("✓ Metrics dashboard retrieved:")
    print(f"  - Queries per minute: {data['real_time']['queries_per_minute']:.2f}")
    print(f"  - Success rate: {data['real_time']['success_rate']:.1f}%")
    print(f"  - Active sessions: {data['real_time']['active_sessions']}")
    print(f"  - Errors last 5min: {data['real_time']['errors_last_5min']}")

    # Show model performance if available
    if data["by_model"]:
        print("\n  Model performance:")
        for model, metrics in data["by_model"].items():
            print(f"  - {model}: {metrics.get('total_queries', 0)} queries")


def test_metrics_history():
    """Test metrics history endpoint"""
    print("\n=== Testing GET /api/eval/metrics/history ===")

    # Test different time periods
    for period in ["last_hour", "last_24h", "last_7d"]:
        response = client.get(f"/api/eval/metrics/history?time_period={period}")
        assert response.status_code == 200
        data = response.json()
        print(
            f"✓ {period} metrics: {data['total_queries']} queries, {data['success_rate']:.1f}% success"
        )


def test_ab_test_management():
    """Test A/B test creation and management"""
    print("\n=== Testing A/B Test Management ===")

    # Create A/B test
    test_config = {
        "name": "GPT-5 vs Claude Sonnet",
        "description": "Compare response quality between models",
        "variant_a": {"model": "gpt-5"},
        "variant_b": {"model": "claude-sonnet-4-20250514"},
        "sample_size": 50,
    }

    response = client.post("/api/eval/ab-test/create", json=test_config)
    assert response.status_code == 200
    data = response.json()
    test_id = data["test_id"]
    print(f"✓ Created A/B test: {test_id}")

    # Run A/B test
    response = client.post(f"/api/eval/ab-test/{test_id}/run")
    assert response.status_code == 200
    data = response.json()
    print(f"✓ A/B test completed:")
    print(
        f"  - Variant A success rate: {data['results']['variant_a']['success_rate']}%"
    )
    print(
        f"  - Variant B success rate: {data['results']['variant_b']['success_rate']}%"
    )
    print(f"  - Recommendation: {data['recommendation']}")


def test_quality_report():
    """Test quality report generation"""
    print("\n=== Testing GET /api/eval/reports/quality ===")

    response = client.get("/api/eval/reports/quality?days=7")
    assert response.status_code == 200
    data = response.json()

    print("✓ Quality report generated:")
    print(
        f"  - Overall quality score: {data['executive_summary']['overall_quality_score']}/10"
    )
    print(f"  - Test pass rate: {data['executive_summary']['test_pass_rate']}%")
    print(f"  - User satisfaction: {data['executive_summary']['user_satisfaction']}/5")

    if data["executive_summary"]["key_issues"]:
        print("\n  Key issues identified:")
        for issue in data["executive_summary"]["key_issues"]:
            print(f"  - {issue}")

    if data["recommendations"]:
        print("\n  Recommendations:")
        for i, rec in enumerate(data["recommendations"][:3], 1):
            print(f"  {i}. {rec}")


def test_feedback_submission():
    """Test user feedback submission"""
    print("\n=== Testing POST /api/eval/feedback ===")

    response = client.post(
        "/api/eval/feedback",
        json={
            "query_id": "test-query-123",
            "satisfaction": 4,
            "feedback": "Response was accurate but could be more detailed",
        },
    )
    assert response.status_code == 200
    data = response.json()
    print(f"✓ Feedback submitted: {data['message']}")


def run_all_tests():
    """Run all evaluation API tests"""
    print("\n" + "=" * 60)
    print("AI EVALUATION FRAMEWORK API TESTS")
    print("=" * 60)

    test_run_test_suite()
    test_llm_judge()
    test_batch_judge()
    test_metrics_dashboard()
    test_metrics_history()
    test_ab_test_management()
    test_quality_report()
    test_feedback_submission()

    print("\n" + "=" * 60)
    print("✅ ALL EVALUATION API TESTS COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
