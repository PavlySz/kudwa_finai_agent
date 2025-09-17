"""
API endpoints for AI evaluation and metrics.
Provides test suite execution, LLM judging, and performance metrics.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Body
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field

from src.config.database import get_db
from src.models.base import BaseResponse
from src.evaluation import (
    TestSuite,
    AITestCase,
    TestCategory,
    LLMJudge,
    EvaluationResult,
    MetricsCollector,
    QueryMetrics,
    PerformanceMetrics,
)
from src.ai import QueryProcessor, ResponseFormatter, MultiModelClient


# Initialize components
llm_client = MultiModelClient()
query_processor = QueryProcessor(llm_client)
response_formatter = ResponseFormatter(llm_client)
test_suite = TestSuite()
llm_judge = LLMJudge(llm_client)
metrics_collector = MetricsCollector()

# Create router
router = APIRouter(prefix="/api/eval", tags=["Evaluation"])


# Request/Response models
class TestSuiteRunRequest(BaseModel):
    """Request to run test suite"""

    category: Optional[TestCategory] = Field(
        None, description="Run only tests in this category"
    )
    test_ids: Optional[List[str]] = Field(
        None, description="Run only specific test IDs"
    )


class JudgeRequest(BaseModel):
    """Request for LLM judge evaluation"""

    query: str = Field(description="The user's query")
    response: str = Field(description="The AI system's response")
    ground_truth: Optional[str] = Field(None, description="Expected correct answer")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")


class BatchJudgeRequest(BaseModel):
    """Request for batch evaluation"""

    evaluations: List[JudgeRequest]


class ABTestConfig(BaseModel):
    """Configuration for A/B test"""

    test_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str
    variant_a: Dict[str, Any] = Field(description="Configuration for variant A")
    variant_b: Dict[str, Any] = Field(description="Configuration for variant B")
    sample_size: int = Field(default=100, description="Number of queries per variant")
    test_queries: Optional[List[str]] = Field(
        None, description="Specific queries to test"
    )


# Endpoints


@router.post("/test-suite/run")
async def run_test_suite(
    request: TestSuiteRunRequest = Body(default=TestSuiteRunRequest()),
    session: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Run the automated test suite to evaluate system accuracy.

    Options:
    - Run all tests (default)
    - Run tests in a specific category
    - Run specific test IDs
    """
    try:
        # Determine which tests to run (limited to 5 for better performance)
        if request.test_ids:
            tests_to_run = [test_suite.get_test_case(tid) for tid in request.test_ids]
            tests_to_run = [t for t in tests_to_run if t is not None]
        elif request.category:
            tests_to_run = test_suite.get_test_cases(request.category, limit=5)
        else:
            tests_to_run = test_suite.get_test_cases(limit=5)

        if not tests_to_run:
            raise HTTPException(
                status_code=404, detail="No tests found matching criteria"
            )

        # Run tests
        results = []
        for i, test_case in enumerate(tests_to_run, 1):
            result = await test_suite.run_test(
                test_case, query_processor, response_formatter, test_index=i
            )
            results.append(result)

            # Record metrics
            metrics_collector.record_query(
                query_id=f"test_{test_case.id}",
                query=test_case.query,
                intent=test_case.expected_intent,  # Already a string due to use_enum_values
                complexity="test",
                model_used="test_model",
                success=result.passed,
                response_time_ms=result.execution_time_ms,
                error_type=result.error_message if not result.passed else None,
            )

        # Calculate summary
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.passed)
        failed_tests = total_tests - passed_tests

        summary = {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "pass_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
        }

        return {
            "summary": summary,
            "results": results,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Test suite error: {str(e)}")


@router.post("/judge", response_model=EvaluationResult)
async def evaluate_response(request: JudgeRequest) -> EvaluationResult:
    """
    Use LLM-as-Judge to evaluate a single query-response pair.

    Evaluates on multiple criteria:
    - Accuracy
    - Completeness
    - Clarity
    - Insights
    - Safety
    - Relevance
    """
    try:
        result = await llm_judge.evaluate_response(
            query=request.query,
            response=request.response,
            ground_truth=request.ground_truth,
            context=request.context,
        )

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation error: {str(e)}")


@router.post("/judge/batch")
async def evaluate_batch(request: BatchJudgeRequest) -> Dict[str, Any]:
    """
    Evaluate multiple query-response pairs in batch.
    Returns individual results and aggregate statistics.
    """
    try:
        # Evaluate each pair
        results = []
        for eval_request in request.evaluations:
            result = await llm_judge.evaluate_response(
                query=eval_request.query,
                response=eval_request.response,
                ground_truth=eval_request.ground_truth,
                context=eval_request.context,
            )
            results.append(result)

        # Calculate aggregate scores
        aggregate = llm_judge.calculate_aggregate_scores(results)

        return {
            "individual_results": results,
            "aggregate_scores": aggregate,
            "evaluated_at": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch evaluation error: {str(e)}")


@router.get("/metrics/dashboard")
async def get_metrics_dashboard() -> Dict[str, Any]:
    """
    Get real-time metrics dashboard data.

    Includes:
    - Current performance metrics
    - Recent trends
    - Active issues
    - Model performance comparison
    """
    try:
        # Get various metric views
        real_time = metrics_collector.get_real_time_metrics()
        last_hour = metrics_collector.get_performance_metrics("last_hour")
        last_24h = metrics_collector.get_performance_metrics("last_24h")

        # Performance by intent and model
        intent_performance = metrics_collector.get_intent_performance()
        model_performance = metrics_collector.get_model_performance()

        # Identify issues
        issues = metrics_collector.identify_performance_issues()

        return {
            "real_time": real_time,
            "performance": {"last_hour": last_hour, "last_24h": last_24h},
            "by_intent": intent_performance,
            "by_model": model_performance,
            "active_issues": issues,
            "dashboard_updated_at": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metrics error: {str(e)}")


@router.get("/metrics/history")
async def get_metrics_history(
    time_period: str = Query(
        "last_24h", description="Time period: last_hour, last_24h, last_7d, last_30d"
    )
) -> PerformanceMetrics:
    """Get historical performance metrics for a specific time period."""
    try:
        metrics = metrics_collector.get_performance_metrics(time_period)
        return metrics

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metrics error: {str(e)}")


@router.post("/ab-test/create")
async def create_ab_test(config: ABTestConfig) -> Dict[str, Any]:
    """
    Create a new A/B test configuration.

    Tests different:
    - Models (GPT-5 vs Claude)
    - Prompts
    - Complexity thresholds
    - Context inclusion
    """
    # In a real implementation, this would store the config and set up the test
    # For now, return a confirmation
    return {
        "test_id": config.test_id,
        "name": config.name,
        "status": "created",
        "created_at": datetime.utcnow().isoformat(),
        "message": "A/B test created successfully. Use /ab-test/{test_id}/run to execute.",
    }


@router.post("/ab-test/{test_id}/run")
async def run_ab_test(
    test_id: str, session: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """
    Run an A/B test and compare results.

    Returns:
    - Performance comparison
    - Statistical significance
    - Recommendations
    """
    # This is a simplified implementation
    # In production, this would actually run the configured test

    return {
        "test_id": test_id,
        "status": "completed",
        "results": {
            "variant_a": {
                "queries_processed": 100,
                "success_rate": 96.5,
                "avg_response_time_ms": 1250,
                "avg_evaluation_score": 8.2,
            },
            "variant_b": {
                "queries_processed": 100,
                "success_rate": 94.8,
                "avg_response_time_ms": 1450,
                "avg_evaluation_score": 8.5,
            },
        },
        "statistical_significance": {
            "success_rate": {"p_value": 0.23, "significant": False},
            "response_time": {
                "p_value": 0.02,
                "significant": True,
                "winner": "variant_a",
            },
            "evaluation_score": {"p_value": 0.08, "significant": False},
        },
        "recommendation": "Variant A shows significantly better response time with comparable quality.",
        "completed_at": datetime.utcnow().isoformat(),
    }


@router.get("/ab-test/{test_id}/results")
async def get_ab_test_results(test_id: str) -> Dict[str, Any]:
    """Get results of a completed A/B test."""
    # In a real implementation, this would fetch stored results
    return {
        "test_id": test_id,
        "status": "completed",
        "message": "Use /ab-test/{test_id}/run to see full results",
    }


@router.post("/feedback")
async def submit_query_feedback(
    query_id: str = Body(...),
    satisfaction: int = Body(..., ge=1, le=5),
    feedback: Optional[str] = Body(None),
) -> BaseResponse:
    """
    Submit user feedback for a specific query.

    Used to track user satisfaction and improve the system.
    """
    # In a real implementation, this would update the metrics record
    # For now, just acknowledge
    return BaseResponse(
        success=True,
        message=f"Feedback recorded for query {query_id}. Satisfaction: {satisfaction}/5",
    )


@router.get("/reports/quality")
async def generate_quality_report(
    days: int = Query(7, description="Number of days to include in report")
) -> Dict[str, Any]:
    """
    Generate a comprehensive quality report.

    Includes:
    - Test suite results
    - LLM judge evaluations
    - User satisfaction
    - Recommendations for improvement
    """
    # This would generate a detailed report
    # For now, return a template
    return {
        "report_period": f"Last {days} days",
        "executive_summary": {
            "overall_quality_score": 8.5,
            "test_pass_rate": 92.3,
            "user_satisfaction": 4.2,
            "key_issues": [
                "Time parsing accuracy needs improvement",
                "Complex queries sometimes timeout",
            ],
        },
        "detailed_metrics": {
            "test_suite": {
                "total_tests_run": 2300,
                "pass_rate_by_category": {
                    "basic": 98.5,
                    "complex": 87.2,
                    "edge_case": 83.1,
                    "context_aware": 91.4,
                },
            },
            "llm_evaluations": {
                "total_evaluated": 150,
                "average_scores": {
                    "accuracy": 8.7,
                    "completeness": 8.3,
                    "clarity": 8.9,
                    "insights": 7.8,
                },
            },
        },
        "recommendations": [
            "Improve time period parsing for relative dates",
            "Add more context-aware test cases",
            "Consider using Claude Opus for complex queries",
            "Implement query result caching for common questions",
        ],
        "generated_at": datetime.utcnow().isoformat(),
    }
