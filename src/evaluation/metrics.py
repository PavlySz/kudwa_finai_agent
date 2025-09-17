"""
Metrics collection and tracking for AI system performance.
Tracks query success rates, response times, token usage, and more.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict, deque
from pydantic import BaseModel, Field
import statistics

from src.ai.query_processor import QueryIntent


class QueryMetrics(BaseModel):
    """Metrics for a single query"""

    query_id: str
    timestamp: datetime
    query: str
    intent: str
    complexity: str
    model_used: str
    success: bool
    response_time_ms: int
    tokens_used: Optional[int] = None
    cost_estimate: Optional[float] = None
    error_type: Optional[str] = None
    user_satisfaction: Optional[int] = Field(None, ge=1, le=5)


class PerformanceMetrics(BaseModel):
    """Aggregated performance metrics"""

    time_period: str
    total_queries: int
    success_rate: float
    average_response_time_ms: float
    p50_response_time_ms: float
    p90_response_time_ms: float
    p99_response_time_ms: float
    queries_by_intent: Dict[str, int]
    queries_by_model: Dict[str, int]
    total_tokens: int
    total_cost: float
    error_rate: float
    errors_by_type: Dict[str, int]
    average_user_satisfaction: Optional[float] = None


class MetricsCollector:
    """
    Collects and analyzes metrics for the AI system.
    Provides real-time and historical performance data.
    """

    def __init__(self, retention_days: int = 30):
        self.metrics: deque = deque(maxlen=retention_days * 10000)  # Rough capacity
        self.retention_days = retention_days

        # Real-time tracking
        self.current_hour_metrics = []
        self.current_hour = datetime.utcnow().hour

        # Model costs (example rates)
        self.model_costs = {
            "gpt-5": 0.01,  # per 1k tokens
            "claude-sonnet-4-20250514": 0.015,
            "claude-opus-4-1-20250805": 0.02,
        }

    def record_query(
        self,
        query_id: str,
        query: str,
        intent: str,
        complexity: str,
        model_used: str,
        success: bool,
        response_time_ms: int,
        tokens_used: Optional[int] = None,
        error_type: Optional[str] = None,
        user_satisfaction: Optional[int] = None,
    ) -> QueryMetrics:
        """Record metrics for a single query"""
        # Estimate cost if tokens provided
        cost_estimate = None
        if tokens_used and model_used in self.model_costs:
            cost_estimate = (tokens_used / 1000) * self.model_costs[model_used]

        metrics = QueryMetrics(
            query_id=query_id,
            timestamp=datetime.utcnow(),
            query=query,
            intent=intent,
            complexity=complexity,
            model_used=model_used,
            success=success,
            response_time_ms=response_time_ms,
            tokens_used=tokens_used,
            cost_estimate=cost_estimate,
            error_type=error_type,
            user_satisfaction=user_satisfaction,
        )

        # Add to collections
        self.metrics.append(metrics)
        self._update_current_hour(metrics)

        return metrics

    def _update_current_hour(self, metrics: QueryMetrics):
        """Update current hour metrics"""
        current_hour = datetime.utcnow().hour

        # Reset if new hour
        if current_hour != self.current_hour:
            self.current_hour_metrics = []
            self.current_hour = current_hour

        self.current_hour_metrics.append(metrics)

    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time metrics for monitoring dashboard"""
        now = datetime.utcnow()

        # Last 5 minutes
        five_min_ago = now - timedelta(minutes=5)
        recent_metrics = [
            m for m in self.current_hour_metrics if m.timestamp >= five_min_ago
        ]

        if not recent_metrics:
            return {
                "queries_per_minute": 0,
                "success_rate": 0,
                "average_response_time_ms": 0,
                "active_sessions": 0,
                "errors_last_5min": 0,
            }

        # Calculate real-time stats
        success_count = sum(1 for m in recent_metrics if m.success)
        error_count = len(recent_metrics) - success_count
        response_times = [m.response_time_ms for m in recent_metrics]

        return {
            "queries_per_minute": len(recent_metrics) / 5,
            "success_rate": (success_count / len(recent_metrics)) * 100,
            "average_response_time_ms": statistics.mean(response_times),
            "median_response_time_ms": statistics.median(response_times),
            "active_sessions": len(
                set(m.query_id.split("-")[0] for m in recent_metrics)
            ),
            "errors_last_5min": error_count,
            "queries_by_model": self._count_by_attribute(recent_metrics, "model_used"),
        }

    def get_performance_metrics(
        self, time_period: str = "last_24h"
    ) -> PerformanceMetrics:
        """Get aggregated performance metrics for a time period"""
        # Determine time range
        now = datetime.utcnow()
        if time_period == "last_hour":
            start_time = now - timedelta(hours=1)
        elif time_period == "last_24h":
            start_time = now - timedelta(days=1)
        elif time_period == "last_7d":
            start_time = now - timedelta(days=7)
        elif time_period == "last_30d":
            start_time = now - timedelta(days=30)
        else:
            start_time = now - timedelta(days=1)

        # Filter metrics
        period_metrics = [m for m in self.metrics if m.timestamp >= start_time]

        if not period_metrics:
            return PerformanceMetrics(
                time_period=time_period,
                total_queries=0,
                success_rate=0,
                average_response_time_ms=0,
                p50_response_time_ms=0,
                p90_response_time_ms=0,
                p99_response_time_ms=0,
                queries_by_intent={},
                queries_by_model={},
                total_tokens=0,
                total_cost=0,
                error_rate=0,
                errors_by_type={},
            )

        # Calculate metrics
        success_count = sum(1 for m in period_metrics if m.success)
        error_count = len(period_metrics) - success_count
        response_times = sorted([m.response_time_ms for m in period_metrics])

        # Percentiles
        p50_idx = int(len(response_times) * 0.5)
        p90_idx = int(len(response_times) * 0.9)
        p99_idx = int(len(response_times) * 0.99)

        # User satisfaction
        satisfaction_scores = [
            m.user_satisfaction for m in period_metrics if m.user_satisfaction
        ]
        avg_satisfaction = (
            statistics.mean(satisfaction_scores) if satisfaction_scores else None
        )

        return PerformanceMetrics(
            time_period=time_period,
            total_queries=len(period_metrics),
            success_rate=(success_count / len(period_metrics)) * 100,
            average_response_time_ms=statistics.mean(response_times),
            p50_response_time_ms=(
                response_times[p50_idx] if p50_idx < len(response_times) else 0
            ),
            p90_response_time_ms=(
                response_times[p90_idx] if p90_idx < len(response_times) else 0
            ),
            p99_response_time_ms=(
                response_times[p99_idx] if p99_idx < len(response_times) else 0
            ),
            queries_by_intent=self._count_by_attribute(period_metrics, "intent"),
            queries_by_model=self._count_by_attribute(period_metrics, "model_used"),
            total_tokens=sum(m.tokens_used or 0 for m in period_metrics),
            total_cost=sum(m.cost_estimate or 0 for m in period_metrics),
            error_rate=(error_count / len(period_metrics)) * 100,
            errors_by_type=self._count_errors(period_metrics),
            average_user_satisfaction=avg_satisfaction,
        )

    def _count_by_attribute(
        self, metrics: List[QueryMetrics], attribute: str
    ) -> Dict[str, int]:
        """Count metrics by a specific attribute"""
        counts = defaultdict(int)
        for m in metrics:
            value = getattr(m, attribute)
            counts[value] += 1
        return dict(counts)

    def _count_errors(self, metrics: List[QueryMetrics]) -> Dict[str, int]:
        """Count errors by type"""
        error_counts = defaultdict(int)
        for m in metrics:
            if not m.success and m.error_type:
                error_counts[m.error_type] += 1
        return dict(error_counts)

    def get_intent_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get performance breakdown by intent type"""
        intent_metrics = defaultdict(list)

        # Group by intent
        for m in self.metrics:
            intent_metrics[m.intent].append(m)

        results = {}
        for intent, metrics in intent_metrics.items():
            if metrics:
                success_count = sum(1 for m in metrics if m.success)
                response_times = [m.response_time_ms for m in metrics]

                results[intent] = {
                    "total_queries": len(metrics),
                    "success_rate": (success_count / len(metrics)) * 100,
                    "average_response_time_ms": statistics.mean(response_times),
                    "median_response_time_ms": statistics.median(response_times),
                }

        return results

    def get_model_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get performance breakdown by model"""
        model_metrics = defaultdict(list)

        # Group by model
        for m in self.metrics:
            model_metrics[m.model_used].append(m)

        results = {}
        for model, metrics in model_metrics.items():
            if metrics:
                success_count = sum(1 for m in metrics if m.success)
                response_times = [m.response_time_ms for m in metrics]
                total_tokens = sum(m.tokens_used or 0 for m in metrics)
                total_cost = sum(m.cost_estimate or 0 for m in metrics)

                results[model] = {
                    "total_queries": len(metrics),
                    "success_rate": (success_count / len(metrics)) * 100,
                    "average_response_time_ms": statistics.mean(response_times),
                    "total_tokens": total_tokens,
                    "total_cost": round(total_cost, 2),
                    "cost_per_query": (
                        round(total_cost / len(metrics), 4) if metrics else 0
                    ),
                }

        return results

    def identify_performance_issues(self) -> List[Dict[str, Any]]:
        """Identify potential performance issues"""
        issues = []

        # Get recent metrics
        recent_metrics = self.get_performance_metrics("last_hour")

        # Check success rate
        if recent_metrics.success_rate < 95:
            issues.append(
                {
                    "type": "low_success_rate",
                    "severity": "high",
                    "value": recent_metrics.success_rate,
                    "threshold": 95,
                    "message": f"Success rate ({recent_metrics.success_rate:.1f}%) below threshold",
                }
            )

        # Check response time
        if recent_metrics.p90_response_time_ms > 3000:
            issues.append(
                {
                    "type": "high_response_time",
                    "severity": "medium",
                    "value": recent_metrics.p90_response_time_ms,
                    "threshold": 3000,
                    "message": f"P90 response time ({recent_metrics.p90_response_time_ms}ms) exceeds 3 seconds",
                }
            )

        # Check error patterns
        if recent_metrics.errors_by_type:
            most_common_error = max(
                recent_metrics.errors_by_type.items(), key=lambda x: x[1]
            )
            if most_common_error[1] > 5:
                issues.append(
                    {
                        "type": "frequent_errors",
                        "severity": "high",
                        "error_type": most_common_error[0],
                        "count": most_common_error[1],
                        "message": f"Frequent {most_common_error[0]} errors ({most_common_error[1]} occurrences)",
                    }
                )

        return issues

    def export_metrics(self, format: str = "json") -> Any:
        """Export metrics for analysis"""
        if format == "json":
            return [m.dict() for m in self.metrics]
        else:
            # Could add CSV export, etc.
            raise ValueError(f"Unsupported export format: {format}")
