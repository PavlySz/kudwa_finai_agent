"""
API endpoints for natural language queries and AI-powered financial insights.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field

from src.config.database import get_db
from src.services.data_import import DataQueryService
from src.ai.llm_client import MultiModelClient
from src.ai.query_processor import QueryProcessor
from src.ai.response_formatter import ResponseFormatter
from src.ai.context_manager import ContextManager
from src.models.base import BaseResponse


# Initialize AI components
llm_client = MultiModelClient()
query_processor = QueryProcessor(llm_client)
response_formatter = ResponseFormatter(llm_client)
context_manager = ContextManager(llm_client)

# Create router
router = APIRouter(prefix="/api/queries", tags=["AI Queries"])


# Request/Response models
class NaturalQueryRequest(BaseModel):
    """Request model for natural language queries"""

    query: str = Field(description="Natural language query about financial data")
    session_id: Optional[str] = Field(
        default=None, description="Session ID for conversation context"
    )
    context: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional context (e.g., company_id)"
    )
    stream: bool = Field(default=False, description="Stream the response")
    model_preference: Optional[str] = Field(
        default=None, description="Preferred model to use"
    )


class QueryMetadata(BaseModel):
    """Metadata about query processing"""

    intent: str
    complexity: str
    time_period: Optional[str] = None
    metrics: List[str]
    sql_generated: Optional[str] = None
    processing_time_ms: int
    model_used: str
    confidence: float


class NaturalQueryResponse(BaseModel):
    """Response model for natural language queries"""

    success: bool
    session_id: str
    answer: str
    key_insights: List[str]
    supporting_data: Dict[str, Any]
    recommendations: Optional[List[str]] = None
    metadata: QueryMetadata


class SessionInfo(BaseModel):
    """Information about a conversation session"""

    session_id: str
    started_at: datetime
    turns_count: int
    entities: Dict[str, Any]
    summary: Optional[str] = None


class FeedbackRequest(BaseModel):
    """User feedback for query responses"""

    session_id: str
    query_id: str
    rating: int = Field(ge=1, le=5, description="Rating from 1-5")
    feedback: Optional[str] = None


@router.post("/natural", response_model=NaturalQueryResponse)
async def natural_language_query(
    request: NaturalQueryRequest, session: AsyncSession = Depends(get_db)
) -> NaturalQueryResponse:
    """
    Process natural language queries about financial data.

    Examples:
    - "What was the total revenue in Q1 2024?"
    - "Compare expenses between January and February"
    - "Show me the top 5 expense categories this year"
    - "What's the profit margin trend over the last 6 months?"
    """
    start_time = datetime.utcnow()

    # Generate session ID if not provided
    session_id = request.session_id or str(uuid4())

    try:
        # Get conversation context
        context_data = context_manager.get_context_for_query(session_id, request.query)

        # Add any request context
        if request.context:
            context_data.update(request.context)

        # Process the query
        parsed_query, sql_query = await query_processor.process_query(
            request.query, context_data
        )

        # Resolve any references from context
        parsed_query = context_manager.resolve_references(
            request.query, parsed_query, context_data
        )

        # Check if this is a forecast or anomaly detection query
        if parsed_query.intent == "forecast":
            # Handle forecast queries
            result = await _handle_forecast_query(
                parsed_query, sql_query, session, query_processor
            )
            data = result.get("data", [])
            # Add forecast results to context
            context_data["forecast_results"] = result.get("forecast", {})
        elif parsed_query.intent == "anomaly":
            # Handle anomaly detection queries
            result = await _handle_anomaly_query(
                parsed_query, sql_query, session, query_processor
            )
            data = result.get("data", [])
            # Add anomaly results to context
            context_data["anomaly_results"] = result.get("anomalies", {})
        else:
            # Execute the SQL query normally
            query_service = DataQueryService(session)

            # For now, execute the SQL directly (in production, use proper ORM queries)
            result = await session.execute(sql_query.query, sql_query.parameters or {})
            data = [dict(row) for row in result]

        # Format the response
        narrative = await response_formatter.format_response(
            request.query, parsed_query, sql_query, data, context_data
        )

        # Update conversation context
        await context_manager.add_turn(
            session_id,
            request.query,
            parsed_query,
            sql_query,
            narrative.summary,
            {"row_count": len(data), "has_data": len(data) > 0},
        )

        # Calculate processing time
        processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)

        # Get model used
        model_used = llm_client.last_model_used or "unknown"

        return NaturalQueryResponse(
            success=True,
            session_id=session_id,
            answer=narrative.summary,
            key_insights=narrative.key_insights,
            supporting_data=narrative.supporting_data,
            recommendations=narrative.recommendations,
            metadata=QueryMetadata(
                intent=parsed_query.intent.value,
                complexity=sql_query.complexity.value,
                time_period=(
                    str(parsed_query.time_period) if parsed_query.time_period else None
                ),
                metrics=parsed_query.metrics,
                sql_generated=sql_query.query if not sql_query.is_safe else None,
                processing_time_ms=processing_time,
                model_used=model_used,
                confidence=narrative.confidence,
            ),
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Log the error
        print(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process query")


@router.post("/natural/stream")
async def natural_language_query_stream(
    request: NaturalQueryRequest, session: AsyncSession = Depends(get_db)
):
    """
    Stream natural language query responses for better UX.
    Returns Server-Sent Events (SSE) stream.
    """

    async def generate():
        try:
            # Send initial acknowledgment
            yield f"data: {{'type': 'start', 'session_id': '{request.session_id or str(uuid4())}'}}\n\n"

            # Get context
            session_id = request.session_id or str(uuid4())
            context_data = context_manager.get_context_for_query(
                session_id, request.query
            )

            # Send parsing status
            yield f"data: {{'type': 'status', 'message': 'Understanding your query...'}}\n\n"

            # Parse query
            parsed_query, sql_query = await query_processor.process_query(
                request.query, context_data
            )

            yield f"data: {{'type': 'intent', 'intent': '{parsed_query.intent.value}'}}\n\n"

            # Execute query
            yield f"data: {{'type': 'status', 'message': 'Fetching data...'}}\n\n"

            query_service = DataQueryService(session)
            result = await session.execute(sql_query.query, sql_query.parameters or {})
            data = [dict(row) for row in result]

            yield f"data: {{'type': 'data_count', 'count': {len(data)}}}\n\n"

            # Stream the narrative generation
            yield f"data: {{'type': 'status', 'message': 'Generating insights...'}}\n\n"

            # For now, generate full response (in future, could stream token by token)
            narrative = await response_formatter.format_response(
                request.query, parsed_query, sql_query, data, context_data
            )

            # Send response in chunks
            words = narrative.summary.split()
            chunk_size = 10
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i : i + chunk_size])
                yield f"data: {{'type': 'content', 'text': '{chunk} '}}\n\n"
                await asyncio.sleep(0.05)  # Small delay for streaming effect

            # Send insights
            yield f"data: {{'type': 'insights', 'insights': {narrative.key_insights}}}\n\n"

            # Send completion
            yield f"data: {{'type': 'complete', 'confidence': {narrative.confidence}}}\n\n"

        except Exception as e:
            yield f"data: {{'type': 'error', 'message': '{str(e)}'}}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.get("/sessions/{session_id}", response_model=SessionInfo)
async def get_session_info(session_id: str) -> SessionInfo:
    """Get information about a conversation session"""
    context = context_manager.sessions.get(session_id)
    if not context:
        raise HTTPException(status_code=404, detail="Session not found")

    summary = context_manager.get_session_summary(session_id)

    return SessionInfo(
        session_id=session_id,
        started_at=context.started_at,
        turns_count=len(context.turns),
        entities=context.entities,
        summary=summary,
    )


@router.delete("/sessions/{session_id}")
async def clear_session(session_id: str) -> BaseResponse:
    """Clear a conversation session"""
    context_manager.clear_session(session_id)
    return BaseResponse(success=True, message=f"Session {session_id} cleared")


@router.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest) -> BaseResponse:
    """Submit feedback for a query response"""
    # In a real implementation, this would store feedback in a database
    # For now, just log it
    print(
        f"Feedback received: Session {feedback.session_id}, "
        f"Rating: {feedback.rating}, Feedback: {feedback.feedback}"
    )

    return BaseResponse(success=True, message="Feedback received. Thank you!")


@router.get("/examples")
async def get_query_examples() -> Dict[str, List[str]]:
    """Get example queries organized by category"""
    return {
        "simple_queries": [
            "What was the total revenue last quarter?",
            "Show me all expenses for March 2024",
            "What's our current cash position?",
            "How much did we spend on payroll this year?",
        ],
        "comparison_queries": [
            "Compare revenue between Q1 and Q2",
            "How did expenses change from last month?",
            "What's the difference in profit margin between 2023 and 2024?",
            "Compare top expense categories year over year",
        ],
        "trend_queries": [
            "Show me the revenue trend for the last 6 months",
            "What's the monthly burn rate trend?",
            "How has our profit margin evolved this year?",
            "Display cash flow trends by quarter",
        ],
        "ranking_queries": [
            "What are the top 5 expense categories?",
            "Show me the highest revenue months",
            "Which categories have the lowest margins?",
            "Rank quarters by profitability",
        ],
        "calculation_queries": [
            "Calculate the gross margin for Q1",
            "What's the year-to-date profit margin?",
            "Show me the expense ratio by category",
            "Calculate monthly growth rate",
        ],
        "follow_up_queries": [
            "What about the previous quarter?",
            "Break that down by category",
            "Show me the same for expenses",
            "Can you compare that to last year?",
        ],
    }


@router.get("/health")
async def ai_health_check() -> Dict[str, Any]:
    """Check AI service health and model availability"""
    models_status = {}

    # Check each configured model
    for model_name, model in llm_client.models.items():
        try:
            # Simple test query
            test_response = await model.ainvoke("Say 'OK'")
            models_status[model_name] = "healthy"
        except Exception as e:
            models_status[model_name] = f"error: {str(e)[:50]}"

    return {
        "service": "ai_queries",
        "status": (
            "healthy"
            if any(s == "healthy" for s in models_status.values())
            else "degraded"
        ),
        "models": models_status,
        "default_model": llm_client.default_model,
        "sessions_active": len(context_manager.sessions),
    }


# Helper functions for analytics
async def _handle_forecast_query(
    parsed_query, sql_query, session: AsyncSession, query_processor: QueryProcessor
) -> Dict[str, Any]:
    """Handle forecast queries using the forecaster"""
    # First get historical data
    result = await session.execute(sql_query.query, sql_query.parameters or {})
    historical_data = [dict(row) for row in result]

    if not historical_data:
        return {
            "data": [],
            "forecast": {"error": "No historical data available for forecasting"},
        }

    # Extract the primary metric from the query
    metric = parsed_query.metrics[0] if parsed_query.metrics else "value"

    # Determine forecast periods (default to 3)
    periods = 3
    if "month" in parsed_query.original_query.lower():
        periods = 1
    elif "quarter" in parsed_query.original_query.lower():
        periods = 3
    elif "year" in parsed_query.original_query.lower():
        periods = 12

    # Run forecast
    forecast_result = await query_processor.forecaster.forecast(
        metric=metric, historical_data=historical_data, periods=periods
    )

    return {"data": historical_data, "forecast": forecast_result}


async def _handle_anomaly_query(
    parsed_query, sql_query, session: AsyncSession, query_processor: QueryProcessor
) -> Dict[str, Any]:
    """Handle anomaly detection queries"""
    # Get data for analysis
    result = await session.execute(sql_query.query, sql_query.parameters or {})
    data = [dict(row) for row in result]

    if not data:
        return {
            "data": [],
            "anomalies": {"error": "No data available for anomaly detection"},
        }

    # Extract the primary metric
    metric = parsed_query.metrics[0] if parsed_query.metrics else "value"

    # Detect anomalies
    anomaly_result = await query_processor.anomaly_detector.detect_anomalies(
        data=data, metric=metric, method="zscore"  # Could be made configurable
    )

    return {"data": data, "anomalies": anomaly_result}
