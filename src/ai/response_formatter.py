"""
Response formatter for generating financial narratives from data.
Transforms raw query results into human-readable insights.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import json
from decimal import Decimal

from langchain.schema import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.ai.llm_client import MultiModelClient, QueryComplexity
from src.ai.query_processor import QueryIntent, ParsedQuery, SQLQuery


class FinancialNarrative(BaseModel):
    """Structured financial narrative response"""
    summary: str = Field(description="Executive summary of the findings")
    key_insights: List[str] = Field(description="Bullet points of key insights")
    supporting_data: Dict[str, Any] = Field(description="Formatted supporting data")
    recommendations: Optional[List[str]] = Field(default=None, description="Optional recommendations")
    confidence: float = Field(description="Confidence score 0-1")


class ResponseFormatter:
    """
    Generates human-readable financial narratives from query results.
    Handles different response types based on query intent.
    """
    
    def __init__(self, llm_client: Optional[MultiModelClient] = None):
        self.llm_client = llm_client or MultiModelClient()
        
    async def format_response(
        self,
        query: str,
        parsed_query: ParsedQuery,
        sql_query: SQLQuery,
        data: List[Dict],
        context: Optional[Dict] = None
    ) -> FinancialNarrative:
        """
        Format query results into a financial narrative.
        
        Args:
            query: Original user query
            parsed_query: Parsed query object
            sql_query: Generated SQL query
            data: Raw query results
            context: Optional context
            
        Returns:
            FinancialNarrative with formatted response
        """
        # Format data based on query intent
        formatted_data = self._format_data(data, parsed_query)
        
        # Generate narrative based on intent
        if parsed_query.intent == QueryIntent.AGGREGATION:
            return await self._format_aggregation_response(
                query, parsed_query, formatted_data
            )
        elif parsed_query.intent == QueryIntent.COMPARISON:
            return await self._format_comparison_response(
                query, parsed_query, formatted_data
            )
        elif parsed_query.intent == QueryIntent.TREND:
            return await self._format_trend_response(
                query, parsed_query, formatted_data
            )
        elif parsed_query.intent == QueryIntent.RANKING:
            return await self._format_ranking_response(
                query, parsed_query, formatted_data
            )
        elif parsed_query.intent == QueryIntent.CALCULATION:
            return await self._format_calculation_response(
                query, parsed_query, formatted_data
            )
        else:
            return await self._format_generic_response(
                query, parsed_query, formatted_data
            )
    
    def _format_data(self, data: List[Dict], parsed_query: ParsedQuery) -> Dict[str, Any]:
        """Format raw data for presentation"""
        if not data:
            return {"empty": True, "message": "No data found for the specified criteria"}
        
        # Format numbers
        formatted = []
        for row in data:
            formatted_row = {}
            for key, value in row.items():
                if isinstance(value, (int, float, Decimal)):
                    formatted_row[key] = self._format_number(value)
                elif isinstance(value, datetime):
                    formatted_row[key] = value.strftime("%Y-%m-%d")
                else:
                    formatted_row[key] = value
            formatted.append(formatted_row)
        
        return {
            "rows": formatted,
            "count": len(formatted),
            "metrics": parsed_query.metrics,
            "period": self._format_period(parsed_query)
        }
    
    def _format_number(self, value: float) -> str:
        """Format numbers for readability"""
        if value >= 1_000_000:
            return f"${value/1_000_000:.1f}M"
        elif value >= 1_000:
            return f"${value/1_000:.1f}K"
        else:
            return f"${value:,.2f}"
    
    def _format_period(self, parsed_query: ParsedQuery) -> str:
        """Format time period for display"""
        if not parsed_query.time_period:
            return "all time"
        
        period = parsed_query.time_period
        if hasattr(period, 'start_date') and hasattr(period, 'end_date'):
            start = period.start_date.strftime("%b %Y")
            end = period.end_date.strftime("%b %Y")
            if start == end:
                return start
            return f"{start} to {end}"
        return str(period)
    
    async def _format_aggregation_response(
        self,
        query: str,
        parsed_query: ParsedQuery,
        data: Dict[str, Any]
    ) -> FinancialNarrative:
        """Format aggregation query response"""
        if data.get("empty"):
            return FinancialNarrative(
                summary=data["message"],
                key_insights=[],
                supporting_data={},
                confidence=1.0
            )
        
        # Generate natural language summary
        prompt = f"""
        Generate a financial summary for this query: "{query}"
        
        Data: {json.dumps(data['rows'], indent=2)}
        Metrics: {', '.join(parsed_query.metrics)}
        Period: {data['period']}
        
        Provide a clear, concise summary that answers the user's question directly.
        """
        
        messages = [
            SystemMessage(content="You are a financial analyst providing clear insights."),
            HumanMessage(content=prompt)
        ]
        
        response = await self.llm_client.query(
            messages,
            complexity=QueryComplexity.SIMPLE
        )
        
        # Extract key values
        key_insights = []
        if data['rows']:
            row = data['rows'][0]
            for metric in parsed_query.metrics:
                if metric in row:
                    key_insights.append(f"{metric.title()}: {row[metric]}")
        
        return FinancialNarrative(
            summary=response,
            key_insights=key_insights,
            supporting_data=data,
            confidence=0.95
        )
    
    async def _format_comparison_response(
        self,
        query: str,
        parsed_query: ParsedQuery,
        data: Dict[str, Any]
    ) -> FinancialNarrative:
        """Format comparison query response"""
        if data.get("empty"):
            return FinancialNarrative(
                summary=data["message"],
                key_insights=[],
                supporting_data={},
                confidence=1.0
            )
        
        # Calculate comparison metrics
        if len(data['rows']) >= 2:
            row1, row2 = data['rows'][0], data['rows'][1]
            comparisons = []
            
            for metric in parsed_query.metrics:
                if metric in row1 and metric in row2:
                    val1 = self._extract_number(row1[metric])
                    val2 = self._extract_number(row2[metric])
                    if val1 and val2 and val1 != 0:
                        change = ((val2 - val1) / val1) * 100
                        direction = "increased" if change > 0 else "decreased"
                        comparisons.append(
                            f"{metric.title()} {direction} by {abs(change):.1f}%"
                        )
        
        prompt = f"""
        Generate a comparison analysis for: "{query}"
        
        Data: {json.dumps(data['rows'], indent=2)}
        
        Highlight the key differences and trends between the periods.
        """
        
        messages = [
            SystemMessage(content="You are a financial analyst specializing in comparative analysis."),
            HumanMessage(content=prompt)
        ]
        
        response = await self.llm_client.query(
            messages,
            complexity=QueryComplexity.MEDIUM
        )
        
        return FinancialNarrative(
            summary=response,
            key_insights=comparisons if 'comparisons' in locals() else [],
            supporting_data=data,
            recommendations=["Monitor trends closely", "Investigate significant changes"],
            confidence=0.9
        )
    
    async def _format_trend_response(
        self,
        query: str,
        parsed_query: ParsedQuery,
        data: Dict[str, Any]
    ) -> FinancialNarrative:
        """Format trend analysis response"""
        if data.get("empty"):
            return FinancialNarrative(
                summary=data["message"],
                key_insights=[],
                supporting_data={},
                confidence=1.0
            )
        
        # Analyze trends
        insights = []
        if len(data['rows']) > 1:
            # Simple trend detection
            first_val = self._extract_number(data['rows'][0].get(parsed_query.metrics[0], 0))
            last_val = self._extract_number(data['rows'][-1].get(parsed_query.metrics[0], 0))
            
            if first_val and last_val:
                trend = "upward" if last_val > first_val else "downward"
                insights.append(f"Overall {trend} trend observed")
        
        prompt = f"""
        Analyze the trend for: "{query}"
        
        Data points: {json.dumps(data['rows'], indent=2)}
        
        Identify patterns, trends, and notable changes over time.
        """
        
        messages = [
            SystemMessage(content="You are a financial analyst expert in trend analysis."),
            HumanMessage(content=prompt)
        ]
        
        response = await self.llm_client.query(
            messages,
            complexity=QueryComplexity.MEDIUM
        )
        
        return FinancialNarrative(
            summary=response,
            key_insights=insights,
            supporting_data=data,
            recommendations=["Continue monitoring trend", "Consider seasonal factors"],
            confidence=0.85
        )
    
    async def _format_ranking_response(
        self,
        query: str,
        parsed_query: ParsedQuery,
        data: Dict[str, Any]
    ) -> FinancialNarrative:
        """Format ranking query response"""
        if data.get("empty"):
            return FinancialNarrative(
                summary=data["message"],
                key_insights=[],
                supporting_data={},
                confidence=1.0
            )
        
        # Extract top items
        insights = []
        for i, row in enumerate(data['rows'][:5], 1):
            item_name = row.get('category', row.get('name', f'Item {i}'))
            value = row.get(parsed_query.metrics[0], 'N/A')
            insights.append(f"#{i}: {item_name} - {value}")
        
        prompt = f"""
        Summarize this ranking analysis: "{query}"
        
        Top items: {json.dumps(data['rows'][:10], indent=2)}
        
        Highlight the leaders and any notable patterns.
        """
        
        messages = [
            SystemMessage(content="You are a financial analyst providing ranking insights."),
            HumanMessage(content=prompt)
        ]
        
        response = await self.llm_client.query(
            messages,
            complexity=QueryComplexity.SIMPLE
        )
        
        return FinancialNarrative(
            summary=response,
            key_insights=insights,
            supporting_data=data,
            confidence=0.9
        )
    
    async def _format_calculation_response(
        self,
        query: str,
        parsed_query: ParsedQuery,
        data: Dict[str, Any]
    ) -> FinancialNarrative:
        """Format calculation query response"""
        if data.get("empty"):
            return FinancialNarrative(
                summary=data["message"],
                key_insights=[],
                supporting_data={},
                confidence=1.0
            )
        
        # Extract calculated values
        insights = []
        if data['rows']:
            row = data['rows'][0]
            for key, value in row.items():
                if 'margin' in key.lower() or 'ratio' in key.lower() or 'percent' in key.lower():
                    insights.append(f"{key.replace('_', ' ').title()}: {value}")
        
        prompt = f"""
        Explain this financial calculation: "{query}"
        
        Results: {json.dumps(data['rows'], indent=2)}
        
        Provide context and interpretation of the calculated values.
        """
        
        messages = [
            SystemMessage(content="You are a financial analyst explaining calculations."),
            HumanMessage(content=prompt)
        ]
        
        response = await self.llm_client.query(
            messages,
            complexity=QueryComplexity.MEDIUM
        )
        
        return FinancialNarrative(
            summary=response,
            key_insights=insights,
            supporting_data=data,
            confidence=0.9
        )
    
    async def _format_generic_response(
        self,
        query: str,
        parsed_query: ParsedQuery,
        data: Dict[str, Any]
    ) -> FinancialNarrative:
        """Format generic query response"""
        prompt = f"""
        Answer this financial query: "{query}"
        
        Data: {json.dumps(data, indent=2)}
        
        Provide a clear, comprehensive response.
        """
        
        messages = [
            SystemMessage(content="You are a helpful financial analyst."),
            HumanMessage(content=prompt)
        ]
        
        response = await self.llm_client.query(
            messages,
            complexity=QueryComplexity.SIMPLE
        )
        
        return FinancialNarrative(
            summary=response,
            key_insights=["Data retrieved successfully"],
            supporting_data=data,
            confidence=0.8
        )
    
    def _extract_number(self, value: str) -> Optional[float]:
        """Extract numeric value from formatted string"""
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            # Remove $ and commas
            cleaned = value.replace('$', '').replace(',', '').replace('M', '000000').replace('K', '000')
            try:
                return float(cleaned)
            except ValueError:
                return None
        return None
