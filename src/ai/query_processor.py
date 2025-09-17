"""
Query processor for converting natural language to SQL queries
Uses LangChain for structured query interpretation and SQL generation
"""

import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from enum import Enum

from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import BaseMessage, SystemMessage, HumanMessage

from src.config.settings import settings
from src.ai.llm_client import MultiModelClient, QueryComplexity


class QueryIntent(str, Enum):
    """Types of query intents"""

    LOOKUP = "lookup"  # Simple data retrieval
    AGGREGATION = "aggregation"  # Sum, average, count
    COMPARISON = "comparison"  # Compare periods or metrics
    TREND = "trend"  # Analyze trends over time
    RANKING = "ranking"  # Top/bottom N items
    CALCULATION = "calculation"  # Profit margins, ratios


class TimeGranularity(str, Enum):
    """Time period granularity"""

    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"


class ParsedQuery(BaseModel):
    """Structured representation of a parsed query"""

    intent: QueryIntent = Field(description="The primary intent of the query")
    metrics: List[str] = Field(
        description="Financial metrics requested (e.g., revenue, expenses, profit)"
    )
    time_period: Optional[str] = Field(
        description="Time period mentioned in natural language"
    )
    time_start: Optional[datetime] = Field(description="Parsed start date")
    time_end: Optional[datetime] = Field(description="Parsed end date")
    granularity: Optional[TimeGranularity] = Field(
        description="Time granularity for grouping"
    )
    filters: Dict[str, Any] = Field(
        default_factory=dict, description="Additional filters (e.g., company, category)"
    )
    aggregations: List[str] = Field(
        default_factory=list, description="Aggregation functions needed"
    )
    limit: Optional[int] = Field(description="Limit for top/bottom queries")
    comparison_period: Optional[str] = Field(description="Period to compare against")


class SQLQuery(BaseModel):
    """Generated SQL query with metadata"""

    query: str = Field(description="The SQL query string")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Query parameters for safe execution"
    )
    explanation: str = Field(
        description="Human-readable explanation of what the query does"
    )
    complexity: QueryComplexity = Field(description="Query complexity level")


class QueryProcessor:
    """
    Converts natural language queries to SQL using LangChain
    """

    def __init__(self, llm_client: Optional[MultiModelClient] = None):
        """Initialize with LLM client"""
        self.llm_client = llm_client or MultiModelClient()

        # Common financial metric mappings
        self.metric_mappings = {
            "revenue": ["revenue", "income", "sales"],
            "expenses": ["expenses", "costs", "expenditure"],
            "profit": ["profit", "net income", "earnings"],
            "cash": ["cash", "cash flow"],
            "margin": ["margin", "profit margin"],
        }

        # Time period patterns - ordered from most specific to least specific
        self.time_patterns = {
            r"[Qq](\d) (\d{4})": self._parse_quarter,
            r"(january|february|march|april|may|june|july|august|september|october|november|december) (\d{4})": self._parse_month,
            r"last (\d+) (days?|weeks?|months?|quarters?|years?)": self._parse_relative,
            r"(ytd|year to date)": self._parse_ytd,
            r"(this|current) (month|quarter|year)": self._parse_current,
            r"(\d{4})": self._parse_year,  # Moved to end to avoid matching years in "Q1 2024"
        }

    async def process_query(
        self, query: str, context: Optional[Dict] = None
    ) -> Tuple[ParsedQuery, SQLQuery]:
        """
        Process natural language query into structured format and SQL

        Args:
            query: Natural language query
            context: Optional context (company_id, user preferences, etc.)

        Returns:
            Tuple of (ParsedQuery, SQLQuery)
        """
        # Validate query
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        # Check for SQL injection attempts
        query_lower = query.lower()
        dangerous_keywords = ["delete", "drop", "truncate", "update", "insert", "alter"]
        if any(keyword in query_lower for keyword in dangerous_keywords):
            raise ValueError(f"Query contains potentially dangerous SQL keywords")

        # Step 1: Parse the natural language query
        parsed_query = await self._parse_query(query, context)

        # Validate that we got valid financial metrics
        if not parsed_query.metrics or len(parsed_query.metrics) == 0:
            # Check if this is an off-topic query
            finance_keywords = [
                "revenue",
                "expense",
                "profit",
                "cash",
                "cost",
                "margin",
                "sales",
                "income",
            ]
            if not any(keyword in query.lower() for keyword in finance_keywords):
                raise ValueError("Query does not appear to be about financial data")

        # Step 2: Generate SQL from parsed query
        sql_query = await self._generate_sql(parsed_query, context)

        return parsed_query, sql_query

    async def _parse_query(
        self, query: str, context: Optional[Dict] = None
    ) -> ParsedQuery:
        """Parse natural language query into structured format"""

        # Create parser
        parser = PydanticOutputParser(pydantic_object=ParsedQuery)

        # Create prompt template
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(
                    """You are a financial query interpreter. Analyze the user's query and extract structured information.
                
                Available metrics: revenue, expenses, profit, cash flow, margins
                Time periods: Can be specific dates, quarters (Q1-Q4), months, years, or relative (last N days/months)
                
                {format_instructions}
                
                Current date: {current_date}
                """
                ),
                HumanMessagePromptTemplate.from_template("{query}"),
            ]
        )

        # Format prompt
        messages = prompt.format_messages(
            format_instructions=parser.get_format_instructions(),
            current_date=datetime.now().strftime("%Y-%m-%d"),
            query=query,
        )

        # Get structured output from LLM
        result = await self.llm_client.query_with_structured_output(
            messages=messages,
            output_schema=ParsedQuery,
            model_override=settings.COMPLEX_QUERY_MODEL,  # Use better model for parsing
        )
        
        # Ensure result is a ParsedQuery instance
        if isinstance(result, dict):
            result = ParsedQuery(**result)

        # Post-process time periods
        if result.time_period and not result.time_start:
            result.time_start, result.time_end = self._parse_time_period(
                result.time_period
            )

        # Add context filters
        if context:
            if "company_id" in context and context["company_id"]:
                result.filters["company_id"] = context["company_id"]

        return result

    async def _generate_sql(
        self, parsed_query: ParsedQuery, context: Optional[Dict] = None
    ) -> SQLQuery:
        """Generate SQL query from parsed query structure"""

        # Build SQL based on intent
        if parsed_query.intent == QueryIntent.LOOKUP:
            sql = self._build_lookup_query(parsed_query)
        elif parsed_query.intent == QueryIntent.AGGREGATION:
            sql = self._build_aggregation_query(parsed_query)
        elif parsed_query.intent == QueryIntent.COMPARISON:
            sql = self._build_comparison_query(parsed_query)
        elif parsed_query.intent == QueryIntent.TREND:
            sql = self._build_trend_query(parsed_query)
        elif parsed_query.intent == QueryIntent.RANKING:
            sql = self._build_ranking_query(parsed_query)
        else:
            sql = self._build_calculation_query(parsed_query)

        # Create explanation
        explanation = self._explain_query(parsed_query)

        # Assess complexity
        complexity = self._assess_sql_complexity(sql)

        return SQLQuery(
            query=sql["query"],
            parameters=sql["parameters"],
            explanation=explanation,
            complexity=complexity,
        )

    def _build_lookup_query(self, parsed: ParsedQuery) -> Dict[str, Any]:
        """Build simple lookup query"""
        query_parts = []
        parameters = {}

        # SELECT clause
        select_metrics = []
        for metric in parsed.metrics:
            if metric in ["revenue", "income"]:
                select_metrics.append(
                    "SUM(CASE WHEN category = 'Revenue' OR category = 'Income' THEN value ELSE 0 END) as revenue"
                )
            elif metric in ["expenses", "costs"]:
                select_metrics.append(
                    "SUM(CASE WHEN category = 'Expenses' THEN value ELSE 0 END) as expenses"
                )
            elif metric == "profit":
                select_metrics.append(
                    """
                    (SUM(CASE WHEN category IN ('Revenue', 'Income') THEN value ELSE 0 END) - 
                     SUM(CASE WHEN category = 'Expenses' THEN value ELSE 0 END)) as profit
                """
                )

        query_parts.append(
            f"SELECT {', '.join(select_metrics) if select_metrics else 'SUM(value) as total'}"
        )

        # FROM clause
        query_parts.append(
            """
            FROM financial_line_items fli
            JOIN financial_records fr ON fli.financial_record_id = fr.id
            JOIN companies c ON fr.company_id = c.id
        """
        )

        # WHERE clause
        where_conditions = []

        if parsed.time_start:
            where_conditions.append("fr.period_start >= :start_date")
            parameters["start_date"] = parsed.time_start

        if parsed.time_end:
            where_conditions.append("fr.period_end <= :end_date")
            parameters["end_date"] = parsed.time_end

        if parsed.filters.get("company_id"):
            where_conditions.append("c.id = :company_id")
            parameters["company_id"] = parsed.filters["company_id"]

        if where_conditions:
            query_parts.append(f"WHERE {' AND '.join(where_conditions)}")

        return {"query": "\n".join(query_parts), "parameters": parameters}

    def _build_aggregation_query(self, parsed: ParsedQuery) -> Dict[str, Any]:
        """Build aggregation query with grouping"""
        base_query = self._build_lookup_query(parsed)

        # Add GROUP BY based on granularity
        if parsed.granularity:
            group_by = self._get_group_by_clause(parsed.granularity)
            base_query["query"] += f"\nGROUP BY {group_by}\nORDER BY {group_by}"

        return base_query

    def _build_comparison_query(self, parsed: ParsedQuery) -> Dict[str, Any]:
        """Build comparison query between periods"""
        # This would build a more complex query with period comparisons
        # For now, using simple approach
        return self._build_aggregation_query(parsed)

    def _build_trend_query(self, parsed: ParsedQuery) -> Dict[str, Any]:
        """Build trend analysis query"""
        query = self._build_aggregation_query(parsed)
        query["query"] = query["query"].replace(
            "SELECT", "SELECT strftime('%Y-%m', fr.period_start) as period,"
        )
        if "GROUP BY" not in query["query"]:
            query["query"] += "\nGROUP BY strftime('%Y-%m', fr.period_start)"
        query["query"] += "\nORDER BY period"
        return query

    def _build_ranking_query(self, parsed: ParsedQuery) -> Dict[str, Any]:
        """Build ranking query (top/bottom N)"""
        query = self._build_lookup_query(parsed)

        # Add grouping by line item
        query["query"] = query["query"].replace(
            "SELECT", "SELECT fli.name as item_name,"
        )
        query["query"] += "\nGROUP BY fli.name"
        query["query"] += f"\nORDER BY total DESC"

        if parsed.limit:
            query["query"] += f"\nLIMIT {parsed.limit}"

        return query

    def _build_calculation_query(self, parsed: ParsedQuery) -> Dict[str, Any]:
        """Build calculation query (margins, ratios, etc.)"""
        # This would handle more complex calculations
        return self._build_lookup_query(parsed)

    def _get_group_by_clause(self, granularity: TimeGranularity) -> str:
        """Get GROUP BY clause for time granularity"""
        if granularity == TimeGranularity.DAY:
            return "DATE(fr.period_start)"
        elif granularity == TimeGranularity.MONTH:
            return "strftime('%Y-%m', fr.period_start)"
        elif granularity == TimeGranularity.QUARTER:
            return "strftime('%Y-Q', fr.period_start) || ((CAST(strftime('%m', fr.period_start) AS INTEGER) - 1) / 3 + 1)"
        elif granularity == TimeGranularity.YEAR:
            return "strftime('%Y', fr.period_start)"
        else:
            return "fr.period_start"

    def _parse_time_period(
        self, period_text: str
    ) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Parse natural language time period into start/end dates"""
        period_lower = period_text.lower()

        for pattern, parser_func in self.time_patterns.items():
            match = re.search(pattern, period_lower)
            if match:
                return parser_func(match)

        return None, None

    def _parse_quarter(self, match) -> Tuple[datetime, datetime]:
        """Parse quarter (e.g., Q1 2024)"""
        quarter = int(match.group(1))
        year = int(match.group(2))

        quarter_starts = {
            1: datetime(year, 1, 1),
            2: datetime(year, 4, 1),
            3: datetime(year, 7, 1),
            4: datetime(year, 10, 1),
        }

        quarter_ends = {
            1: datetime(year, 3, 31, 23, 59, 59),
            2: datetime(year, 6, 30, 23, 59, 59),
            3: datetime(year, 9, 30, 23, 59, 59),
            4: datetime(year, 12, 31, 23, 59, 59),
        }

        return quarter_starts[quarter], quarter_ends[quarter]

    def _parse_month(self, match) -> Tuple[datetime, datetime]:
        """Parse month (e.g., January 2024)"""
        month_name = match.group(1).lower()
        year = int(match.group(2))

        months = {
            "january": 1,
            "february": 2,
            "march": 3,
            "april": 4,
            "may": 5,
            "june": 6,
            "july": 7,
            "august": 8,
            "september": 9,
            "october": 10,
            "november": 11,
            "december": 12,
        }

        month = months[month_name]
        start = datetime(year, month, 1)

        # Get last day of month
        if month == 12:
            end = datetime(year, 12, 31, 23, 59, 59)
        else:
            # Calculate last day of month
            next_month_first = datetime(year, month + 1, 1)
            end = next_month_first - timedelta(days=1)
            end = end.replace(hour=23, minute=59, second=59)

        return start, end

    def _parse_year(self, match) -> Tuple[datetime, datetime]:
        """Parse year (e.g., 2024)"""
        year = int(match.group(1))
        return datetime(year, 1, 1), datetime(year, 12, 31, 23, 59, 59)

    def _parse_relative(self, match) -> Tuple[datetime, datetime]:
        """Parse relative time (e.g., last 3 months)"""
        number = int(match.group(1))
        unit = match.group(2).rstrip("s")  # Remove plural 's'

        end = datetime.now()

        if unit == "day":
            start = end - timedelta(days=number)
        elif unit == "week":
            start = end - timedelta(weeks=number)
        elif unit == "month":
            start = end - timedelta(days=30 * number)  # Approximate
        elif unit == "quarter":
            start = end - timedelta(days=90 * number)  # Approximate
        elif unit == "year":
            start = end - timedelta(days=365 * number)  # Approximate
        else:
            start = end

        return start, end

    def _parse_ytd(self, match) -> Tuple[datetime, datetime]:
        """Parse year-to-date"""
        current_year = datetime.now().year
        return datetime(current_year, 1, 1), datetime.now()

    def _parse_current(self, match) -> Tuple[datetime, datetime]:
        """Parse current period (e.g., this month)"""
        period = match.group(2)
        now = datetime.now()

        if period == "month":
            start = datetime(now.year, now.month, 1)
            if now.month == 12:
                end = datetime(now.year + 1, 1, 1) - timedelta(days=1)
            else:
                end = datetime(now.year, now.month + 1, 1) - timedelta(days=1)
        elif period == "quarter":
            quarter = (now.month - 1) // 3 + 1

            # Create a mock match object for quarter parsing
            class MockMatch:
                def group(self, n):
                    if n == 1:
                        return str(quarter)
                    else:
                        return str(now.year)

            return self._parse_quarter(MockMatch())
        elif period == "year":
            start = datetime(now.year, 1, 1)
            end = datetime(now.year, 12, 31)
        else:
            start = end = now

        return start, end

    def _explain_query(self, parsed: ParsedQuery) -> str:
        """Generate human-readable explanation of the query"""
        parts = []

        # Intent
        intent_explanations = {
            QueryIntent.LOOKUP: "Looking up",
            QueryIntent.AGGREGATION: "Calculating total",
            QueryIntent.COMPARISON: "Comparing",
            QueryIntent.TREND: "Analyzing trends for",
            QueryIntent.RANKING: "Ranking",
            QueryIntent.CALCULATION: "Calculating",
        }
        parts.append(intent_explanations.get(parsed.intent, "Querying"))

        # Metrics
        if parsed.metrics:
            parts.append(f"{', '.join(parsed.metrics)}")

        # Time period
        if parsed.time_period:
            parts.append(f"for {parsed.time_period}")
        elif parsed.time_start and parsed.time_end:
            parts.append(
                f"from {parsed.time_start.strftime('%Y-%m-%d')} to {parsed.time_end.strftime('%Y-%m-%d')}"
            )

        # Filters
        if parsed.filters:
            filter_parts = [f"{k}={v}" for k, v in parsed.filters.items()]
            parts.append(f"filtered by {', '.join(filter_parts)}")

        return " ".join(parts)

    def _assess_sql_complexity(self, sql: Dict[str, Any]) -> QueryComplexity:
        """Assess complexity of generated SQL query"""
        query = sql["query"].lower()

        # Simple: Basic SELECT with minimal JOINs
        if query.count("join") <= 2 and "group by" not in query:
            return QueryComplexity.SIMPLE

        # Complex: Multiple JOINs, subqueries, or complex aggregations
        if "subquery" in query or query.count("join") > 3 or "union" in query:
            return QueryComplexity.COMPLEX

        # Medium: Everything else
        return QueryComplexity.MEDIUM
