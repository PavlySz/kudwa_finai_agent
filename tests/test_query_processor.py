"""
Tests for the query processor functionality
"""

import asyncio
import sys
import os
import pytest
from datetime import datetime, timedelta
from typing import List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ai.query_processor import QueryProcessor, QueryIntent, ParsedQuery, SQLQuery
from src.ai.llm_client import MultiModelClient


class TestQueryProcessor:
    """Test suite for query processor"""

    @pytest.fixture
    async def processor(self):
        """Create query processor instance"""
        return QueryProcessor()

    @pytest.mark.asyncio
    async def test_simple_revenue_query(self, processor):
        """Test simple revenue lookup query"""
        query = "What is the total revenue?"

        parsed, sql = await processor.process_query(query)

        assert parsed.intent in [QueryIntent.LOOKUP, QueryIntent.AGGREGATION]
        assert "revenue" in parsed.metrics
        assert sql.query is not None
        assert "SUM(" in sql.query
        assert "Revenue" in sql.query or "Income" in sql.query

        print(f"\n✓ Simple revenue query processed")
        print(f"  Intent: {parsed.intent}")
        print(f"  Metrics: {parsed.metrics}")
        print(f"  SQL: {sql.query[:100]}...")

    @pytest.mark.asyncio
    async def test_time_period_query(self, processor):
        """Test query with specific time period"""
        queries = [
            "What was the revenue in Q1 2024?",
            "Show me expenses for January 2024",
            "Total profit in 2023",
        ]

        for query in queries:
            parsed, sql = await processor.process_query(query)

            assert parsed.time_period is not None or parsed.time_start is not None
            assert ":start_date" in sql.query or ":end_date" in sql.query
            assert sql.parameters  # Should have date parameters

            print(f"\n✓ Time period query: '{query}'")
            print(f"  Time period: {parsed.time_period}")
            print(f"  Start: {parsed.time_start}, End: {parsed.time_end}")

    @pytest.mark.asyncio
    async def test_comparison_query(self, processor):
        """Test comparison queries"""
        query = "Compare revenue between Q1 and Q2 2024"

        parsed, sql = await processor.process_query(query)

        assert parsed.intent == QueryIntent.COMPARISON
        assert parsed.metrics is not None
        assert len(parsed.metrics) > 0

        print(f"\n✓ Comparison query processed")
        print(f"  Intent: {parsed.intent}")
        print(f"  SQL explanation: {sql.explanation}")

    @pytest.mark.asyncio
    async def test_trend_query(self, processor):
        """Test trend analysis queries"""
        query = "Show me the revenue trend for the last 6 months"

        parsed, sql = await processor.process_query(query)

        assert parsed.intent == QueryIntent.TREND
        assert "revenue" in parsed.metrics
        assert "GROUP BY" in sql.query or "group by" in sql.query.lower()

        print(f"\n✓ Trend query processed")
        print(f"  Intent: {parsed.intent}")
        print(f"  Granularity: {parsed.granularity}")

    @pytest.mark.asyncio
    async def test_ranking_query(self, processor):
        """Test ranking/top N queries"""
        query = "What are the top 5 expense categories?"

        parsed, sql = await processor.process_query(query)

        assert parsed.intent == QueryIntent.RANKING
        assert parsed.limit == 5
        assert "ORDER BY" in sql.query
        assert "LIMIT" in sql.query

        print(f"\n✓ Ranking query processed")
        print(f"  Intent: {parsed.intent}")
        print(f"  Limit: {parsed.limit}")

    @pytest.mark.asyncio
    async def test_complex_query(self, processor):
        """Test complex multi-part query"""
        query = "Compare profit margins between Q1 and Q2 2024, broken down by month"

        parsed, sql = await processor.process_query(query)

        assert parsed.intent in [QueryIntent.COMPARISON, QueryIntent.CALCULATION]
        assert parsed.metrics is not None
        assert sql.complexity.value in ["medium", "complex"]

        print(f"\n✓ Complex query processed")
        print(f"  Intent: {parsed.intent}")
        print(f"  Complexity: {sql.complexity}")
        print(f"  Explanation: {sql.explanation}")

    @pytest.mark.asyncio
    async def test_with_context(self, processor):
        """Test query processing with context"""
        query = "Show me total expenses"
        context = {"company_id": 1}

        parsed, sql = await processor.process_query(query, context)

        assert parsed.filters.get("company_id") == 1
        assert ":company_id" in sql.query
        assert sql.parameters.get("company_id") == 1

        print(f"\n✓ Context-aware query processed")
        print(f"  Filters: {parsed.filters}")

    def test_time_parsing(self, processor):
        """Test various time period parsing"""
        test_cases = [
            ("Q1 2024", datetime(2024, 1, 1), datetime(2024, 3, 31)),
            ("January 2024", datetime(2024, 1, 1), datetime(2024, 1, 31)),
            ("2023", datetime(2023, 1, 1), datetime(2023, 12, 31)),
        ]

        for period_text, expected_start, expected_end in test_cases:
            start, end = processor._parse_time_period(period_text)

            if expected_start:
                assert start.date() == expected_start.date()
            if expected_end:
                assert end.date() == expected_end.date()

            print(
                f"✓ Parsed '{period_text}' → {start.date() if start else None} to {end.date() if end else None}"
            )

    @pytest.mark.asyncio
    async def test_sql_safety(self, processor):
        """Test SQL injection prevention"""
        # Try to inject SQL
        malicious_query = "Show revenue'; DROP TABLE companies; --"

        parsed, sql = await processor.process_query(malicious_query)

        # Should use parameters, not direct string interpolation
        assert (
            ":start_date" in sql.query
            or ":end_date" in sql.query
            or "parameters" in str(sql)
        )
        assert "DROP TABLE" not in sql.query

        print(f"\n✓ SQL injection prevented")
        print(f"  Safe query generated with parameters")


async def run_all_tests():
    """Run all query processor tests"""
    print("=" * 60)
    print("QUERY PROCESSOR TESTS")
    print("=" * 60)

    tests = TestQueryProcessor()
    processor = QueryProcessor()

    # Run each test
    test_methods = [
        tests.test_simple_revenue_query,
        tests.test_time_period_query,
        tests.test_comparison_query,
        tests.test_trend_query,
        tests.test_ranking_query,
        tests.test_complex_query,
        tests.test_with_context,
        tests.test_sql_safety,
    ]

    for test_method in test_methods:
        try:
            print(f"\n{test_method.__name__}:")
            await test_method(processor)
        except Exception as e:
            print(f"✗ {test_method.__name__} failed: {e}")
            import traceback

            traceback.print_exc()

    # Test time parsing separately (not async)
    print(f"\ntest_time_parsing:")
    tests.test_time_parsing(processor)

    print("\n" + "=" * 60)
    print("✓ QUERY PROCESSOR TESTS COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_all_tests())
