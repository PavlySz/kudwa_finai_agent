"""
Simple test to debug evaluation framework
"""

import sys
import os
import asyncio

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation import TestSuite, AITestCase, TestCategory
from src.ai import QueryProcessor, ResponseFormatter, MultiModelClient


async def test_simple():
    """Run a simple test to see what's happening"""
    print("Testing evaluation framework...")
    
    # Initialize components
    llm_client = MultiModelClient()
    query_processor = QueryProcessor(llm_client)
    response_formatter = ResponseFormatter(llm_client)
    test_suite = TestSuite()
    
    # Get just one basic test
    basic_tests = test_suite.get_test_cases(TestCategory.BASIC)
    if basic_tests:
        test_case = basic_tests[0]
        print(f"\nTesting: {test_case.query}")
        print(f"Expected intent: {test_case.expected_intent}")
        print(f"Expected metrics: {test_case.expected_metrics}")
        
        try:
            result = await test_suite.run_test(test_case, query_processor, response_formatter)
            print(f"\nResult:")
            print(f"  Passed: {result.passed}")
            print(f"  Actual intent: {result.actual_intent}")
            print(f"  Actual metrics: {result.actual_metrics}")
            print(f"  Details: {result.details}")
            if result.error_message:
                print(f"  Error: {result.error_message}")
        except Exception as e:
            print(f"\nError during test: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_simple())
