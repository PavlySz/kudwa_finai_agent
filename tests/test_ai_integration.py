"""
Integration tests for the complete AI system.
Tests the full flow from natural language query to financial insights.
"""

import sys
import os
import asyncio
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ai import (
    MultiModelClient,
    QueryProcessor,
    ResponseFormatter,
    ContextManager
)
from src.config.settings import settings


async def test_full_query_flow():
    """Test the complete query processing flow"""
    print("\n" + "="*60)
    print("FULL AI INTEGRATION TEST")
    print("="*60)
    
    # Initialize components
    llm_client = MultiModelClient()
    query_processor = QueryProcessor(llm_client)
    response_formatter = ResponseFormatter(llm_client)
    context_manager = ContextManager(llm_client)
    
    # Test queries
    test_queries = [
        "What was the total revenue in Q1 2024?",
        "Compare expenses between January and February 2024",
        "Show me the top 3 expense categories"
    ]
    
    session_id = "test-session"
    
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        
        try:
            # Get context
            context = context_manager.get_context_for_query(session_id, query)
            
            # Process query
            parsed_query, sql_query = await query_processor.process_query(query, context)
            
            print(f"✓ Intent: {parsed_query.intent.value}")
            print(f"✓ Metrics: {parsed_query.metrics}")
            print(f"✓ Complexity: {sql_query.complexity.value}")
            print(f"✓ SQL Generated: {sql_query.query[:100]}...")
            
            # Simulate data (in real app, this would come from database)
            mock_data = [
                {"revenue": 150000, "period": "Q1 2024"},
                {"category": "Payroll", "amount": 50000},
                {"category": "Office", "amount": 20000}
            ]
            
            # Format response
            narrative = await response_formatter.format_response(
                query,
                parsed_query,
                sql_query,
                mock_data[:1],  # Use appropriate data for each query
                context
            )
            
            print(f"✓ Response Summary: {narrative.summary[:150]}...")
            print(f"✓ Confidence: {narrative.confidence}")
            
            # Update context
            await context_manager.add_turn(
                session_id,
                query,
                parsed_query,
                sql_query,
                narrative.summary,
                {"row_count": len(mock_data)}
            )
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Test context continuity
    print("\n" + "-"*60)
    print("CONTEXT CONTINUITY TEST")
    print("-"*60)
    
    follow_up = "What about expenses?"
    context = context_manager.get_context_for_query(session_id, follow_up)
    
    print(f"Session has {len(context['recent_queries'])} previous queries")
    print(f"Entities tracked: {list(context['entities'].keys())}")
    
    # Test session summary
    summary = context_manager.get_session_summary(session_id)
    print(f"\nSession Summary:\n{summary}")
    
    print("\n✅ Full AI integration test completed!")


async def test_streaming_response():
    """Test streaming response capability"""
    print("\n" + "="*60)
    print("STREAMING RESPONSE TEST")
    print("="*60)
    
    llm_client = MultiModelClient()
    
    query = "Explain the financial performance for Q1"
    messages = [
        {"role": "system", "content": "You are a financial analyst."},
        {"role": "user", "content": query}
    ]
    
    print(f"Query: {query}")
    print("Streaming response:")
    
    # Convert dict messages to BaseMessage objects
    from langchain.schema import SystemMessage, HumanMessage
    langchain_messages = [
        SystemMessage(content=messages[0]["content"]),
        HumanMessage(content=messages[1]["content"])
    ]
    
    try:
        stream = await llm_client.query(langchain_messages, stream=True)
        
        async for chunk in stream:
            if hasattr(chunk, 'content'):
                print(chunk.content, end='', flush=True)
        
        print("\n\n✅ Streaming test completed!")
        
    except Exception as e:
        print(f"\n❌ Streaming error: {str(e)}")


async def test_error_handling():
    """Test error handling and fallback mechanisms"""
    print("\n" + "="*60)
    print("ERROR HANDLING TEST")
    print("="*60)
    
    query_processor = QueryProcessor()
    
    # Test invalid query
    invalid_queries = [
        "DELETE FROM financial_records",  # SQL injection attempt
        "",  # Empty query
        "What is the meaning of life?",  # Off-topic query
    ]
    
    for query in invalid_queries:
        print(f"\nTesting: '{query}'")
        try:
            parsed, sql = await query_processor.process_query(query)
            # Check if query has parameters (safe) or embedded values (potentially unsafe)
            if sql.parameters:
                print("✓ Query uses parameters (safe)")
            else:
                print("✓ Query generated (check manually for safety)")
        except ValueError as e:
            print(f"✓ Correctly rejected: {str(e)}")
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")
    
    print("\n✅ Error handling test completed!")


async def run_all_tests():
    """Run all integration tests"""
    print("\n" + "="*60)
    print("AI INTEGRATION TEST SUITE")
    print("="*60)
    print(f"Using models: {list(MultiModelClient().models.keys())}")
    print(f"Default model: {settings.DEFAULT_MODEL}")
    
    await test_full_query_flow()
    await test_streaming_response()
    await test_error_handling()
    
    print("\n" + "="*60)
    print("✅ ALL AI INTEGRATION TESTS COMPLETED!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(run_all_tests())
