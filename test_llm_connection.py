"""
Test script to verify LLM connections
"""

import asyncio
import sys
from langchain.schema import HumanMessage, SystemMessage

# Add src to path
sys.path.append(".")

from src.config.settings import settings, ModelName
from src.ai.llm_client import MultiModelClient, QueryComplexity


async def test_llm_connections():
    """Test all configured LLM models"""
    print("Testing LLM Connections...")
    print(f"Default Model: {settings.DEFAULT_MODEL}")
    print("-" * 50)

    try:
        # Initialize client
        client = MultiModelClient()
        print(f"Available models: {client.get_available_models()}")
        print("-" * 50)

        # Test messages
        messages = [
            SystemMessage(content="You are a helpful financial assistant."),
            HumanMessage(content="What is 2+2? Reply with just the number."),
        ]

        # Test each model
        for model_name in [
            ModelName.GPT5,
            ModelName.CLAUDE_SONNET,
            ModelName.CLAUDE_OPUS,
        ]:
            if model_name in client.models:
                print(f"\nTesting {model_name}...")
                try:
                    response = await client.query(
                        messages=messages, model_override=model_name, stream=False
                    )
                    print(f"✓ {model_name} response: {response}")
                except Exception as e:
                    print(f"✗ {model_name} error: {str(e)}")
            else:
                print(f"\n✗ {model_name} not available (no API key?)")

        # Test complexity routing
        print("\n" + "-" * 50)
        print("Testing complexity routing...")

        test_queries = [
            ("What is the total revenue?", QueryComplexity.SIMPLE),
            (
                "Compare Q1 and Q2 performance with trend analysis",
                QueryComplexity.COMPLEX,
            ),
            (
                "Verify the accuracy of last quarter's forecast",
                QueryComplexity.VERIFICATION,
            ),
        ]

        for query, expected_complexity in test_queries:
            messages = [HumanMessage(content=query)]
            complexity = client._assess_complexity(query)
            selected_model = client._select_model(complexity)
            print(f"\nQuery: '{query}'")
            print(f"  Complexity: {complexity} (expected: {expected_complexity})")
            print(f"  Selected model: {selected_model}")

        print("\n✓ All tests completed!")

    except Exception as e:
        print(f"\n✗ Error during testing: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_llm_connections())
