"""
Multi-model LLM client using LangChain
Provides unified interface for GPT-5, Claude Sonnet, and Claude Opus
"""

from typing import Dict, List, Optional, Any, Literal
from enum import Enum
import logging

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from src.config.settings import settings, ModelName

logger = logging.getLogger(__name__)


class QueryComplexity(Enum):
    """Query complexity levels for model routing"""

    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    VERIFICATION = "verification"


class MultiModelClient:
    """
    Manages multiple LLM providers with unified interface
    All models are configurable via settings.py
    """

    def __init__(self):
        """Initialize LLM clients for all configured models"""
        self.models = self._initialize_models()
        self.default_model = settings.DEFAULT_MODEL
        self.last_model_used = None

    def _initialize_models(self) -> Dict[str, Any]:
        """Initialize all available models"""
        models = {}

        # Initialize GPT-5
        if settings.OPENAI_API_KEY:
            models[ModelName.GPT5] = ChatOpenAI(
                model=ModelName.GPT5,
                api_key=settings.OPENAI_API_KEY,
                temperature=0.1,  # Low temperature for financial accuracy
                streaming=True,
                callbacks=[StreamingStdOutCallbackHandler()] if settings.DEBUG else [],
            )
            logger.info(f"Initialized {ModelName.GPT5}")

        # Initialize Claude models
        if settings.ANTHROPIC_API_KEY:
            # Claude Sonnet
            models[ModelName.CLAUDE_SONNET] = ChatAnthropic(
                model=ModelName.CLAUDE_SONNET,
                anthropic_api_key=settings.ANTHROPIC_API_KEY,
                temperature=0.1,
                streaming=True,
                max_tokens=4096,
            )
            logger.info(f"Initialized {ModelName.CLAUDE_SONNET}")

            # Claude Opus
            models[ModelName.CLAUDE_OPUS] = ChatAnthropic(
                model=ModelName.CLAUDE_OPUS,
                anthropic_api_key=settings.ANTHROPIC_API_KEY,
                temperature=0.1,
                streaming=True,
                max_tokens=4096,
            )
            logger.info(f"Initialized {ModelName.CLAUDE_OPUS}")

        if not models:
            raise ValueError("No API keys configured. Please set API keys in keys.env")

        return models

    def _assess_complexity(
        self, query: str, context: Optional[Dict] = None
    ) -> QueryComplexity:
        """
        Assess query complexity to determine model routing

        Args:
            query: User's natural language query
            context: Optional context for complexity assessment

        Returns:
            QueryComplexity enum
        """
        # Simple heuristics for now, can be enhanced with ML classification
        query_lower = query.lower()

        # Check for verification/evaluation requests
        if any(
            word in query_lower for word in ["verify", "check", "validate", "evaluate"]
        ):
            return QueryComplexity.VERIFICATION

        # Check for complex analysis keywords
        complex_keywords = [
            "compare",
            "trend",
            "forecast",
            "analyze",
            "breakdown",
            "year-over-year",
            "quarter-over-quarter",
            "variance",
            "correlation",
            "prediction",
            "insight",
        ]
        if any(keyword in query_lower for keyword in complex_keywords):
            return QueryComplexity.COMPLEX

        # Check for simple lookup keywords
        simple_keywords = [
            "total",
            "sum",
            "count",
            "what is",
            "how much",
            "show me",
            "get",
            "find",
        ]
        if any(keyword in query_lower for keyword in simple_keywords):
            # Check if it's asking for multiple things
            if query.count("and") > 1 or query.count(",") > 2:
                return QueryComplexity.MEDIUM
            return QueryComplexity.SIMPLE

        # Default to medium
        return QueryComplexity.MEDIUM

    def _select_model(self, complexity: QueryComplexity) -> ModelName:
        """
        Select appropriate model based on complexity and configuration

        Args:
            complexity: Assessed query complexity

        Returns:
            Selected model name
        """
        # Map complexity to configured models
        model_mapping = {
            QueryComplexity.SIMPLE: settings.SIMPLE_QUERY_MODEL,
            QueryComplexity.MEDIUM: settings.DEFAULT_MODEL,
            QueryComplexity.COMPLEX: settings.COMPLEX_QUERY_MODEL,
            QueryComplexity.VERIFICATION: settings.VERIFICATION_MODEL,
        }

        selected_model = model_mapping.get(complexity, settings.DEFAULT_MODEL)

        # Fallback if model not available
        if selected_model not in self.models:
            logger.warning(f"Model {selected_model} not available, using default")
            return settings.DEFAULT_MODEL

        return selected_model

    async def query(
        self,
        messages: List[BaseMessage],
        model_override: Optional[ModelName] = None,
        complexity: Optional[QueryComplexity] = None,
        stream: bool = False,
        **kwargs,
    ) -> str:
        """
        Query the appropriate LLM model

        Args:
            messages: List of messages for the conversation
            model_override: Optional model override (ignores routing)
            complexity: Optional complexity override
            stream: Whether to stream the response
            **kwargs: Additional model-specific parameters

        Returns:
            Model response as string
        """
        # Determine which model to use
        if model_override:
            selected_model = model_override
        else:
            # Assess complexity if not provided
            if not complexity and messages:
                last_human_msg = next(
                    (
                        msg.content
                        for msg in reversed(messages)
                        if isinstance(msg, HumanMessage)
                    ),
                    "",
                )
                complexity = self._assess_complexity(last_human_msg)

            selected_model = self._select_model(complexity or QueryComplexity.MEDIUM)

        # Get the model
        model = self.models.get(selected_model)
        if not model:
            raise ValueError(f"Model {selected_model} not initialized")

        logger.info(f"Using model: {selected_model} for complexity: {complexity}")
        self.last_model_used = selected_model

        try:
            if stream:
                # Return async generator for streaming
                return model.astream(messages, **kwargs)
            else:
                # Return complete response
                response = await model.ainvoke(messages, **kwargs)
                return response.content

        except Exception as e:
            logger.error(f"Error with model {selected_model}: {str(e)}")

            # Try fallback model if available
            if (
                selected_model != settings.DEFAULT_MODEL
                and settings.DEFAULT_MODEL in self.models
            ):
                logger.info(f"Falling back to {settings.DEFAULT_MODEL}")
                fallback_model = self.models[settings.DEFAULT_MODEL]
                response = await fallback_model.ainvoke(messages, **kwargs)
                return response.content
            else:
                raise

    async def query_with_structured_output(
        self,
        messages: List[BaseMessage],
        output_schema: Any,
        model_override: Optional[ModelName] = None,
        **kwargs,
    ) -> Any:
        """
        Query model and parse response into structured output

        Args:
            messages: Conversation messages
            output_schema: Pydantic model or schema for parsing
            model_override: Optional model override
            **kwargs: Additional parameters

        Returns:
            Parsed structured output
        """
        # For structured output, prefer more capable models
        if not model_override:
            model_override = settings.COMPLEX_QUERY_MODEL

        model = self.models.get(model_override, self.models[settings.DEFAULT_MODEL])
        self.last_model_used = model_override or settings.DEFAULT_MODEL

        # Use LangChain's structured output functionality
        structured_model = model.with_structured_output(output_schema)

        try:
            result = await structured_model.ainvoke(messages, **kwargs)
            # Ensure result is an instance of the schema, not a dict
            if isinstance(result, dict):
                try:
                    # Try to instantiate the schema with the dict
                    return output_schema(**result)
                except:
                    # If that fails, return as is
                    pass
            return result
        except Exception as e:
            logger.error(f"Error parsing structured output: {str(e)}")
            raise

    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        return list(self.models.keys())

    def get_current_default(self) -> str:
        """Get current default model from settings"""
        return settings.DEFAULT_MODEL

    def get_model(self, model_name: Optional[str] = None):
        """Get a specific model instance"""
        if model_name is None:
            model_name = self.default_model
        return self.models.get(model_name, self.models[self.default_model])
