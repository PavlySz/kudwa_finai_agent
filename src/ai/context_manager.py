"""
Context manager for maintaining conversation state and history.
Uses LangChain memory components for efficient context management.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import json

from langchain.memory import ConversationSummaryBufferMemory
from pydantic import BaseModel, Field

from src.ai.llm_client import MultiModelClient
from src.ai.query_processor import ParsedQuery, SQLQuery


class ConversationTurn(BaseModel):
    """Single turn in a conversation"""

    timestamp: datetime
    user_query: str
    parsed_query: Optional[ParsedQuery] = None
    sql_query: Optional[SQLQuery] = None
    response: str
    data_summary: Optional[Dict[str, Any]] = None
    entities: Dict[str, Any] = Field(default_factory=dict)


class ConversationContext(BaseModel):
    """Complete conversation context"""

    session_id: str
    started_at: datetime
    turns: List[ConversationTurn] = Field(default_factory=list)
    entities: Dict[str, Any] = Field(default_factory=dict)
    active_filters: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


class ContextManager:
    """
    Manages conversation context using LangChain memory.
    Maintains entity state and enables follow-up questions.
    """

    def __init__(self, llm_client: Optional[MultiModelClient] = None):
        self.llm_client = llm_client or MultiModelClient()
        self.sessions: Dict[str, ConversationContext] = {}
        self.memory_store: Dict[str, ConversationSummaryBufferMemory] = {}

    def get_or_create_session(self, session_id: str) -> ConversationContext:
        """Get existing session or create new one"""
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationContext(
                session_id=session_id, started_at=datetime.utcnow()
            )
            # Initialize LangChain memory for this session
            self.memory_store[session_id] = ConversationSummaryBufferMemory(
                llm=self.llm_client.get_model(),
                max_token_limit=2000,
                return_messages=True,
            )
        return self.sessions[session_id]

    async def add_turn(
        self,
        session_id: str,
        user_query: str,
        parsed_query: Optional[ParsedQuery],
        sql_query: Optional[SQLQuery],
        response: str,
        data_summary: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a conversation turn and update context"""
        context = self.get_or_create_session(session_id)

        # Extract entities from current turn
        entities = self._extract_entities(parsed_query, data_summary)

        # Create turn record
        turn = ConversationTurn(
            timestamp=datetime.utcnow(),
            user_query=user_query,
            parsed_query=parsed_query,
            sql_query=sql_query,
            response=response,
            data_summary=data_summary,
            entities=entities,
        )

        context.turns.append(turn)

        # Update session entities
        context.entities.update(entities)

        # Update LangChain memory
        if session_id not in self.memory_store:
            self.memory_store[session_id] = ConversationSummaryBufferMemory(
                llm=self.llm_client.get_model(),
                max_token_limit=2000,
                return_messages=True,
            )

        memory = self.memory_store[session_id]
        memory.chat_memory.add_user_message(user_query)
        memory.chat_memory.add_ai_message(response)

        # Update active filters if any
        if parsed_query and parsed_query.filters:
            context.active_filters.update(parsed_query.filters)

    def get_context_for_query(
        self, session_id: str, current_query: str
    ) -> Dict[str, Any]:
        """Get relevant context for processing a query"""
        context = self.get_or_create_session(session_id)

        # Get recent conversation history
        recent_turns = list(context.turns[-5:])  # Last 5 turns

        # Get memory summary from LangChain
        memory = self.memory_store.get(session_id)
        memory_messages = []
        if memory:
            messages = memory.chat_memory.messages
            # Get last 10 messages or summary
            if len(messages) > 10:
                memory_messages = memory.moving_summary_buffer
            else:
                memory_messages = [msg.content for msg in messages]

        # Identify referenced entities
        referenced_entities = self._identify_references(current_query, context)

        return {
            "session_id": session_id,
            "entities": context.entities,
            "active_filters": context.active_filters,
            "recent_queries": [turn.user_query for turn in recent_turns],
            "recent_results": [
                turn.data_summary for turn in recent_turns if turn.data_summary
            ],
            "memory_summary": memory_messages,
            "referenced_entities": referenced_entities,
            "turns_count": len(context.turns),
        }

    def _extract_entities(
        self,
        parsed_query: Optional[ParsedQuery],
        data_summary: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Extract entities from query and results"""
        entities = {}

        if parsed_query:
            # Extract time period
            if parsed_query.time_period:
                entities["last_time_period"] = {
                    "original": parsed_query.time_period,
                    "type": "time_period",
                }

            # Extract metrics
            if parsed_query.metrics:
                entities["last_metrics"] = parsed_query.metrics

            # Extract any mentioned companies or categories from filters
            if parsed_query.filters:
                for key, value in parsed_query.filters.items():
                    if key in ["company_id", "category", "name"]:
                        entities[f"filter_{key}"] = {"type": "filter", "value": value}

        if data_summary:
            # Extract data-related entities
            if "company_name" in data_summary:
                entities["current_company"] = data_summary["company_name"]
            if "categories" in data_summary:
                entities["available_categories"] = data_summary["categories"]

        return entities

    def _identify_references(
        self, query: str, context: ConversationContext
    ) -> Dict[str, Any]:
        """Identify pronouns and references in query"""
        references = {}
        query_lower = query.lower()

        # Check for pronouns referring to previous queries
        if any(word in query_lower for word in ["that", "those", "it", "them"]):
            if context.turns:
                last_turn = context.turns[-1]
                references["previous_subject"] = {
                    "query": last_turn.user_query,
                    "metrics": (
                        last_turn.parsed_query.metrics
                        if last_turn.parsed_query
                        else None
                    ),
                    "entities": last_turn.entities,
                }

        # Check for relative time references
        if any(word in query_lower for word in ["previous", "last", "before"]):
            if "last_time_period" in context.entities:
                references["time_reference"] = context.entities["last_time_period"]

        # Check for "same" references
        if "same" in query_lower:
            if context.turns:
                last_turn = context.turns[-1]
                if last_turn.parsed_query:
                    references["same_filters"] = last_turn.parsed_query.filters
                    references["same_period"] = last_turn.parsed_query.time_period

        return references

    def resolve_references(
        self, query: str, parsed_query: ParsedQuery, context_data: Dict[str, Any]
    ) -> ParsedQuery:
        """Resolve pronouns and references in parsed query"""
        references = context_data.get("referenced_entities", {})

        # Handle "that" or "those" references
        if not parsed_query.metrics and "previous_subject" in references:
            prev_metrics = references["previous_subject"].get("metrics")
            if prev_metrics:
                parsed_query.metrics = prev_metrics

        # Handle relative time references
        if not parsed_query.time_period and "time_reference" in references:
            # This would need more sophisticated logic to handle "previous quarter" etc.
            pass

        # Apply persistent filters
        active_filters = context_data.get("active_filters", {})
        if active_filters:
            if not parsed_query.filters:
                parsed_query.filters = {}
            # Don't override explicit filters
            for key, value in active_filters.items():
                if key not in parsed_query.filters:
                    parsed_query.filters[key] = value

        return parsed_query

    def clear_session(self, session_id: str) -> None:
        """Clear a conversation session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
        if session_id in self.memory_store:
            del self.memory_store[session_id]

    def get_session_summary(self, session_id: str) -> Optional[str]:
        """Get a summary of the conversation session"""
        context = self.sessions.get(session_id)
        if not context:
            return None

        summary_parts = [
            f"Session started: {context.started_at.strftime('%Y-%m-%d %H:%M')}",
            f"Total queries: {len(context.turns)}",
        ]

        if context.entities:
            summary_parts.append(
                f"Active entities: {', '.join(context.entities.keys())}"
            )

        if context.active_filters:
            summary_parts.append(
                f"Active filters: {json.dumps(context.active_filters)}"
            )

        # Get memory summary
        memory = self.memory_store.get(session_id)
        if memory and hasattr(memory, "moving_summary_buffer"):
            summary_parts.append(
                f"Conversation summary: {memory.moving_summary_buffer}"
            )

        return "\n".join(summary_parts)
