"""
LLM-as-Judge system for evaluating AI response quality.
Uses Claude Opus to evaluate GPT-5 responses for accuracy and quality.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum

from langchain.schema import SystemMessage, HumanMessage

from src.config.settings import settings, ModelName
from src.ai.llm_client import MultiModelClient


class EvaluationCriteria(str, Enum):
    """Criteria for evaluating responses"""

    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    CLARITY = "clarity"
    INSIGHTS = "insights"
    SAFETY = "safety"
    RELEVANCE = "relevance"


class CriterionScore(BaseModel):
    """Score for a single evaluation criterion"""

    criterion: EvaluationCriteria
    score: int = Field(ge=0, le=10, description="Score from 0-10")
    explanation: str = Field(description="Why this score was given")


class EvaluationResult(BaseModel):
    """Complete evaluation result for a query-response pair"""

    query: str
    response: str
    overall_score: float = Field(ge=0, le=10)
    scores: List[CriterionScore]
    strengths: List[str]
    weaknesses: List[str]
    suggestions: List[str]
    evaluated_at: datetime = Field(default_factory=datetime.utcnow)
    evaluator_model: str


class LLMJudge:
    """
    Uses an LLM (Claude Opus) to evaluate the quality of AI responses.
    Provides detailed feedback on multiple criteria.
    """

    def __init__(self, llm_client: Optional[MultiModelClient] = None):
        self.llm_client = llm_client or MultiModelClient()
        self.evaluator_model = (
            ModelName.CLAUDE_OPUS
        )  # Use the most capable model for evaluation

    async def evaluate_response(
        self,
        query: str,
        response: str,
        ground_truth: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> EvaluationResult:
        """
        Evaluate a query-response pair on multiple criteria.

        Args:
            query: The original user query
            response: The AI system's response
            ground_truth: Optional expected/correct answer
            context: Optional context about the query

        Returns:
            Detailed evaluation result with scores and feedback
        """
        # Build evaluation prompt
        evaluation_prompt = self._build_evaluation_prompt(
            query, response, ground_truth, context
        )

        # Get evaluation from judge model
        messages = [
            SystemMessage(content=self._get_judge_system_prompt()),
            HumanMessage(content=evaluation_prompt),
        ]

        # Use structured output for reliable parsing
        evaluation_response = await self.llm_client.query_with_structured_output(
            messages, EvaluationResponse, model_override=self.evaluator_model
        )

        # Convert to result format
        scores = []
        total_score = 0

        for criterion in EvaluationCriteria:
            score_data = getattr(evaluation_response, f"{criterion.value}_score", 5)
            explanation = getattr(
                evaluation_response,
                f"{criterion.value}_explanation",
                "No explanation provided",
            )

            scores.append(
                CriterionScore(
                    criterion=criterion, score=score_data, explanation=explanation
                )
            )
            total_score += score_data

        overall_score = total_score / len(EvaluationCriteria)

        return EvaluationResult(
            query=query,
            response=response,
            overall_score=round(overall_score, 2),
            scores=scores,
            strengths=evaluation_response.strengths,
            weaknesses=evaluation_response.weaknesses,
            suggestions=evaluation_response.suggestions,
            evaluator_model=str(self.evaluator_model),
        )

    def _get_judge_system_prompt(self) -> str:
        """System prompt for the judge model"""
        return """You are an expert AI response evaluator specializing in financial data systems.
        
Your task is to evaluate AI responses based on these criteria:

1. ACCURACY (0-10): Is the financial data and calculations correct?
2. COMPLETENESS (0-10): Does the response fully answer the user's question?
3. CLARITY (0-10): Is the response clear and easy to understand?
4. INSIGHTS (0-10): Does the response provide valuable insights beyond raw data?
5. SAFETY (0-10): Is the SQL/query safe and the response free from errors?
6. RELEVANCE (0-10): Does the response stay focused on what was asked?

For each criterion, provide:
- A score from 0-10 (10 being perfect)
- A brief explanation of why you gave that score

Also identify:
- Key strengths of the response
- Main weaknesses or areas for improvement
- Specific suggestions for making the response better

Be objective and constructive in your evaluation."""

    def _build_evaluation_prompt(
        self,
        query: str,
        response: str,
        ground_truth: Optional[str],
        context: Optional[Dict[str, Any]],
    ) -> str:
        """Build the evaluation prompt"""
        prompt = f"""Please evaluate this financial AI system's response:

USER QUERY: {query}

AI RESPONSE: {response}
"""

        if ground_truth:
            prompt += f"\n\nEXPECTED/CORRECT ANSWER: {ground_truth}"

        if context:
            prompt += f"\n\nADDITIONAL CONTEXT: {context}"

        prompt += """

Evaluate the response on all criteria and provide detailed feedback."""

        return prompt

    async def batch_evaluate(
        self,
        query_response_pairs: List[Dict[str, str]],
        ground_truths: Optional[List[str]] = None,
    ) -> List[EvaluationResult]:
        """Evaluate multiple query-response pairs"""
        results = []

        for i, pair in enumerate(query_response_pairs):
            ground_truth = (
                ground_truths[i] if ground_truths and i < len(ground_truths) else None
            )

            result = await self.evaluate_response(
                query=pair["query"],
                response=pair["response"],
                ground_truth=ground_truth,
            )
            results.append(result)

        return results

    def calculate_aggregate_scores(
        self, evaluation_results: List[EvaluationResult]
    ) -> Dict[str, Any]:
        """Calculate aggregate scores across multiple evaluations"""
        if not evaluation_results:
            return {"error": "No evaluation results provided"}

        # Calculate average scores by criterion
        criterion_scores = {criterion: [] for criterion in EvaluationCriteria}

        for result in evaluation_results:
            for score in result.scores:
                criterion_scores[score.criterion].append(score.score)

        aggregate = {
            "total_evaluations": len(evaluation_results),
            "overall_average": round(
                sum(r.overall_score for r in evaluation_results)
                / len(evaluation_results),
                2,
            ),
            "by_criterion": {},
        }

        for criterion, scores in criterion_scores.items():
            if scores:
                aggregate["by_criterion"][criterion] = {
                    "average": round(sum(scores) / len(scores), 2),
                    "min": min(scores),
                    "max": max(scores),
                    "count": len(scores),
                }

        # Common strengths and weaknesses
        all_strengths = []
        all_weaknesses = []

        for result in evaluation_results:
            all_strengths.extend(result.strengths)
            all_weaknesses.extend(result.weaknesses)

        # Count occurrences
        strength_counts = {}
        for strength in all_strengths:
            strength_counts[strength] = strength_counts.get(strength, 0) + 1

        weakness_counts = {}
        for weakness in all_weaknesses:
            weakness_counts[weakness] = weakness_counts.get(weakness, 0) + 1

        # Sort by frequency
        aggregate["common_strengths"] = sorted(
            strength_counts.items(), key=lambda x: x[1], reverse=True
        )[:5]

        aggregate["common_weaknesses"] = sorted(
            weakness_counts.items(), key=lambda x: x[1], reverse=True
        )[:5]

        return aggregate


# Response model for structured output parsing
class EvaluationResponse(BaseModel):
    """Structured response from the judge model"""

    accuracy_score: int = Field(ge=0, le=10)
    accuracy_explanation: str

    completeness_score: int = Field(ge=0, le=10)
    completeness_explanation: str

    clarity_score: int = Field(ge=0, le=10)
    clarity_explanation: str

    insights_score: int = Field(ge=0, le=10)
    insights_explanation: str

    safety_score: int = Field(ge=0, le=10)
    safety_explanation: str

    relevance_score: int = Field(ge=0, le=10)
    relevance_explanation: str

    strengths: List[str]
    weaknesses: List[str]
    suggestions: List[str]
