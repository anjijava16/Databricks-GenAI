"""
WARNING: This API is deprecated. Please use the new built-in LLM scorers API in `MLflow 3 <https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/predefined-judge-scorers>`_.
"""

from databricks.rag_eval.callable_builtin_judges import (
    chunk_relevance,
    context_sufficiency,
    correctness,
    groundedness,
    guideline_adherence,
    guidelines,
    relevance_to_query,
    safety,
)
from databricks.rag_eval.custom_prompt_judge import custom_prompt_judge

__all__ = [
    # Callable judges
    "chunk_relevance",
    "context_sufficiency",
    "correctness",
    "groundedness",
    "guideline_adherence",
    "guidelines",
    "relevance_to_query",
    "safety",
    "custom_prompt_judge",
]
