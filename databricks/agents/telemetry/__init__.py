from databricks.rag_eval.telemetry.custom_judge_model import (
    record_judge_model_usage_failure,
    record_judge_model_usage_success,
)

__all__ = [
    "record_judge_model_usage_success",
    "record_judge_model_usage_failure",
]
