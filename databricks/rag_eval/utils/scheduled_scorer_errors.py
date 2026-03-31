"""Custom exceptions for scheduled scorer deserialization errors."""

from typing import List


class UndeserializableScorerError(Exception):
    """Raised when a scorer cannot be deserialized."""

    def __init__(self, scorer_name: str, error: Exception):
        self.scorer_name = scorer_name
        self.error = error
        super().__init__(f"Failed to deserialize scorer '{scorer_name}': {str(error)}")


class AtLeastOneUndeserializableScorerError(Exception):
    """Raised when at least one scorer cannot be deserialized."""

    def __init__(self, errors: List[UndeserializableScorerError]):
        self.errors = errors
        error_messages = [f"- {e.scorer_name}" for e in errors]

        super().__init__(
            f"{len(errors)} of your scorers failed to deserialize:\n"
            + "\n".join(error_messages)
            + "\n\nCommon causes of deserialization errors:"
            + "\n- Invalid code environment"
            + "\n- Incompatible MLflow version"
            + "\n- Non self-contained code (packages used within scorer function body should be imported inline)"
            + "\n- Type hints requiring imports in scorer function signature"
            + "\n\nTo fix this issue, please delete the undeserializable scorer(s) using "
            + "mlflow.genai.scorers.delete_scorer(name=...) and recreate them with the correct configuration."
        )
