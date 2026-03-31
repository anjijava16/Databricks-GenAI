import dataclasses
from typing import Literal, Protocol

from databricks.rag_eval.evaluation import custom_metrics


class AssessmentConfig(Protocol):
    name: Literal[
        "safety",
        "groundedness",
        "relevance_to_query",
        "chunk_relevance",
        "guideline_adherence",
    ]
    sample_rate: float | None


AI_JUDGE_SAFETY: Literal["safety"] = "safety"
AI_JUDGE_GROUNDEDNESS: Literal["groundedness"] = "groundedness"
AI_JUDGE_RELEVANCE_TO_QUERY: Literal["relevance_to_query"] = "relevance_to_query"
AI_JUDGE_CHUNK_RELEVANCE: Literal["chunk_relevance"] = "chunk_relevance"
AI_JUDGE_GUIDELINE_ADHERENCE: Literal["guideline_adherence"] = "guideline_adherence"


_BUILT_IN_JUDGE_NAMES = [
    AI_JUDGE_SAFETY,
    AI_JUDGE_GROUNDEDNESS,
    AI_JUDGE_RELEVANCE_TO_QUERY,
    AI_JUDGE_CHUNK_RELEVANCE,
]

# Invalid characters not allowed in a delta table's column names
_INVALID_COLUMN_CHARS = " ,;{}()\n\t="


@dataclasses.dataclass
class CustomMetric(AssessmentConfig):
    """Configuration for a custom metric to be run on traces from a GenAI application.

    Raises:
        ValueError: When the provided function is not annotated with @metric.
    """

    metric_fn: custom_metrics.CustomMetric
    sample_rate: float | None = None

    def __post_init__(self):
        self.name = self.metric_fn.name
        if not isinstance(self.metric_fn, custom_metrics.CustomMetric):
            raise ValueError("Custom metric function must be annotated with @metric.")


@dataclasses.dataclass
class BuiltinJudge(AssessmentConfig):
    """Configuration for a builtin judge to be run on traces from a GenAI application.

    Raises:
        ValueError: When the judge name is invalid.
    """

    name: Literal["safety", "groundedness", "relevance_to_query", "chunk_relevance"]
    # Even though `AssessmentConfig` has this property already, we need to include it
    # in these child classes so that the dataclass __init__ parameter order is preserved.
    sample_rate: float | None = None

    def __post_init__(self):
        if self.sample_rate is not None and not (0.0 <= self.sample_rate <= 1.0):
            raise ValueError("Sample rate must be between 0.0 and 1.0 (inclusive).")

        if isinstance(self.name, str):
            if self.name not in _BUILT_IN_JUDGE_NAMES:
                raise ValueError(f"Invalid judge name '{self.name}'. Must be oneof {_BUILT_IN_JUDGE_NAMES}.")
            return


@dataclasses.dataclass
class GuidelinesJudge(AssessmentConfig):
    """Configuration for a guideline adherence judge to be run on traces from an GenAI application.

    Raises:
        ValueError: When there are duplicate keys in `guidelines` dict.
        ValueError: When there are duplicate values for a key in `guidelines` dict.
    """

    name: Literal["guideline_adherence"] = dataclasses.field(
        default=AI_JUDGE_GUIDELINE_ADHERENCE, init=False, repr=False
    )
    guidelines: dict[str, list[str]]
    # Even though `AssessmentConfig` has this property already, we need to include it
    # in these child classes so that the dataclass __init__ parameter order is preserved.
    sample_rate: float | None = None

    def __post_init__(self):
        if self.sample_rate is not None and not (0.0 <= self.sample_rate <= 1.0):
            raise ValueError("Sample rate must be between 0.0 and 1.0 (inclusive).")

        if self.guidelines is None:
            raise ValueError("guidelines cannot be None")

        # This warning message is necessary as the server can set an
        # empty guidelines judge as a default.
        if len(self.guidelines) == 0:
            print("WARNING: Your guidelines adherence judge has no guidelines.")
            return

        for key, values in self.guidelines.items():
            # check for empty guideline values within each key
            if not values:
                raise ValueError(f"Empty values found for key '{key}' in `guidelines` dict.")

            # check for duplicate guideline values within each key
            if len(values) != len(set(values)):
                raise ValueError(f"Duplicate values found for key '{key}' in `guidelines` dict.")

            # TODO(ML-52001): Remove this check once we move away from exploded assessments stored in delta
            if any(char in key for char in _INVALID_COLUMN_CHARS):
                raise ValueError(
                    f"Invalid character in key '{key}' in `guidelines` dict. Please ensure "
                    f"guideline names follow the format provided in "
                    f"https://learn.microsoft.com/en-us/azure/databricks/sql/language-manual/sql-ref-names"
                )


@dataclasses.dataclass
class AssessmentsSuiteConfig:
    """Configuration for a suite of assessments to be run on traces from a GenAI application.

    Raises:
        ValueError: When sample is not between 0.0 and 1.0 (inclusive).
        ValueError: When more than one guidelines judge is found.
        ValueError: When duplicate builtin judges are found.
    """

    sample: float | None = None
    paused: bool | None = None
    assessments: list[AssessmentConfig] | None = None

    def __post_init__(self):
        if self.sample is not None and not (0.0 <= self.sample <= 1.0):
            raise ValueError("Sample must be between 0.0 and 1.0 (inclusive).")

        if self.assessments is None:
            return

        # Enforce unique metrics
        seen_preset_judges = set()
        for assessment in self.assessments:
            if isinstance(assessment, BuiltinJudge):
                if str(assessment.name) in seen_preset_judges:
                    raise ValueError(f'Duplicate builtin judge found: "{assessment.name}"')
                seen_preset_judges.add(str(assessment.name))

        # Enforce one guidelines judge in assessments
        if len(list(filter(lambda a: isinstance(a, GuidelinesJudge), self.assessments))) > 1:
            raise ValueError("Only one guidelines judge is allowed in assessments.")

    @classmethod
    def from_dict(cls, data: dict):
        assessments: list[AssessmentConfig] | None = None
        if "assessments" in data:
            assessments = []
            for assessment in data["assessments"]:
                if assessment["name"] == AI_JUDGE_GUIDELINE_ADHERENCE:
                    assessments.append(
                        GuidelinesJudge(
                            guidelines=assessment["guidelines"],
                            sample_rate=assessment.get("sample_rate"),
                        )
                    )
                    continue
                assessments.append(
                    BuiltinJudge(
                        name=assessment["name"],
                        sample_rate=assessment.get("sample_rate"),
                    )
                )

        return cls(
            sample=data.get("sample"),
            paused=data.get("paused"),
            assessments=assessments,
        )

    def get_guidelines_judge(self) -> GuidelinesJudge | None:
        """Get the first GuidelinesJudge from the assessments list, or None if not found."""
        if self.assessments is None:
            return None
        for assessment in self.assessments:
            if isinstance(assessment, GuidelinesJudge):
                return assessment
        return None
