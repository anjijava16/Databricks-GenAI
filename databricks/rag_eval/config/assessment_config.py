"""All the internal configs."""

import dataclasses
import numbers
from dataclasses import field
from typing import Any, List, Optional

import mlflow.models

from databricks.rag_eval import schemas
from databricks.rag_eval.config import example_config
from databricks.rag_eval.utils import enum_utils, error_utils

METRIC_METADATA__ASSESSMENT_TYPE = "assessment_type"
METRIC_METADATA__SCORE_THRESHOLD = "score_threshold"


@dataclasses.dataclass(frozen=True)
class BinaryConversion:
    """
    Conversion for the result of an assessment to a binary result.
    """

    threshold: float
    """
    Threshold value for converting to the binary.
    If not None, it means the output of the metric can be converted to a binary result.
    """
    greater_is_true: bool = field(default=True)
    """
    Whether to convert to True when the metric value is greater than the threshold or vice versa.
    If True, the binary result is True when the metric value score is greater than or equal to the threshold.
    If False, the binary result is True when the metric value score is less than or equal to the threshold.
    """

    def convert(self, score: Any) -> Optional[bool]:
        """
        Convert the score to a binary result based on the threshold and greater_is_true.

        If the score is not a real number, return None.
        """
        if isinstance(score, numbers.Real):
            # noinspection PyTypeChecker
            return score >= self.threshold if self.greater_is_true else score <= self.threshold
        else:
            return None

    def convert_to_score(self, rating: example_config.ExampleRating) -> str:
        """
        Convert an example rating to a score based on the threshold and greater_is_true.
        """
        return "5" if self.greater_is_true == (rating == example_config.ExampleRating.YES) else "1"


class AssessmentType(enum_utils.StrEnum):
    """Type of the assessment."""

    RETRIEVAL = "RETRIEVAL"
    """Assessment for a retrieved chunk. This is used to assess the quality of retrieval over a single chunk."""
    RETRIEVAL_LIST = "RETRIEVAL_LIST"
    """Assessment for all retrievals. This is used to assess the quality of retrieval over the whole context."""
    ANSWER = "ANSWER"
    """Assessment for answer. This is used to assess the quality of answer."""


class AssessmentInputRequirements(enum_utils.StrEnum):
    ASSESSMENT_INPUT_REQUIREMENTS_UNSPECIFIED = "ASSESSMENT_INPUT_REQUIREMENTS_UNSPECIFIED"
    CHAT_REQUEST = "CHAT_REQUEST"
    CHAT_RESPONSE = "CHAT_RESPONSE"
    RETRIEVAL_CONTEXT = "RETRIEVAL_CONTEXT"
    GROUND_TRUTH_CHAT_RESPONSE = "GROUND_TRUTH_CHAT_RESPONSE"
    GROUND_TRUTH_RETRIEVAL_CONTEXT = "GROUND_TRUTH_RETRIEVAL_CONTEXT"
    GRADING_NOTES = "GRADING_NOTES"
    EXPECTED_FACTS = "EXPECTED_FACTS"
    GUIDELINES = "GUIDELINES"

    @classmethod
    def to_user_facing_column_name(cls, input_requirement: "AssessmentInputRequirements") -> str:
        match input_requirement:
            case cls.CHAT_REQUEST:
                return schemas.REQUEST_COL
            case cls.CHAT_RESPONSE:
                return schemas.RESPONSE_COL
            case cls.RETRIEVAL_CONTEXT:
                return schemas.RETRIEVED_CONTEXT_COL
            case cls.GROUND_TRUTH_CHAT_RESPONSE:
                return schemas.EXPECTED_RESPONSE_COL
            case cls.GROUND_TRUTH_RETRIEVAL_CONTEXT:
                return schemas.EXPECTED_RETRIEVED_CONTEXT_COL
            case cls.GRADING_NOTES:
                return schemas.GRADING_NOTES_COL
            case cls.EXPECTED_FACTS:
                return schemas.EXPECTED_FACTS_COL
            case cls.GUIDELINES:
                return schemas.GUIDELINES_COL
            case _:
                raise ValueError(f"Unrecognized input requirement: {input_requirement}")


@dataclasses.dataclass(frozen=True)
class AssessmentInputRequirementExpression:
    required: List[AssessmentInputRequirements] = field(default_factory=list)
    """Required columns for the assessment."""

    at_least_one_of: List[AssessmentInputRequirements] = field(default_factory=list)
    """At least one of the columns is required for the assessment."""

    at_most_one_of: List[AssessmentInputRequirements] = field(default_factory=list)
    """At most one of the columns should be provided for the assessment."""

    @classmethod
    def get_user_facing_requirement_names(cls, requirements: List[AssessmentInputRequirements]) -> List[str]:
        return [AssessmentInputRequirements.to_user_facing_column_name(requirement) for requirement in requirements]


@dataclasses.dataclass(frozen=True)
class AssessmentConfig:
    assessment_name: str

    assessment_type: AssessmentType

    flip_rating: bool = field(default=False)
    """Whether to flip the rating from the service."""

    # TODO(ML-44244): Call the /chat-assessments-definitions endpoints to get input requirements
    require_question: bool = field(default=False)
    """Whether the assessment requires input to be present in the dataset to eval."""

    require_answer: bool = field(default=False)
    """Whether the assessment requires output to be present in the dataset to eval."""

    require_retrieval_context: bool = field(default=False)
    """Whether the assessment requires retrieval context to be present in the dataset to eval."""

    require_retrieval_context_array: bool = field(default=False)
    """Whether the assessment requires retrieval context array to be present in the dataset to eval."""

    require_ground_truth_answer: bool = field(default=False)
    """Whether the assessment requires ground truth answer to be present in the dataset to eval."""

    require_ground_truth_answer_or_expected_facts: bool = field(default=False)
    """Whether the assessment requires ground truth answer or expected facts to be present in the dataset to eval."""

    require_guidelines: bool = field(default=False)
    """Whether the assessment requires guidelines to be present in the dataset to eval."""


@dataclasses.dataclass(frozen=True)
class BuiltinAssessmentConfig(AssessmentConfig):
    """
    Assessment represents a method to assess the quality of a RAG system.

    The method is defined by an MLflow EvaluationMetric object.
    """

    user_facing_assessment_name: Optional[str] = field(default=None)
    """If the service uses a different assessment name than the client, this is the user-facing name."""

    def __hash__(self):
        """
        Allow this object to be used as a key in a dictionary.
        """
        return hash(self.assessment_name)


@dataclasses.dataclass(frozen=True)
class EvaluationMetricAssessmentConfig(AssessmentConfig):
    """
    Represents a provided evaluation metric assessment configuration.

    This is used to represent an assessment that is provided by the user as an MLflow EvaluationMetric object.
    """

    binary_conversion: Optional[BinaryConversion] = field(default=None)
    """
    Configs how the result can be converted to binary.
    None if the result is not for converting to binary.
    """

    evaluation_metric: mlflow.models.EvaluationMetric = field(default=None)

    @classmethod
    def from_eval_metric(cls, evaluation_metric: mlflow.models.EvaluationMetric):
        """
        Create a EvaluationMetricAssessmentConfig object from an MLflow EvaluationMetric object.
        """
        try:
            assessment_type = AssessmentType(
                evaluation_metric.metric_metadata.get(METRIC_METADATA__ASSESSMENT_TYPE, "").upper()
            )
        except Exception:
            raise error_utils.ValidationError(
                f"Invalid assessment type in evaluation metric: {evaluation_metric.name}. Evaluation metric "
                f"must contain metric metadata with key 'assessment_type' and value 'RETRIEVAL', 'RETRIEVAL_LIST', or 'ANSWER'."
            )

        threshold = evaluation_metric.metric_metadata.get(METRIC_METADATA__SCORE_THRESHOLD, 3)

        return cls(
            assessment_name=evaluation_metric.name,
            assessment_type=AssessmentType(assessment_type),
            evaluation_metric=evaluation_metric,
            binary_conversion=BinaryConversion(
                threshold=threshold, greater_is_true=evaluation_metric.greater_is_better
            ),
        )

    def __hash__(self):
        """
        Allow this object to be used as a key in a dictionary.
        """
        return hash(self.assessment_name)


def create_builtin_assessment_configs(
    assessment_list: List[str],
) -> List[BuiltinAssessmentConfig]:
    """
    Parse a list of builtin assessments (and optional examples) into a list of BuiltinAssessmentConfigs
    """

    assessment_configs = []
    for assessment_name in assessment_list:
        builtin_assessment_conf = get_builtin_assessment_config_with_eval_assessment_name(assessment_name)

        assessment_configs.append(builtin_assessment_conf)

    return assessment_configs


def create_custom_eval_metric_assessment_configs(
    eval_metrics: Optional[List[mlflow.models.EvaluationMetric]],
) -> List[EvaluationMetricAssessmentConfig]:
    """
    Create AssessmentJudge objects from a list of custom evaluation metrics.
    """
    if eval_metrics is None:
        return []
    return [
        EvaluationMetricAssessmentConfig.from_eval_metric(metric)
        for metric in eval_metrics
        if isinstance(metric, mlflow.models.EvaluationMetric)
    ]


# ================ Builtin Assessments ================
GROUNDEDNESS = BuiltinAssessmentConfig(
    assessment_name="groundedness",
    assessment_type=AssessmentType.ANSWER,
    require_question=True,
    require_answer=True,
    require_retrieval_context=True,
)

CORRECTNESS = BuiltinAssessmentConfig(
    assessment_name="correctness",
    assessment_type=AssessmentType.ANSWER,
    require_question=True,
    require_answer=True,
    require_ground_truth_answer_or_expected_facts=True,
)

HARMFULNESS = BuiltinAssessmentConfig(
    assessment_name="harmfulness",
    user_facing_assessment_name="safety",
    assessment_type=AssessmentType.ANSWER,
    require_answer=True,
    flip_rating=True,
)

RELEVANCE_TO_QUERY = BuiltinAssessmentConfig(
    assessment_name="relevance_to_query",
    assessment_type=AssessmentType.ANSWER,
    require_question=True,
    require_answer=True,
)

CONTEXT_SUFFICIENCY = BuiltinAssessmentConfig(
    assessment_name="context_sufficiency",
    assessment_type=AssessmentType.RETRIEVAL_LIST,
    require_question=True,
    require_ground_truth_answer_or_expected_facts=True,
    require_retrieval_context=True,
)

CHUNK_RELEVANCE = BuiltinAssessmentConfig(
    assessment_name="chunk_relevance",
    assessment_type=AssessmentType.RETRIEVAL,
    require_question=True,
    require_retrieval_context_array=True,
)

GUIDELINE_ADHERENCE = BuiltinAssessmentConfig(
    assessment_name="guideline_adherence",
    assessment_type=AssessmentType.ANSWER,
    require_question=True,
    require_answer=True,
    require_guidelines=True,
)

GLOBAL_GUIDELINE_ADHERENCE = BuiltinAssessmentConfig(
    assessment_name="guideline_adherence",
    user_facing_assessment_name="global_guideline_adherence",
    assessment_type=AssessmentType.ANSWER,
    require_question=True,
    require_answer=True,
    require_guidelines=True,
)

GUIDELINES = BuiltinAssessmentConfig(
    assessment_name="guidelines",
    assessment_type=AssessmentType.ANSWER,
    require_question=True,
    require_answer=True,
    require_guidelines=True,
)


def _builtin_assessment_configs() -> List[BuiltinAssessmentConfig]:
    """Returns the list of built-in assessment configs for default evaluation"""
    return [
        HARMFULNESS,
        GROUNDEDNESS,
        CORRECTNESS,
        RELEVANCE_TO_QUERY,
        CHUNK_RELEVANCE,
        CONTEXT_SUFFICIENCY,
        GUIDELINE_ADHERENCE,
    ]


def _all_builtin_assessment_configs() -> List[BuiltinAssessmentConfig]:
    """Returns all available built-in assessment configs including specialized ones"""
    return _builtin_assessment_configs() + [
        GUIDELINES,
    ]


def builtin_assessment_names() -> List[str]:
    """Returns the list of built-in assessment names"""
    return [assessment_config.assessment_name for assessment_config in _builtin_assessment_configs()]


def builtin_answer_assessment_names() -> List[str]:
    """Returns the list of built-in answer assessment configs"""
    return [
        assessment_config.assessment_name
        for assessment_config in _builtin_assessment_configs()
        if assessment_config.assessment_type == AssessmentType.ANSWER
    ]


def builtin_retrieval_assessment_names() -> List[str]:
    """Returns the list of built-in retrieval assessment configs"""
    return [
        assessment_config.assessment_name
        for assessment_config in _builtin_assessment_configs()
        if assessment_config.assessment_type == AssessmentType.RETRIEVAL
    ]


def builtin_retrieval_list_assessment_names() -> List[str]:
    """Returns the list of built-in retrieval assessment configs"""
    return [
        assessment_config.assessment_name
        for assessment_config in _builtin_assessment_configs()
        if assessment_config.assessment_type == AssessmentType.RETRIEVAL_LIST
    ]


def get_builtin_assessment_config_with_service_assessment_name(
    name: str,
) -> BuiltinAssessmentConfig:
    """
    Returns the built-in assessment config with the given service assessment name
    :param name: The service assessment name of the assessment
    :returns: The built-in assessment config
    """
    for assessment_config in _all_builtin_assessment_configs():
        if assessment_config.assessment_name == name:
            return assessment_config

    all_available_names = [config.assessment_name for config in _all_builtin_assessment_configs()]
    raise ValueError(
        f"Assessment '{name}' not found in the builtin assessments. " f"Available assessments: {all_available_names}."
    )


def get_builtin_assessment_config_with_eval_assessment_name(
    name: str,
) -> BuiltinAssessmentConfig:
    """
    Returns the built-in assessment config with the given eval assessment name
    :param name: The eval assessment name of the assessment
    :returns: The built-in assessment config
    """
    available_assessment_names = []
    for assessment_config in _all_builtin_assessment_configs():
        eval_assessment_name = (
            assessment_config.user_facing_assessment_name
            if assessment_config.user_facing_assessment_name is not None
            else assessment_config.assessment_name
        )
        if eval_assessment_name == name:
            return assessment_config

        available_assessment_names.append(eval_assessment_name)

    raise ValueError(
        f"Assessment '{name}' not found in the builtin assessments. "
        f"Available assessments: {available_assessment_names}."
    )


def needs_flip(service_assessment_name: str) -> bool:
    """Returns whether the rating needs to be flipped for a given assessment."""
    return get_builtin_assessment_config_with_service_assessment_name(service_assessment_name).flip_rating
