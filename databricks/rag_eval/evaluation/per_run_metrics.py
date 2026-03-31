"""Generate the metrics logged into MLflow."""

import collections
from dataclasses import dataclass
from typing import Callable, Dict, List, Union

from databricks.rag_eval.config import assessment_config, evaluation_config
from databricks.rag_eval.evaluation import entities
from databricks.rag_eval.utils import aggregation_utils, rating_utils

_PERCENTAGE_SUFFIX = "/percentage"
RunMetrics = Dict[str, float]


@dataclass
class MetricAggregateData:
    """Data class to store aggregate information for a metric."""

    count: int
    aggregations: RunMetrics


def generate_per_run_metrics(
    eval_results: List[entities.EvalResult],
    config: evaluation_config.GlobalEvaluationConfig,
) -> RunMetrics:
    """
    Generates per-run MLflow metrics.

    :param eval_results: List of EvalResult objects
    :param config: Global evaluation config containing custom metric configurations
    :return: Dictionary of aggregated MLflow metrics
    """
    # Create mapping of metric function names to their aggregation configs from custom metrics
    # Add "mean" as default aggregation for all metrics
    metric_aggregations = {}
    for metric in config.custom_metrics or []:
        # Note the name here is not the full metric name, but the name of the custom metric function
        metric_aggregations[metric.name] = ["mean"] if metric.aggregations is None else metric.aggregations

    # Extract all aggregation metrics
    aggregation_metrics = {}
    for metric_name, metric_data in compute_aggregate_metric_results(eval_results, metric_aggregations).items():
        for agg_name, agg_value in metric_data.aggregations.items():
            aggregation_metrics[f"{metric_name}/{agg_name}"] = agg_value

    result = {
        **aggregation_metrics,
        # Per-request answer assessments
        **{
            f"{assessment_name}{_PERCENTAGE_SUFFIX}": true_rate
            for assessment_name, true_rate in _compute_true_rate_per_request_assessment(
                eval_results, assessment_config.AssessmentType.ANSWER
            ).items()
        },
        # Per-request retrieval assessments
        **{
            f"{assessment_name}{_PERCENTAGE_SUFFIX}": true_rate
            for assessment_name, true_rate in _compute_true_rate_per_request_assessment(
                eval_results, assessment_config.AssessmentType.RETRIEVAL_LIST
            ).items()
        },
    }

    # Count error in judges
    for assessment_name, error_count in _count_error_in_judges(eval_results).items():
        result[f"judge/{assessment_name}/error_count"] = error_count

    return result


def compute_aggregate_metric_results(
    eval_results: List[entities.EvalResult],
    metric_aggregations: Dict[str, List[Union[str, Callable]]],
) -> Dict[str, MetricAggregateData]:
    """
    Compute aggregations for metrics with numeric, boolean, or pass-fail values.

    If the metric value is an Assessment object, the value of the Assessment is used.

    :param eval_results: List of EvalResult objects
    :param metric_aggregations: Dictionary mapping metric function names to their aggregation configurations
    :return: Dictionary mapping metric names to MetricAggregateData objects containing aggregations
    """
    metric_values: Dict[str, List[float]] = collections.defaultdict(list)
    metric_counts: Dict[str, int] = collections.defaultdict(int)

    # Collect values
    for eval_result in eval_results:
        for metric_result in eval_result.metric_results:
            metric_value = metric_result.metric_value.feedback.value
            metric_name = metric_result.metric_value.name

            if isinstance(metric_value, (int, float, bool)):
                float_value = float(metric_value)
                metric_values[metric_name].append(float_value)
                metric_counts[metric_name] += 1
            elif (
                isinstance(metric_value, str)
                and entities.CategoricalRating(metric_value) != entities.CategoricalRating.UNKNOWN
            ):
                float_value = float(metric_value == entities.CategoricalRating.YES)
                metric_values[metric_name].append(float_value)
                metric_counts[metric_name] += 1

    # Compute aggregates
    result = {}
    for metric_name in metric_values:
        if metric_counts[metric_name] > 0:
            # Get the function name from the returned metric name. Otherwise, fall back to metric name
            metric_function_name = metric_name.split("/")[1] if len(metric_name.split("/")) > 1 else metric_name
            # Get aggregations for this metric, defaulting to just ["mean"]
            aggregations = metric_aggregations.get(metric_function_name, ["mean"])
            result[metric_name] = MetricAggregateData(
                count=metric_counts[metric_name],
                aggregations=aggregation_utils.get_aggregate_results(metric_values[metric_name], aggregations),
            )

    return result


def _compute_true_rate_per_request_assessment(
    eval_results: List[entities.EvalResult],
    expected_assessment_type: assessment_config.AssessmentType,
) -> Dict[str, float]:
    """
    Compute the rate of `True` in per-request assessment results.

    rate of `True` = count of `True` / count of non-null values.

    :param eval_results: List of EvalResult objects
    :param expected_assessment_type: Type of per-request assessment to compute results for (e.g., answer, retrieval_list)
    :return: Dictionary of rate of `True` for each per-request assessment
    """
    true_counts = collections.defaultdict(int)
    non_null_counts = collections.defaultdict(int)
    for eval_result in eval_results:
        for assessment_result in eval_result.assessment_results:
            # TODO(ML-45046): remove assessment type lookup in harness, rely on service
            # Get the assessment type from the built-in metrics. If the metric is not found, use the provided assessment type.
            try:
                builtin_assessment_config = (
                    assessment_config.get_builtin_assessment_config_with_service_assessment_name(
                        assessment_result.assessment_name
                    )
                )
                assessment_type = builtin_assessment_config.assessment_type
            except ValueError:
                assessment_type = assessment_result.assessment_type

            if (
                isinstance(assessment_result, entities.PerRequestAssessmentResult)
                and assessment_type == expected_assessment_type
            ):
                true_counts[assessment_result.assessment_name] += (
                    assessment_result.rating.categorical_value == entities.CategoricalRating.YES
                )
                non_null_counts[assessment_result.assessment_name] += (
                    assessment_result.rating.categorical_value is not None
                )

    return {
        assessment_name: true_counts[assessment_name] / non_null_counts[assessment_name]
        for assessment_name in true_counts
        if non_null_counts[assessment_name] > 0
    }


def _count_error_in_judges(
    eval_results: List[entities.EvalResult],
) -> Dict[str, int]:
    """
    Count the number of errors in the assessment results.

    :param eval_results: List of EvalResult objects
    :return: Dictionary of count of errors for each assessment
    """
    error_counts = collections.defaultdict(int)
    for eval_result in eval_results:
        for assessment_result in eval_result.assessment_results:
            if isinstance(assessment_result, entities.PerRequestAssessmentResult):
                if _is_real_error_rating(assessment_result.rating):
                    error_counts[assessment_result.assessment_name] += 1
            elif isinstance(assessment_result, entities.PerChunkAssessmentResult):
                for positional_rating in assessment_result.positional_rating.values():
                    if _is_real_error_rating(positional_rating):
                        error_counts[assessment_result.assessment_name] += 1

    return error_counts


def _is_real_error_rating(rating: entities.Rating) -> bool:
    """Check if the rate is a real error. Missing input error is not considered as a real error."""
    return (
        rating.error_message is not None
        and not rating_utils.is_missing_input_error(rating.error_message)
        and not rating_utils.has_conflicting_input_error(rating.error_message)
    )
