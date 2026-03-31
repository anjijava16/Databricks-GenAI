"""Entry point to the evaluation harness"""

from __future__ import annotations

import dataclasses
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partialmethod
from typing import Dict, List, Optional, Tuple, Union

import mlflow
import mlflow.tracing.constant as tracing_constant
from mlflow import entities as mlflow_entities
from tqdm.auto import tqdm

from databricks.rag_eval import context, env_vars, session
from databricks.rag_eval.clients.managedrag import managed_rag_client
from databricks.rag_eval.config import assessment_config, evaluation_config
from databricks.rag_eval.evaluation import (
    assessments,
    datasets,
    entities,
    metrics,
    models,
    per_run_metrics,
    traces,
)
from databricks.rag_eval.evaluation import custom_metrics as agent_custom_metrics
from databricks.rag_eval.utils import input_output_utils, rate_limit, trace_utils

_logger = logging.getLogger(__name__)
_FAIL_TO_GET_TRACE_WARNING_MSG = re.compile(r"Failed to get trace from the tracking store")
_FAIL_TO_SET_TRACE_TAG_WARNING_MSG = re.compile(r".*is immutable and cannot be set on a trace.")

EvalResults = List[entities.EvalResult]


def _get_current_time() -> float:
    """
    Get the current time in seconds since the epoch.
    This method is extracted to make it easier to mock in tests.

    Returns:
        float: Current time in seconds since the epoch.
    """
    return time.perf_counter()


def run(
    *,
    eval_dataset: Union[datasets.EvaluationDataframe, List[entities.EvalItem]],
    config: evaluation_config.GlobalEvaluationConfig,
    experiment_id: Optional[str] = None,
    run_id: Optional[str] = None,
    model=None,
) -> EvalResults:
    """
    Run the logic of the eval harness.

    :param eval_dataset: The evaluation dataset
    :param config: The evaluation config
    :param experiment_id: The MLflow experiment ID to log the results to (used for logging traces)
    :param run_id: The MLflow run ID to log the results to (used for logging traces)
    :param model: Optional model to use for generating responses and traces
    :return: EvalResults
    """

    eval_items = eval_dataset.eval_items if isinstance(eval_dataset, datasets.EvaluationDataframe) else eval_dataset

    # Disable tqdm progress bar by default so that the progress bars inside MLflow eval_fn do not show
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

    client = context.get_context().build_managed_rag_client()
    rate_limiter = _build_rate_limiter_for_assessment()

    # Emit usage events prior to all logic
    _emit_custom_assessments_usage_event_if_present(client, config.global_assessment_configs)

    ctx = context.get_context()
    # Ensure there's always a valid experiment ID. Note this internal method will fall back to the
    # default experiment ID if there is no current experiment. In Databricks, it's the
    # notebook-based experiment and in OSS it is `experiment_id=0`.
    experiment_id = ctx.get_mlflow_experiment_id() if experiment_id is None else experiment_id
    # Try to get the active run from the context
    run_id = ctx.get_mlflow_run_id() if run_id is None else run_id

    eval_results = []
    with ThreadPoolExecutor(max_workers=env_vars.RAG_EVAL_MAX_WORKERS.get()) as executor:
        futures = [
            executor.submit(
                _run_single,
                eval_item=eval_item,
                config=config.get_eval_item_eval_config(eval_item.question_id),
                model=model,
                client=client,
                rate_limiter=rate_limiter,
                current_session=session.current_session(),
                experiment_id=experiment_id,
                run_id=run_id,
            )
            for eval_item in eval_items
        ]

        futures_as_completed = as_completed(futures)
        # Add a progress bar to show the progress of the assessments
        futures_as_completed = tqdm(
            futures_as_completed,
            total=len(futures),
            disable=False,
            desc="Evaluating",
            smoothing=0,  # 0 means using average speed for remaining time estimates
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [Elapsed: {elapsed}, Remaining: {remaining}] {postfix}",
        )

        total_agent_time = 0.0
        total_metric_time = 0.0
        for future in futures_as_completed:
            eval_result, eval_times = future.result()
            eval_results.append(eval_result)
            if eval_times.agent_invocation_time is not None:
                total_agent_time += eval_times.agent_invocation_time
            if eval_times.metric_computation_time is not None:
                total_metric_time += eval_times.metric_computation_time

            if total_agent_time > 0 or total_metric_time > 0:
                agent_invocation_percentage = (total_agent_time / (total_agent_time + total_metric_time)) * 100
                metric_computation_percentage = 100 - agent_invocation_percentage
                futures_as_completed.set_postfix(
                    {
                        "Time breakdown": f"({agent_invocation_percentage:.2f}% predict_fn, {metric_computation_percentage:.2f}% scorers)"
                    }
                )

    # Batch link traces to run at the end of evaluation to avoid rate limits
    _batch_link_traces_to_run(run_id, eval_results)

    # Compute aggregate metrics if there are custom metrics configured
    aggregate_metrics = {}
    if config.custom_metrics:
        try:
            aggregate_metrics = per_run_metrics.compute_aggregate_metric_results(
                # We pass in an empty dict which will default to computing mean for all metrics
                # This is because our telemetry only logs the average right now
                eval_results,
                {},
            )
        except Exception as e:
            _logger.error(
                "Failed to compute aggregate metrics. Skipping emitting custom metric usage event.",
                exc_info=e,
            )

    _emit_custom_metric_usage_event_if_present(client, config.custom_metrics, metric_stats=aggregate_metrics)

    return eval_results


@dataclasses.dataclass
class EvalTimes:
    """Dataclass to track timing information for evaluation runs.

    Attributes:
        agent_invocation_time: Time taken for agent invocation in seconds
        metric_computation_time: Time taken for metric computation in seconds
    """

    agent_invocation_time: Optional[float] = None
    metric_computation_time: Optional[float] = None


def _run_single(
    eval_item: entities.EvalItem,
    config: evaluation_config.ItemEvaluationConfig,
    client: managed_rag_client.ManagedRagClient,
    rate_limiter: rate_limit.RateLimiter,
    experiment_id: Optional[str],
    run_id: Optional[str],
    model: Optional[mlflow.pyfunc.PyFuncModel] = None,
    current_session: Optional[session.Session] = None,
) -> Tuple[entities.EvalResult, EvalTimes]:
    """
    Run the logic of the eval harness for a single eval item.

    :param eval_item: The eval item to evaluate
    :param config: The evaluation config
    :param model: Optional model to use for generating responses and traces
    :param mlflow_run_id: MLflow run ID to use for this evaluation
    :return: EvalResult, EvalTimes) where EvalTimes is a dataclass with agent_invocation_time and metric_computation_time
    """
    session.set_session(current_session)
    # Set the MLflow run ID in the context for this thread
    if run_id:
        # Manually set the mlflow_run_id for this context to be the same as was set in the parent thread.
        # This is required because MLflow runs are thread-local.
        ctx = context.get_context()
        ctx.set_mlflow_run_id(run_id)

    trace_error_message = None
    model_invocation_time = None
    if model:
        start_time = _get_current_time()
        eval_item = _populate_model_result_to_eval_item(
            eval_item=eval_item,
            model_result=models.invoke_model(model, eval_item),
        )
        model_invocation_time = _get_current_time() - start_time
    elif eval_item.trace is not None:
        # Catch any issues with malformed traces
        try:
            # If logging to MLflow is disabled, we don't need to clone the trace
            if env_vars.AGENT_EVAL_LOG_TRACES_TO_MLFLOW_ENABLED.get() and _should_clone_trace(
                eval_item.trace, experiment_id
            ):
                prepared_trace = trace_utils.clone_trace_to_reupload(eval_item.trace)
                cloned_trace = trace_utils.inject_experiment_run_id_to_trace(prepared_trace, experiment_id, run_id)
                eval_item.trace = cloned_trace
            eval_item = _populate_eval_item_with_trace(eval_item)
        except Exception as e:
            trace_error_message = str(e)
    else:
        minimal_trace = _create_minimal_trace(eval_item)
        eval_item.trace = minimal_trace
        eval_item = _populate_eval_item_with_trace(eval_item)

    if eval_item.model_error_message is not None:
        try:
            client.emit_client_error_usage_event(eval_item.model_error_message)
        except Exception:
            # Telemetry logging failures are non-fatal.
            pass

    # Skip the evaluation if invoking the model failed or there's a malformed trace
    eval_item_error_message = eval_item.model_error_message or trace_error_message
    if eval_item_error_message:
        eval_result = entities.EvalResult(
            eval_item=eval_item,
            eval_error=eval_item_error_message,
        )
        metric_computation_time = 0.0
    else:
        start_time = _get_current_time()
        assessment_results = assessments.generate_llm_assessments(
            client=client,
            rate_limiter=rate_limiter,
            eval_item=eval_item,
            config=config,
        )
        metric_results = metrics.compute_eval_metrics(
            eval_item=eval_item,
            assessment_results=assessment_results,
            metrics=metrics.BUILT_IN_METRICS + config.custom_metrics,
        )
        metric_computation_time = _get_current_time() - start_time
        eval_result = entities.EvalResult(
            eval_item=eval_item,
            assessment_results=assessment_results,
            metric_results=metric_results,
        )

    try:
        logged_trace = log_traces_and_assessments(
            experiment_id=experiment_id,
            run_id=run_id,
            eval_item=eval_item,
            assessments=eval_result.assessments,
        )
        eval_result.eval_item.trace = logged_trace
    except Exception as e:
        # Failures in logging to MLflow should not fail the entire harness run
        _logger.warning(f"Failed to log trace and assessments to MLflow: {e}")

    return eval_result, EvalTimes(
        agent_invocation_time=model_invocation_time or 0.0,
        metric_computation_time=metric_computation_time,
    )


def _should_clone_trace(trace: Optional[mlflow_entities.Trace], experiment_id: str) -> bool:
    """
    Determine if we should clone the trace.

    :param trace: The trace to check
    :param experiment_id: The experiment ID to check against
    """
    if trace is None:
        return False

    # MLflow experiment location can now be None if the trace is from a delta table.
    if trace.info.trace_location.mlflow_experiment is None:
        return False

    # Check if the trace is from the same experiment. If it isn't, we need to clone the trace
    is_trace_from_same_exp = trace.info.trace_location.mlflow_experiment.experiment_id == experiment_id
    return not is_trace_from_same_exp


def _should_link_trace_to_run(trace: Optional[mlflow_entities.Trace], run_id: Optional[str]) -> bool:
    """
    Determine if we should link the trace to the run.

    :param trace: The trace to check
    :param run_id: The run ID to check against
    """
    if trace is None or run_id is None:
        return False

    # Do a best effort attempt to retrieve the run ID from the trace metadata
    trace_run_id = trace.info.trace_metadata.get(tracing_constant.TraceMetadataKey.SOURCE_RUN)
    # If the trace is from the same experiment but a different run, we need to
    # link the trace to the run.
    return trace_run_id is None or trace_run_id != run_id


def _populate_model_result_to_eval_item(
    eval_item: entities.EvalItem, model_result: models.ModelResult
) -> entities.EvalItem:
    """
    Populate the model result to the eval item in place.

    :param eval_item: The eval item to populate the model result
    :param model_result: The model result to populate
    :return: The populated eval item
    """
    eval_item.answer = model_result.response
    try:
        eval_item.raw_response = model_result.raw_model_output
        # Ensure the raw response is json-serializable.
        input_output_utils.to_dict(eval_item.raw_response)
    except Exception as e:
        raise ValueError(
            f"The response from the model must be JSON serializable: {type(model_result.raw_model_output)}. "
        ) from e
    eval_item.retrieval_context = model_result.retrieval_context
    eval_item.tool_calls = model_result.tool_calls
    eval_item.trace = model_result.trace
    eval_item.model_error_message = model_result.error_message
    return eval_item


def _create_minimal_trace(eval_item: entities.EvalItem) -> mlflow_entities.Trace:
    if eval_item.retrieval_context is not None:
        trace = trace_utils.create_minimal_trace(
            input_output_utils.to_dict(eval_item.raw_request),
            input_output_utils.to_dict(eval_item.raw_response),
            (eval_item.retrieval_context.to_mlflow_documents() if eval_item.retrieval_context is not None else None),
        )
        eval_item.retrieval_context.span_id = traces._get_top_level_retrieval_spans(trace)[0].span_id
        return trace
    else:
        return trace_utils.create_minimal_trace(
            input_output_utils.to_dict(eval_item.raw_request),
            input_output_utils.to_dict(eval_item.raw_response),
        )


def _populate_eval_item_with_trace(eval_item: entities.EvalItem) -> entities.EvalItem:
    """
    Populate the eval item in place by extracting additional information from the trace.

    Keep the existing values in the eval item if they already exist.
    """
    # Extract tool calls from the trace, or response if trace is not available.
    eval_item.tool_calls = traces.extract_tool_calls(
        response=input_output_utils.to_dict(eval_item.raw_response),
        trace=eval_item.trace,
    )

    # Skip if the trace is None
    if eval_item.trace is None:
        return eval_item

    eval_item.raw_response = input_output_utils.to_dict(traces.extract_model_output_from_trace(eval_item.trace))

    eval_item.answer = (
        input_output_utils.response_to_string(traces.extract_model_output_from_trace(eval_item.trace))
        if eval_item.answer is None
        else eval_item.answer
    )

    eval_item.retrieval_context = (
        traces.extract_retrieval_context_from_trace(eval_item.trace)
        if eval_item.retrieval_context is None
        else eval_item.retrieval_context
    )

    return eval_item


def _emit_custom_assessments_usage_event_if_present(
    client: managed_rag_client.ManagedRagClient,
    assessment_configs: List[assessment_config.AssessmentConfig],
):
    # TODO: change this to use the new usage tracking API
    evaluation_metric_configs = [
        assessment_conf
        for assessment_conf in assessment_configs
        if isinstance(assessment_conf, assessment_config.EvaluationMetricAssessmentConfig)
    ]

    if evaluation_metric_configs:
        try:
            batch_size = session.current_session().session_batch_size
            client.emit_chat_assessment_usage_event(evaluation_metric_configs, batch_size)
        except Exception:
            # Telemetry logging failures are non-fatal.
            # Don't want to indicate to users that we're emitting data
            # TODO [ML-43811]: handle this case better since it means we have a loss of billing data
            pass


def _emit_custom_metric_usage_event_if_present(
    client: managed_rag_client.ManagedRagClient,
    custom_metrics: List[agent_custom_metrics.CustomMetric],
    metric_stats: Dict[str, per_run_metrics.MetricAggregateData],
):
    # Filter out built-in metrics before emitting custom metric usage events
    # Built-in metrics either:
    # 1. Start with "agent/" prefix
    # 2. Have the _is_builtin_scorer attribute set to True
    actual_custom_metrics = [
        metric
        for metric in custom_metrics
        if not metric.name.startswith("agent/") and not getattr(metric, "_is_builtin_scorer", False)
    ]

    if actual_custom_metrics:
        try:
            batch_size = session.current_session().session_batch_size
            # Create a set of actual custom metric names for faster lookup
            actual_custom_metric_names = {metric.name for metric in actual_custom_metrics}

            client.emit_custom_metric_usage_event(
                custom_metrics=actual_custom_metrics,
                eval_count=batch_size,
                metric_stats={k: v for k, v in metric_stats.items() if k in actual_custom_metric_names},
            )
        except Exception:
            # Telemetry logging failures are non-fatal.
            # Don't want to indicate to users that we're emitting data
            # TODO [ML-43811]: handle this case better since it means we have a loss of billing data
            pass


def _build_rate_limiter_for_assessment() -> rate_limit.RateLimiter:
    """Build a rate limiter for the assessment."""
    # Return a no-op rate limiter if the rate limiter for assessment is not enabled
    if not env_vars.RAG_EVAL_ENABLE_RATE_LIMIT_FOR_ASSESSMENT.get():
        return rate_limit.RateLimiter.no_op()

    # For now, rate limiter config is from environment variables
    rate_limit_config = rate_limit.RateLimitConfig(
        quota=env_vars.RAG_EVAL_RATE_LIMIT_QUOTA.get(),
        time_window_in_seconds=env_vars.RAG_EVAL_RATE_LIMIT_TIME_WINDOW_IN_SECONDS.get(),
    )
    return rate_limit.RateLimiter.build(
        quota=rate_limit_config.quota,
        time_window_in_seconds=rate_limit_config.time_window_in_seconds,
    )


def _batch_link_traces_to_run(run_id: Optional[str], eval_results: EvalResults, max_batch_size: int = 100) -> None:
    """
    Batch link traces to a run to avoid rate limits.

    :param run_id: The MLflow run ID to link traces to
    :param eval_results: List of evaluation results containing traces
    :param max_batch_size: Maximum number of traces to link per batch call
    """
    if not run_id or not env_vars.AGENT_EVAL_LOG_TRACES_TO_MLFLOW_ENABLED.get():
        return

    trace_ids_to_link = []
    for eval_result in eval_results:
        trace = eval_result.eval_item.trace
        if trace and _should_link_trace_to_run(trace, run_id):
            trace_ids_to_link.append(trace.info.trace_id)

    if not trace_ids_to_link:
        return

    # Batch the trace IDs to avoid overwhelming the MLflow backend
    mlflow_client = context.get_context().build_mlflow_client()
    for i in range(0, len(trace_ids_to_link), max_batch_size):
        batch = trace_ids_to_link[i : i + max_batch_size]
        try:
            mlflow_client.link_traces_to_run(
                run_id=run_id,
                trace_ids=batch,
            )
        except Exception as e:
            _logger.warning(f"Failed to link batch of traces to run: {e}")


def log_traces_and_assessments(
    experiment_id: Optional[str],
    run_id: Optional[str],
    eval_item: entities.EvalItem,
    assessments: List[mlflow_entities.Assessment],
) -> mlflow_entities.Trace:
    """
    Log the trace and assessments to MLflow. We do this to ensure that MLFlow has a trace for every
    eval row, storing the computed assessments/metrics.

    A trace may have been generated and logged during model invocation, in which case we don't
    need to create a trace. However, if a trace was passed in as part of the eval row, we need to
    make a copy, because we don't know if the trace was used in a previous eval invocation. Without
    a copy, we could end up with multiple evaluation runs adding assessments to the same trace.

    Additionally, this function will set trace tags from the eval_item.tags using the MLflow client APIs.
    """
    trace = eval_item.trace

    if not experiment_id:
        _logger.warning("Failed to log trace and assessments to MLflow because experiment ID is not set")
        return trace

    if not env_vars.AGENT_EVAL_LOG_TRACES_TO_MLFLOW_ENABLED.get():
        return trace

    with (
        mlflow.utils.logging_utils.suppress_logs(mlflow.tracing.fluent.__name__, _FAIL_TO_GET_TRACE_WARNING_MSG),
        mlflow.utils.logging_utils.suppress_logs(mlflow.tracing.client.__name__, _FAIL_TO_SET_TRACE_TAG_WARNING_MSG),
    ):
        mlflow_client = mlflow.tracking.MlflowClient()
        # Ensure that every trace is logged in MLflow, regardless of where it came from.
        # Specifically, if the trace is present in MLflow, do nothing. Otherwise, log the trace.
        if trace.info.trace_id is None or mlflow.get_trace(trace.info.trace_id) is None:
            if trace.info.trace_location.mlflow_experiment is not None:
                trace.info.trace_location.mlflow_experiment.experiment_id = experiment_id
            else:
                trace.info.trace_location.mlflow_experiment = mlflow_entities.MlflowExperimentLocation(
                    experiment_id=experiment_id
                )

            if run_id is not None:
                trace.info.trace_metadata[tracing_constant.TraceMetadataKey.SOURCE_RUN] = run_id

            try:
                stored_trace_id = mlflow_client._log_trace(trace)
                trace.info.trace_id = stored_trace_id
            except Exception as e:
                _logger.warning(f"Failed to log the trace: {e}")
                return trace

        existing_expectation_names = {
            assessment.name
            for assessment in trace.info.assessments
            if isinstance(assessment, mlflow_entities.Expectation)
        }
        # Create the assessments
        for assessment in assessments:
            # Skip logging duplicate expectations based on expectation name
            if isinstance(assessment, mlflow_entities.Expectation) and assessment.name in existing_expectation_names:
                continue

            # Ensure that if we created a new trace, that the updated trace_id is reflected in
            # the assessments.
            assessment.trace_id = trace.info.trace_id

            if run_id is not None:
                assessment.metadata = (
                    {
                        **assessment.metadata,
                        tracing_constant.AssessmentMetadataKey.SOURCE_RUN_ID: run_id,
                    }
                    if assessment.metadata is not None
                    else {tracing_constant.AssessmentMetadataKey.SOURCE_RUN_ID: run_id}
                )
            _log_assessment_to_mlflow(assessment)

        # Set trace tags from eval_item.tags using MLflow client APIs
        if eval_item.tags and trace.info.trace_id:
            for tag_key, tag_value in eval_item.tags.items():
                try:
                    mlflow_client.set_trace_tag(trace.info.trace_id, tag_key, tag_value)
                except Exception as e:
                    _logger.warning(f"Failed to set trace tag {tag_key}={tag_value}: {e}")

        # Get the trace to fetch newly created assessments.
        return mlflow.get_trace(trace.info.trace_id)


def _log_assessment_to_mlflow(
    assessment: mlflow_entities.Assessment,
) -> Optional[mlflow_entities.Assessment]:
    """
    Creates the given assessment in MLflow.
    """
    # Note that the `log_expectation` and `log_feedback` APIs expect the ID without the "0x" prefix.
    # However, the `encode_trace_id` utility adds the "0x" prefix so we add this check.
    span_id = assessment.span_id.removeprefix("0x")
    try:
        if assessment.expectation is not None:
            return mlflow.log_expectation(
                trace_id=assessment.trace_id,
                name=assessment.name,
                source=assessment.source,
                value=assessment.expectation.value,
                metadata=assessment.metadata,
                span_id=span_id,
            )
        else:
            if assessment.error is not None:
                error = assessment.error
                value = None
            else:
                error = None
                value = assessment.feedback.value

            return mlflow.log_feedback(
                trace_id=assessment.trace_id,
                name=assessment.name,
                source=assessment.source,
                error=error,
                value=value,
                rationale=assessment.rationale,
                metadata=assessment.metadata,
                span_id=span_id,
            )
    except Exception as e:
        _logger.warning(f"Failed to log the assessment: {e}")
        return
