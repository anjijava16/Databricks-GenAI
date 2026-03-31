import dataclasses
import logging
from datetime import datetime
from typing import List, Optional, Union

import mlflow
from mlflow.genai import Scorer

from databricks.rag_eval import context
from databricks.rag_eval.monitoring import assessments, entities
from databricks.rag_eval.monitoring import external as external_monitoring
from databricks.rag_eval.monitoring import utils as monitoring_utils
from databricks.rag_eval.monitoring.utils import _get_experiment_id
from databricks.rag_eval.utils import uc_utils

_logger = logging.getLogger(__name__)


def _warn_monitoring_config_deprecation():
    print(
        "WARNING: The `monitoring_config` argument is deprecated and will be removed in a future version. "
        "Please use the `assessments_config` argument instead.\n"
    )


def _get_managed_evals_client():
    return context.get_context().build_managed_evals_client()


def _parse_monitoring_config(
    monitoring_config: dict | entities.MonitoringConfig,
) -> entities.MonitoringConfig:
    assert monitoring_config is not None, "monitoring_config is required"
    monitoring_config = entities.MonitoringConfig.from_dict(monitoring_config)

    # Validate sampling.
    assert isinstance(monitoring_config.sample, (int, float)), "monitoring_config.sample must be a number"
    assert 0 <= monitoring_config.sample <= 1, "monitoring_config.sample must be between 0 and 1"

    # Validate paused.
    assert monitoring_config.paused is None or isinstance(
        monitoring_config.paused, bool
    ), "monitoring_config.paused must be a boolean"

    # Validate metrics.
    assert monitoring_config.metrics is None or (
        isinstance(monitoring_config.metrics, list)
    ), "monitoring_config.metrics must be a list of strings"

    # Validate guidelines
    if monitoring_config.global_guidelines is not None:
        error_msg = "monitoring_config.global_guidelines must be a dictionary of {<guideline_name>: [<guidelines>]}"
        assert isinstance(monitoring_config.global_guidelines, dict), error_msg
        assert all(isinstance(g, list) for g in monitoring_config.global_guidelines.values()), error_msg
        assert all(isinstance(e, str) for g in monitoring_config.global_guidelines.values() for e in g), error_msg
        assert all(isinstance(k, str) for k in monitoring_config.global_guidelines.keys()), error_msg

    return monitoring_config


@context.eval_context
def create_monitor(
    endpoint_name: str,
    *,
    assessments_config: dict | assessments.AssessmentsSuiteConfig | None = None,
    experiment_id: str | None = None,
    monitoring_config: dict | entities.MonitoringConfig | None = None,
) -> entities.Monitor:
    """
    Create a monitor for a Databricks serving endpoint.

    Args:
        endpoint_name: The name of the serving endpoint.
        assessments_config: The configuration for the suite of assessments to be run on traces.
        experiment_id: The experiment ID to log the monitoring results. Defaults to the currently active MLflow experiment.
        monitoring_config: Deprecated. The monitoring configuration.
    Returns:
        The monitor for the serving endpoint.
    """
    if experiment_id is None:
        experiment_id = mlflow.tracking.fluent._get_experiment_id()
    if (monitoring_config is None) == (assessments_config is None):
        raise ValueError("Exactly one of `monitoring_config` or `assessments_config` must be specified.")

    if monitoring_config:
        _warn_monitoring_config_deprecation()
        monitoring_config = _parse_monitoring_config(monitoring_config)

        # don't let global_guidelines be set without guideline_adherence metric
        if monitoring_config.global_guidelines and assessments.AI_JUDGE_GUIDELINE_ADHERENCE not in (
            monitoring_config.metrics or []
        ):
            raise ValueError("Global guidelines can only be set if 'guideline_adherence' is included in `metrics`.")

        assessments_config = monitoring_config.to_assessments_suite_config()

    if isinstance(assessments_config, dict):
        assessments_config = assessments.AssessmentsSuiteConfig.from_dict(assessments_config)

    monitor = _create_monitor_internal(endpoint_name, assessments_config, experiment_id)
    print(f'Created monitor for experiment "{monitor.experiment_id}".')
    print(f"\nView traces: {monitor.monitoring_page_url}")

    if monitor.assessments_config.assessments:
        print("\nAssessments:")
        for assessment in monitor.assessments_config.assessments:
            if isinstance(assessment, assessments.BuiltinJudge):
                print(f"• {assessment.name}")
            elif isinstance(assessment, assessments.GuidelinesJudge):
                print(f"• {assessments.AI_JUDGE_GUIDELINE_ADHERENCE}")
    else:
        print(
            "\nNo assessments specified. To override the assessments, include `BuiltinJudge` or `GuidelinesJudge` instances in the `assessments` field in `assessments_config`."
        )

    return monitor


@context.eval_context
def _create_monitor_internal(
    endpoint_name: str,
    assessments_config: assessments.AssessmentsSuiteConfig,
    experiment_id: str,
) -> entities.Monitor:
    """
    Internal implementation of create_monitor. This function is called by both `create_monitor` and `agents.deploy`
    for MLflow 2.x models.
    """
    return _get_managed_evals_client().create_monitor(
        endpoint_name=endpoint_name,
        assessments_config=assessments_config,
        experiment_id=experiment_id,
    )


def _do_create_external_monitor(
    *,
    catalog_name: str,
    schema_name: str,
    assessments_config: assessments.AssessmentsSuiteConfig | dict,
    experiment_id: str | None = None,
    experiment_name: str | None = None,
) -> entities.ExternalMonitor:
    """Internal implementation of create_external_monitor."""
    if not catalog_name or not schema_name:
        raise ValueError("Both catalog_name and schema_name must be provided and non-empty.")

    # fresh assessments suite config validation and defaults
    if assessments_config is None:
        raise ValueError("assessments_config is required")
    if isinstance(assessments_config, dict):
        assessments_config = assessments.AssessmentsSuiteConfig.from_dict(assessments_config)
    if assessments_config.sample is None:
        assessments_config.sample = 1.0
    experiment = monitoring_utils.get_databricks_mlflow_experiment(
        experiment_id=experiment_id,
        experiment_name=experiment_name,
    )
    ckpt_table_entity_name = monitoring_utils.create_checkpoint_table_entity_name(
        experiment_name=experiment.name,
    )
    ckpt_table_uc_entity = uc_utils.UnityCatalogEntity.from_fullname(
        f"{catalog_name}.{schema_name}.{ckpt_table_entity_name}"
    )
    endpoint_name = external_monitoring.build_endpoint_name(ckpt_table_uc_entity.entity)

    return _get_managed_evals_client().create_monitor(
        endpoint_name=endpoint_name,
        assessments_config=assessments_config,
        experiment_id=experiment.experiment_id,
        monitoring_table=ckpt_table_uc_entity.fullname,
    )


@context.eval_context
def create_external_monitor(
    *,
    catalog_name: str,
    schema_name: str,
    assessments_config: assessments.AssessmentsSuiteConfig | dict,
    experiment_id: str | None = None,
    experiment_name: str | None = None,
) -> entities.ExternalMonitor:
    """Create a monitor for a GenAI application served outside Databricks.

    Args:
        catalog_name (str): The name of the catalog in UC to create the trace archive table in.
        schema_name (str): The name of the schema in UC to create the trace archive table in.
        assessments_config (AssessmentsSuiteConfig | dict): The configuration for the suite of
            assessments to be run on traces from the GenAI application.
        experiment_id (str | None, optional): ID of Mlflow experiment
            that the monitor should be associated with. Defaults to the
            currently active experiment.
        experiment_name (str | None, optional): The name of the Mlflow experiment that the monitor
            should be associated with. Defaults to the currently active experiment.

    Returns:
        ExternalMonitor: The created monitor.
    """
    monitor = _do_create_external_monitor(
        catalog_name=catalog_name,
        schema_name=schema_name,
        assessments_config=assessments_config,
        experiment_id=experiment_id,
        experiment_name=experiment_name,
    )

    print(f'Created monitor for experiment "{monitor.experiment_id}".')
    print(f"\nView traces: {monitor.monitoring_page_url}")

    if monitor.assessments_config.assessments:
        print("\nBuilt-in AI judges:")
        for assessment in monitor.assessments_config.assessments:
            if isinstance(assessment, assessments.BuiltinJudge):
                metric = str(assessment.name)
                print(f"• {metric}")
            elif isinstance(assessment, assessments.GuidelinesJudge):
                print(f"• {assessments.AI_JUDGE_GUIDELINE_ADHERENCE}")
    else:
        print(
            "\nNo built-in AI judges specified for monitor. To add built-in AI judges, include `BuiltinJudge` instances in the `assessments` field in `assessments_config`."
        )

    print(
        f"\nTo configure an agent to log traces to this experiment, call `mlflow.set_experiment(experiment_id={monitor.experiment_id})`, "
        "or configure the `MLFLOW_EXPERIMENT_ID` environment variable."
    )
    return monitor


@context.eval_context
def get_monitor(
    *,
    endpoint_name: str,
) -> entities.Monitor:
    """
    Retrieves a monitor for a Databricks serving endpoint.

    Args:
        endpoint_name (str, optional): The name of the agent's serving endpoint.

    Returns:
        Monitor | ExternalMonitor metadata. For external monitors, this will include the status of the ingestion endpoint.
    """
    monitor = _get_monitor_internal(
        endpoint_name=endpoint_name,
    )
    print("Monitor URL: ", monitor.monitoring_page_url)
    return monitor


@context.eval_context
def _get_monitor_internal(
    *,
    endpoint_name: str | None = None,
) -> entities.Monitor:
    """
    Internal implementation of get_monitor.
    """
    return _get_managed_evals_client().get_monitor(
        endpoint_name=endpoint_name,
    )


class NoMonitorFoundError(ValueError):
    """Exception raised when no monitor is found for the given parameters."""

    def __init__(self, message: str):
        super().__init__(message)


def _get_external_monitor_opt(
    *,
    experiment_id: str,
) -> Optional[entities.ExternalMonitor]:
    try:
        return get_external_monitor(experiment_id=experiment_id)
    except NoMonitorFoundError:
        pass


@context.eval_context
def get_external_monitor(
    *,
    experiment_id: str | None = None,
    experiment_name: str | None = None,
) -> entities.ExternalMonitor:
    """Gets the monitor for a GenAI application served outside Databricks.

    Args:
        experiment_id (str | None, optional): ID of the Mlflow experiment that
            the monitor is associated with. Defaults to None.
        experiment_name (str | None, optional): Name of the Mlflow experiment that
            the monitor is associated with. Defaults to None.

    Raises:
        ValueError: When neither experiment_id nor experiment_name is provided.
        ValueError: When no monitor is found for the given experiment_id or experiment_name.

    Returns:
        entities.ExternalMonitor: The retrieved external monitor.
    """
    if experiment_id is None and experiment_name is None:
        raise ValueError("Please provide either an experiment_id or experiment_name.")

    experiment = monitoring_utils.get_databricks_mlflow_experiment(
        experiment_id=experiment_id,
        experiment_name=experiment_name,
    )
    monitors: list[entities.ExternalMonitor] = _get_managed_evals_client().list_monitors(
        experiment_id=experiment.experiment_id,
    )
    if not (monitors and getattr(monitors[0], "_checkpoint_table", None)):
        if experiment_name:
            raise NoMonitorFoundError(f"No monitor found for experiment_name '{experiment_name}'.")
        else:
            raise NoMonitorFoundError(f"No monitor found for experiment_id '{experiment_id}'.")

    return _get_managed_evals_client().get_monitor(
        # There should be only one monitor per experiment
        monitoring_table=monitors[0]._checkpoint_table,
    )


@context.eval_context
def update_monitor(
    *,
    endpoint_name: str,
    assessments_config: dict | assessments.AssessmentsSuiteConfig | None = None,
    monitoring_config: dict | entities.MonitoringConfig | None = None,
) -> entities.Monitor:
    """
    Partially update a monitor for a serving endpoint.

    Args:
        endpoint_name (str, optional): The name of the agent's serving endpoint.
            Only supported for agents served on Databricks.
        assessments_config: The updated configuration for the suite of assessments to be run on traces.
            Partial updates of arrays is not supported, so assessments specified here will override your
            monitor's assessments. If unspecified, non-nested fields like `sample` will not be updated.
        monitoring_config: Deprecated. The configuration change, using upsert semantics.

    Returns:
        Monitor: The updated monitor for the serving endpoint.
    """
    if (monitoring_config is None) == (assessments_config is None):
        raise ValueError("Exactly one of `monitoring_config` or `assessments_config` must be specified.")

    if monitoring_config:
        _warn_monitoring_config_deprecation()
        if isinstance(monitoring_config, dict):
            monitoring_config = entities.MonitoringConfig.from_dict(monitoring_config)
        # don't let global_guidelines be set without guideline_adherence metric
        if monitoring_config.global_guidelines and assessments.AI_JUDGE_GUIDELINE_ADHERENCE not in (
            monitoring_config.metrics or []
        ):
            raise ValueError("Global guidelines can only be set if 'guideline_adherence' is included in `metrics`.")
        assessments_config = monitoring_config.to_assessments_suite_config()

    if isinstance(assessments_config, dict):
        assessments_config = assessments.AssessmentsSuiteConfig.from_dict(assessments_config)

    return _get_managed_evals_client().update_monitor(
        endpoint_name=endpoint_name,
        assessments_config=assessments_config,
    )


@context.eval_context
def update_external_monitor(
    *,
    experiment_id: str | None = None,
    experiment_name: str | None = None,
    assessments_config: assessments.AssessmentsSuiteConfig | dict,
) -> entities.ExternalMonitor:
    """Updates the monitor for an GenAI application served outside Databricks.

    Args:
        assessments_config (assessments.AssessmentsSuiteConfig): The updated configuration for
            the suite of assessments to be run on traces from the AI system. Partial updates
            of arrays is not supported, so assessments specified here will override your monitor's
            assessments. If unspecified, non-nested fields like `sample` will not be updated.
        experiment_id (str | None, optional): ID of the Mlflow experiment that the monitor
            is associated with. Defaults to None.
        experiment_name (str | None, optional): Name of the Mlflow experiment that the monitor
            is associated with. Defaults to None.

    Raises:
        ValueError: When assessments_config is not provided.

    Returns:
        entities.ExternalMonitor: The updated external monitor.
    """

    if assessments_config is None:
        raise ValueError("assessments_config is required")
    if isinstance(assessments_config, dict):
        assessments_config = assessments.AssessmentsSuiteConfig.from_dict(assessments_config)

    monitor = get_external_monitor(
        experiment_id=experiment_id,
        experiment_name=experiment_name,
    )
    return _get_managed_evals_client().update_monitor(
        endpoint_name=monitor._legacy_ingestion_endpoint_name,
        assessments_config=assessments_config,
    )


@context.eval_context
def delete_monitor(
    *,
    endpoint_name: str | None = None,
) -> None:
    """
    Deletes a monitor for a Databricks serving endpoint.

    Args:
        endpoint_name (str, optional): The name of the agent's serving endpoint.
    """
    # for external monitors, find the endpoint name using the monitoring table
    return _get_managed_evals_client().delete_monitor(endpoint_name=endpoint_name)


@context.eval_context
def delete_external_monitor(
    *,
    experiment_id: str | None = None,
    experiment_name: str | None = None,
) -> None:
    """Deletes the monitor for a GenAI application served outside Databricks.

    Args:
        experiment_id (str | None, optional): ID of the Mlflow experiment that the monitor
            is associated with. Defaults to None.
        experiment_name (str | None, optional): Name of the Mlflow experiment that the monitor
            is associated with. Defaults to None.
    """

    monitor = get_external_monitor(
        experiment_id=experiment_id,
        experiment_name=experiment_name,
    )
    _get_managed_evals_client().delete_monitor(
        endpoint_name=monitor._legacy_ingestion_endpoint_name,
    )


@dataclasses.dataclass
class BackfillScorerConfig:
    """
    Configuration for a scorer to be used in backfill operations.

    Args:
        scorer: The MLflow Scorer instance (required).
        sample_rate: Optional sample rate override. If not provided, uses scorer.sample_rate.
    """

    scorer: Scorer
    sample_rate: Optional[float] = None


@context.eval_context
def backfill_scorers(
    *,
    experiment_id: Optional[str] = None,
    scorers: Union[List[BackfillScorerConfig], List[str]],
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> dict[str, str]:
    """
    Backfill scorers for a given experiment and time range.

    Args:
        experiment_id: The ID of the experiment to backfill.
        scorers: A list of BackfillScorer objects or a list of scorer names.
            If a list of scorer names is provided, the current sample rates from the experiment's
            scorers will be used. If BackfillScorer objects are provided, their sample_rate will be used as override
        start_time: The start time for backfill (optional).
        end_time: The end time for backfill (optional).

    Returns:
        str: The job ID of the created backfill job
    """
    experiment_id = _get_experiment_id(experiment_id)

    scorer_sample_rates = {}

    registered_scorers = _get_managed_evals_client().get_scheduled_scorers(experiment_id=experiment_id)

    scheduled_scorer_map = {
        scorer_config.scorer.name: scorer_config.sample_rate for scorer_config in registered_scorers
    }

    if isinstance(scorers, list) and all(isinstance(s, str) for s in scorers):
        for scorer_name in scorers:
            if scorer_name not in scheduled_scorer_map:
                raise ValueError(
                    f"Scheduled scorer '{scorer_name}' not found in experiment '{experiment_id}'. "
                    f"Available scorers: {list(scheduled_scorer_map.keys())}"
                )
            scorer_sample_rates[scorer_name] = scheduled_scorer_map[scorer_name]
    elif isinstance(scorers, list) and all(isinstance(s, BackfillScorerConfig) for s in scorers):
        # If scorers is a list of BackfillScorer objects, use their sample rates
        for backfill_scorer in scorers:
            sample_rate = backfill_scorer.sample_rate
            if sample_rate is None:
                # Use the scorer's own sample rate if not overridden
                if backfill_scorer.scorer.name not in scheduled_scorer_map:
                    raise ValueError(
                        f"Scheduled scorer '{backfill_scorer.scorer.name}' not found in experiment '{experiment_id}'. "
                        f"Available scorers: {list(scheduled_scorer_map.keys())}"
                    )
                sample_rate = scheduled_scorer_map[backfill_scorer.scorer.name].sample_rate
            scorer_sample_rates[backfill_scorer.scorer.name] = sample_rate
    else:
        raise ValueError("scorers must be either a list of strings (scorer names) or a list of BackfillScorer objects")

    start_timestamp_ms = None
    end_timestamp_ms = None

    # When neither start_time nor end_time are provided, set start_time to 7 days ago and end_time to current time
    if start_time is None and end_time is None:
        now = datetime.now()
        seven_days_ago = now.timestamp() - (7 * 24 * 60 * 60)  # 7 days in seconds
        start_timestamp_ms = int(seven_days_ago * 1000)
        end_timestamp_ms = int(now.timestamp() * 1000)
    else:
        if start_time is not None:
            start_timestamp_ms = int(start_time.timestamp() * 1000)

        if end_time is not None:
            end_timestamp_ms = int(end_time.timestamp() * 1000)

    result = _get_managed_evals_client().run_metric_backfill(
        experiment_id=experiment_id,
        scorer_sample_rates=scorer_sample_rates,
        start_timestamp_ms=start_timestamp_ms,
        end_timestamp_ms=end_timestamp_ms,
    )

    scorers_str = ", ".join(scorer_sample_rates.keys())
    print(f'Backfill of metrics {scorers_str} in progress. For status, please see the job {result["job_id"]}')
    return result["job_id"]
