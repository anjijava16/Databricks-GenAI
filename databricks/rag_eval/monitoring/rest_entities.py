"""REST API entities for monitoring."""

import dataclasses
from typing import Literal, Optional

from dataclasses_json import config, dataclass_json

from databricks.rag_eval.evaluation import custom_metrics
from databricks.rag_eval.monitoring import assessments, entities
from databricks.rag_eval.monitoring.assessments import CustomMetric


@dataclass_json
@dataclasses.dataclass
class MonitorSettings:
    """
    REST API entity for monitor settings.
    Fields are optional to support partial updates.
    """

    initial_lookback_period_hours: Optional[int] = None
    max_traces_to_process_in_job: Optional[int] = None
    rag_eval_max_workers: Optional[int] = None
    rag_eval_rate_limit_quota: Optional[int] = None
    enable_metrics_job: Optional[bool] = None
    get_trace_max_workers: Optional[int] = None
    max_traces_per_batch: Optional[int] = None
    search_traces_page_size: Optional[int] = None

    def to_entity(self) -> entities.MonitorSettings:
        """
        Convert to the non-optional MonitorSettings entity, validating that required fields are present.

        Returns:
            entities.MonitorSettings: The validated monitor settings

        Raises:
            ValueError: If any required field is None
        """
        # Validate required fields
        required_fields = [
            "initial_lookback_period_hours",
            "max_traces_to_process_in_job",
            "rag_eval_max_workers",
            "rag_eval_rate_limit_quota",
            "get_trace_max_workers",
            "max_traces_per_batch",
            "search_traces_page_size",
        ]

        missing_fields = []
        for field in required_fields:
            if getattr(self, field) is None:
                missing_fields.append(field)

        if missing_fields:
            raise ValueError(f"Missing required monitor settings: {', '.join(missing_fields)}")

        # Required fields (validated above)
        settings = {
            "initial_lookback_period_hours": self.initial_lookback_period_hours,
            "max_traces_to_process_in_job": self.max_traces_to_process_in_job,
            "trace_processing_concurrency": self.trace_processing_concurrency,
            "harness_evaluation_batch_size": self.harness_evaluation_batch_size,
            "rag_eval_max_workers": self.rag_eval_max_workers,
            "rag_eval_rate_limit_quota": self.rag_eval_rate_limit_quota,
            # By default, run the metrics job.
            "enable_metrics_job": (self.enable_metrics_job if self.enable_metrics_job is not None else True),
        }

        # Optional fields - only include if not None
        # These will default to the values set in entities.py
        optional_fields = [
            "get_trace_max_workers",
            "max_traces_per_batch",
            "search_traces_page_size",
        ]
        for field in optional_fields:
            value = getattr(self, field)
            if value is not None:
                settings[field] = value

        # Convert to entity with non-optional fields
        return entities.MonitorSettings(**settings)


_FIELD_IS_UPDATABLE = "FIELD_IS_UPDATABLE"


def ExcludeIfNone(value):
    return value is None


@dataclass_json
@dataclasses.dataclass
class AssessmentConfig:
    name: Optional[str] = None
    sample_rate: Optional[float] = dataclasses.field(
        default=None,
        metadata=config(exclude=ExcludeIfNone),
    )


@dataclass_json
@dataclasses.dataclass
class CustomMetricConfig:
    name: Optional[str] = None
    function_body: Optional[str] = None
    sample_rate: Optional[float] = dataclasses.field(
        default=None,
        metadata=config(exclude=ExcludeIfNone),
    )


@dataclass_json
@dataclasses.dataclass
class NamedGuidelineEntry:
    key: Optional[str] = None
    guidelines: Optional[list[str]] = None


@dataclass_json
@dataclasses.dataclass
class NamedGuidelines:
    entries: Optional[list[NamedGuidelineEntry]]


@dataclass_json
@dataclasses.dataclass
class EvaluationConfig:
    metrics: Optional[list[AssessmentConfig]] = None
    no_metrics: Optional[bool] = None
    named_guidelines: Optional[NamedGuidelines] = dataclasses.field(
        default=None, metadata=config(exclude=ExcludeIfNone)
    )
    custom_metrics: Optional[list[CustomMetricConfig]] = None


@dataclass_json
@dataclasses.dataclass
class SamplingConfig:
    sampling_rate: Optional[float] = None


@dataclass_json
@dataclasses.dataclass
class ScheduleConfig:
    pause_status: Optional[Literal["UNPAUSED", "PAUSED"]] = None


@dataclass_json
@dataclasses.dataclass
class Monitor:
    experiment_id: Optional[str] = dataclasses.field(
        default=None,
        metadata={**config(exclude=ExcludeIfNone), _FIELD_IS_UPDATABLE: False},
    )
    evaluation_config: Optional[EvaluationConfig] = dataclasses.field(
        default=None, metadata=config(exclude=ExcludeIfNone)
    )
    sampling: Optional[SamplingConfig] = dataclasses.field(default=None, metadata=config(exclude=ExcludeIfNone))
    schedule: Optional[ScheduleConfig] = dataclasses.field(default=None, metadata=config(exclude=ExcludeIfNone))
    is_agent_external: Optional[bool] = dataclasses.field(
        default=None,
        metadata={**config(exclude=ExcludeIfNone), _FIELD_IS_UPDATABLE: False},
    )
    evaluated_traces_table: Optional[str] = dataclasses.field(
        default=None,
        metadata={**config(exclude=ExcludeIfNone), _FIELD_IS_UPDATABLE: False},
    )
    trace_archive_table: Optional[str] = dataclasses.field(
        default=None,
        metadata={**config(exclude=ExcludeIfNone), _FIELD_IS_UPDATABLE: False},
    )

    def get_update_mask(self) -> str:
        """Get the update mask for the fields that have changed."""
        return ",".join(
            field.name
            for field in dataclasses.fields(self)
            if getattr(self, field.name) is not None and field.metadata.get(_FIELD_IS_UPDATABLE, True)
        )


@dataclass_json
@dataclasses.dataclass
class MonitorInfo:
    monitor: Optional[Monitor] = None
    endpoint_name: Optional[str] = None
    current_backfill_job_id: Optional[str] = None

    @property
    def is_external(self) -> bool:
        """Returns True if the monitor is external."""
        return self.monitor and self.monitor.is_agent_external

    def _get_assessments_suite_config(self) -> assessments.AssessmentsSuiteConfig:
        """Extract common monitoring configuration logic."""
        monitor = self.monitor or Monitor()
        sampling_config = monitor.sampling or SamplingConfig()
        evaluation_config = monitor.evaluation_config or EvaluationConfig()
        assessment_configs = evaluation_config.metrics or []
        custom_metric_configs = evaluation_config.custom_metrics or []

        # Convert named guidelines
        global_guidelines = None
        if (
            monitor.evaluation_config
            and monitor.evaluation_config.named_guidelines
            and monitor.evaluation_config.named_guidelines.entries
        ):
            global_guidelines = {
                entry.key: entry.guidelines
                for entry in monitor.evaluation_config.named_guidelines.entries
                if entry.key and entry.guidelines
            }

        pause_status = monitor.schedule.pause_status if monitor.schedule else None

        judges = []
        for cfg in assessment_configs:
            if cfg.name != assessments.AI_JUDGE_GUIDELINE_ADHERENCE:
                judges.append(assessments.BuiltinJudge(name=cfg.name, sample_rate=cfg.sample_rate))
            # Note that we do not create a GuidelinesJudge if the guidelines are empty,
            # even if `guideline_adherence` is included in assessment configs from the server.
            # This is because having a GuidelinesJudge without guidelines doesn't make sense.
            elif cfg.name == assessments.AI_JUDGE_GUIDELINE_ADHERENCE and global_guidelines:
                judges.append(
                    assessments.GuidelinesJudge(
                        guidelines=global_guidelines,
                        sample_rate=cfg.sample_rate,
                    )
                )

        custom_metric_configs = [
            CustomMetric(
                metric_fn=custom_metrics._create_metric_from_function_body(
                    custom_metric.name, custom_metric.function_body
                ),
                sample_rate=custom_metric.sample_rate,
            )
            for custom_metric in custom_metric_configs
        ]

        return assessments.AssessmentsSuiteConfig(
            sample=sampling_config.sampling_rate,
            paused=pause_status == entities.SchedulePauseStatus.PAUSED,
            assessments=judges + custom_metric_configs,
        )

    def to_monitor(self) -> entities.Monitor:
        """Converts the REST API response to a Python Monitor object."""
        monitor = self.monitor or Monitor()
        assert not monitor.is_agent_external, "Monitor is external"

        return entities.Monitor(
            endpoint_name=self.endpoint_name,
            evaluated_traces_table=monitor.evaluated_traces_table,
            trace_archive_table=monitor.trace_archive_table,
            experiment_id=monitor.experiment_id,
            assessments_config=self._get_assessments_suite_config(),
            current_backfill_job_id=self.current_backfill_job_id,
        )

    def to_external_monitor(self) -> entities.ExternalMonitor:
        """Converts the REST API response to an External Monitor object."""
        monitor = self.monitor or Monitor()
        assert monitor.is_agent_external, "Monitor is internal"

        return entities.ExternalMonitor(
            assessments_config=self._get_assessments_suite_config(),
            experiment_id=monitor.experiment_id,
            trace_archive_table=monitor.trace_archive_table,
            _checkpoint_table=monitor.evaluated_traces_table,
            _legacy_ingestion_endpoint_name=self.endpoint_name,
            current_backfill_job_id=self.current_backfill_job_id,
        )


@dataclass_json
@dataclasses.dataclass
class JobCompletionEvent:
    success: Optional[bool] = None
    error_message: Optional[str] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None


@dataclass_json
@dataclasses.dataclass
class MonitoringEvent:
    job_start: Optional[dict] = None
    job_completion: Optional[JobCompletionEvent] = None
    source: Optional[str] = None


@dataclasses.dataclass
class MonitoringMetric:
    num_traces: Optional[int] = None
    num_traces_evaluated: Optional[int] = None

    def to_dict(self) -> dict:
        output_dict = {}
        if self.num_traces is not None:
            output_dict["num_traces"] = self.num_traces

        if self.num_traces_evaluated is not None:
            output_dict["num_traces_evaluated"] = self.num_traces_evaluated
        return output_dict


@dataclass_json
@dataclasses.dataclass
class BuiltinScorer:
    name: Optional[str] = None


@dataclass_json
@dataclasses.dataclass
class CustomScorer:
    pass


@dataclass_json
@dataclasses.dataclass
class ScorerConfig:
    name: Optional[str] = None
    serialized_scorer: Optional[str] = None
    builtin: Optional[BuiltinScorer] = dataclasses.field(default=None, metadata=config(exclude=ExcludeIfNone))
    custom: Optional[CustomScorer] = dataclasses.field(default=None, metadata=config(exclude=ExcludeIfNone))
    sample_rate: Optional[float] = None
    filter_string: Optional[str] = None


@dataclass_json
@dataclasses.dataclass
class ScheduledScorers:
    scorers: Optional[list[ScorerConfig]] = None


@dataclass_json
@dataclasses.dataclass
class ScheduledScorersInfo:
    experiment_id: Optional[str] = None
    scheduled_scorers: Optional[ScheduledScorers] = None
