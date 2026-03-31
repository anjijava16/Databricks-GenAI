import dataclasses
from typing import Optional

from dataclasses_json import dataclass_json

from databricks.rag_eval.monitoring import assessments, utils
from databricks.rag_eval.utils.enum_utils import StrEnum


class SchedulePauseStatus(StrEnum):
    UNPAUSED = "UNPAUSED"
    PAUSED = "PAUSED"


@dataclass_json
@dataclasses.dataclass
class MonitorSettings:
    """
    Service-specified settings for a monitor.
    All fields are required and validated by the to_entity method in rest_entities.MonitorSettings.
    """

    # TODO(avesh): Clean up unused settings
    initial_lookback_period_hours: int = 24
    max_traces_to_process_in_job: int = 10000
    rag_eval_max_workers: int = 10
    rag_eval_rate_limit_quota: int = 8
    enable_metrics_job: bool = True
    get_trace_max_workers: int = 10
    max_traces_per_batch: int = 200
    search_traces_page_size: int = 500


@dataclass_json
@dataclasses.dataclass
class MonitoringConfig:
    """
    Deprecated. Configuration for monitoring an GenAI application. All fields are optional for upsert semantics.
    """

    sample: Optional[float] = None
    metrics: Optional[list[str]] = None
    paused: Optional[bool] = None
    global_guidelines: Optional[dict[str, list[str]]] = None

    @classmethod
    def from_assessments_suite_config(
        cls, assessments_suite_config: assessments.AssessmentsSuiteConfig
    ) -> "MonitoringConfig":
        cfg = assessments_suite_config

        # metrics and guidelines uniqueness is enforced in AssessmentsSuiteConfig
        metrics: list[str] | None = None
        global_guidelines: dict[str, list[str]] | None = None

        if cfg.assessments is not None:
            metrics = []
            global_guidelines = {}

            for assessment in cfg.assessments:
                if isinstance(assessment, assessments.GuidelinesJudge):
                    metrics.append(assessments.AI_JUDGE_GUIDELINE_ADHERENCE)
                    global_guidelines = {k: v.copy() for k, v in assessment.guidelines.items()}

                if isinstance(assessment, assessments.BuiltinJudge):
                    metrics.append(str(assessment.name))

        return cls(
            sample=cfg.sample,
            paused=cfg.paused,
            metrics=metrics,
            global_guidelines=global_guidelines,
        )

    def to_assessments_suite_config(self) -> assessments.AssessmentsSuiteConfig:
        ls_assessments: list["assessments.AssessmentConfig"] | None = None

        if self.metrics is not None:
            ls_assessments = []
            for metric in self.metrics or []:
                if metric == assessments.AI_JUDGE_GUIDELINE_ADHERENCE:
                    ls_assessments.append(
                        assessments.GuidelinesJudge(
                            guidelines=(
                                {k: v.copy() for k, v in self.global_guidelines.items()}
                                if self.global_guidelines
                                else {}
                            )
                        )
                    )
                else:
                    ls_assessments.append(assessments.BuiltinJudge(name=metric))

        return assessments.AssessmentsSuiteConfig(
            sample=self.sample,
            paused=self.paused,
            assessments=ls_assessments,
        )


@dataclass_json
@dataclasses.dataclass
class Monitor:
    """
    The monitor for a serving endpoint.
    """

    experiment_id: str
    endpoint_name: str
    assessments_config: assessments.AssessmentsSuiteConfig
    evaluated_traces_table: str
    trace_archive_table: str | None
    current_backfill_job_id: str | None = None

    @property
    def monitoring_page_url(self) -> str:
        return utils.get_monitoring_page_url(self.experiment_id)


@dataclass_json
@dataclasses.dataclass
class ExternalMonitor:
    """
    The monitor for a GenAI application served outside of Databricks.
    """

    experiment_id: str
    assessments_config: assessments.AssessmentsSuiteConfig
    trace_archive_table: str | None
    _checkpoint_table: str
    _legacy_ingestion_endpoint_name: str
    current_backfill_job_id: str | None = None

    @property
    def monitoring_page_url(self) -> str:
        return utils.get_monitoring_page_url(self.experiment_id)
