"""Methods and classes for working with configuration files."""

import dataclasses
from typing import Any, Dict, List, Mapping, Optional, Set, Union

import yaml

from databricks.rag_eval.config import assessment_config
from databricks.rag_eval.evaluation import custom_metrics as agent_custom_metrics
from databricks.rag_eval.utils import error_utils, input_output_utils

BUILTIN_ASSESSMENTS_KEY = "builtin_assessments"
IS_DEFAULT_CONFIG_KEY = "is_default_config"

EVALUATOR_CONFIG__METRICS_KEY = "metrics"
EVALUATOR_CONFIG__GLOBAL_GUIDELINES_KEY = "global_guidelines"
ALLOWED_EVALUATOR_CONFIG_KEYS = {
    EVALUATOR_CONFIG__METRICS_KEY,
    EVALUATOR_CONFIG__GLOBAL_GUIDELINES_KEY,
}

EVALUATOR_CONFIG_ARGS__EXTRA_METRICS_KEY = "extra_metrics"

JSON_STR__METRICS_KEY = "metrics"
JSON_STR__CUSTOM_METRICS_KEY = "custom_metrics"
JSON_STR__GLOBAL_GUIDELINES_KEY = "global_guidelines"


@dataclasses.dataclass
class _BaseEvaluationConfig:
    is_default_config: bool
    custom_metrics: List[agent_custom_metrics.CustomMetric] = dataclasses.field(default_factory=list)
    global_guidelines: Optional[Dict[str, List[str]]] = None


@dataclasses.dataclass
class ItemEvaluationConfig(_BaseEvaluationConfig):
    assessment_configs: List[assessment_config.AssessmentConfig] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class GlobalEvaluationConfig(_BaseEvaluationConfig):
    """Abstraction for `evaluation` config"""

    global_assessment_configs: List[assessment_config.AssessmentConfig] = dataclasses.field(default_factory=list)
    per_item_assessments: dict[
        str,
        list[Union[assessment_config.AssessmentConfig, agent_custom_metrics.CustomMetric]],
    ] = dataclasses.field(
        default_factory=dict
    )  # key: question_id, value: list of assessments (used for monitoring job only)

    def __post_init__(self):
        if self.global_assessment_configs is None:
            self.global_assessment_configs = []

        if self.per_item_assessments is None:
            self.per_item_assessments = {}

        # At most one of global_assessment_configs or per_item_assessment_configs can be non-empty.
        if self.global_assessment_configs and self.per_item_assessments:
            raise error_utils.ValidationError(
                "GlobalEvaluationConfig cannot have both global and per-item assessment configs."
            )

    @classmethod
    def _from_dict(cls, config_dict: Mapping[str, Any]):
        if BUILTIN_ASSESSMENTS_KEY not in config_dict:
            raise error_utils.ValidationError(f"Invalid config {config_dict}: `{BUILTIN_ASSESSMENTS_KEY}` required.")

        try:
            builtin_assessment_configs = config_dict.get(BUILTIN_ASSESSMENTS_KEY, [])

            # Global guidelines
            global_guidelines = config_dict.get(EVALUATOR_CONFIG__GLOBAL_GUIDELINES_KEY, None)
            # Run guideline adherence judge if global guidelines are provided
            if (
                global_guidelines is not None
                and assessment_config.GLOBAL_GUIDELINE_ADHERENCE.assessment_name not in builtin_assessment_configs
            ):
                builtin_assessment_configs.append(assessment_config.GLOBAL_GUIDELINE_ADHERENCE.assessment_name)

            builtin_assessment_configs = assessment_config.create_builtin_assessment_configs(builtin_assessment_configs)
        except (TypeError, KeyError, ValueError) as error:
            raise error_utils.ValidationError(f"Invalid config `{config_dict[BUILTIN_ASSESSMENTS_KEY]}`: {error}")
        # Handle errors internally as we don't want to surface that
        # the extra metrics are handled as a "config"
        extra_metrics = config_dict.get(EVALUATOR_CONFIG_ARGS__EXTRA_METRICS_KEY, None)
        # EvaluationMetric classes, i.e. from make_genai_metric_from_prompt. @metric functions
        # are handled separately.
        legacy_custom_assessment_configs = assessment_config.create_custom_eval_metric_assessment_configs(extra_metrics)
        assessment_confs = builtin_assessment_configs + legacy_custom_assessment_configs
        all_names = [assessment_conf.assessment_name for assessment_conf in assessment_confs]
        dups = {name for name in all_names if all_names.count(name) > 1}
        if dups:
            raise error_utils.ValidationError(
                f"Invalid config `{config_dict}`: assessment names must be unique. Found duplicate assessment names: {dups}"
            )

        # Custom metrics
        custom_metrics = [
            metric for metric in extra_metrics or [] if isinstance(metric, agent_custom_metrics.CustomMetric)
        ]
        seen_custom_metric_names = set()
        for metric in custom_metrics:
            if metric.name in seen_custom_metric_names:
                raise error_utils.ValidationError(
                    f"Invalid config `{config_dict}`: custom metric names must be unique. Found duplicate custom metric name: {metric.name}"
                )
            seen_custom_metric_names.add(metric.name)

        try:
            result = cls(
                is_default_config=config_dict[IS_DEFAULT_CONFIG_KEY],
                global_assessment_configs=assessment_confs,
                custom_metrics=custom_metrics,
                global_guidelines=global_guidelines,
            )
        except (TypeError, KeyError, ValueError) as error:
            raise error_utils.ValidationError(f"Invalid config `{config_dict}`: {error}")

        return result

    @classmethod
    def from_mlflow_evaluate_args(
        cls,
        evaluator_config: Optional[Mapping[str, Any]],
        extra_metrics: Optional[List[Any]] = None,
    ) -> "GlobalEvaluationConfig":
        """Reads the config from an evaluator config"""
        if evaluator_config is None:
            evaluator_config = {}

        invalid_keys = set(evaluator_config.keys()) - ALLOWED_EVALUATOR_CONFIG_KEYS
        if invalid_keys:
            raise error_utils.ValidationError(
                f"Invalid keys in evaluator config: {', '.join(invalid_keys)}. "
                f"Allowed keys: {ALLOWED_EVALUATOR_CONFIG_KEYS}"
            )

        if EVALUATOR_CONFIG__METRICS_KEY in evaluator_config:
            metrics_list = evaluator_config[EVALUATOR_CONFIG__METRICS_KEY]
            if not isinstance(metrics_list, list) or not all(isinstance(metric, str) for metric in metrics_list):
                raise error_utils.ValidationError(
                    f"Invalid metrics: {metrics_list}. " f"Must be a list of metric names."
                )
            config_dict = {
                BUILTIN_ASSESSMENTS_KEY: metrics_list,
                IS_DEFAULT_CONFIG_KEY: False,
            }
        else:
            config_dict = default_config_dict()
            config_dict[IS_DEFAULT_CONFIG_KEY] = True

        if EVALUATOR_CONFIG__GLOBAL_GUIDELINES_KEY in evaluator_config:
            global_guidelines = evaluator_config[EVALUATOR_CONFIG__GLOBAL_GUIDELINES_KEY]

            # Convert list of guidelines to a default mapping
            global_guidelines_mapping = (
                {assessment_config.GLOBAL_GUIDELINE_ADHERENCE.user_facing_assessment_name: global_guidelines}
                if isinstance(global_guidelines, list)
                else global_guidelines
            )

            if not input_output_utils.is_valid_guidelines_mapping(global_guidelines_mapping):
                raise error_utils.ValidationError(
                    f"Invalid global guidelines: {global_guidelines}. Global guidelines must be a list of strings "
                    f"or a mapping from a name of guidelines (string) to a list of strings."
                )
            input_output_utils.check_guidelines_mapping_exceeds_limit(global_guidelines_mapping)

            config_dict[EVALUATOR_CONFIG__GLOBAL_GUIDELINES_KEY] = global_guidelines_mapping

        if extra_metrics is not None:
            config_dict[EVALUATOR_CONFIG_ARGS__EXTRA_METRICS_KEY] = extra_metrics

        return cls._from_dict(config_dict)

    def to_dict(self):
        builtin_configs = [
            conf
            for conf in self.global_assessment_configs
            if isinstance(conf, assessment_config.BuiltinAssessmentConfig)
        ]
        metric_names = [conf.assessment_name for conf in builtin_configs]
        output_dict = {
            JSON_STR__METRICS_KEY: metric_names,
        }
        if self.global_guidelines:
            output_dict[JSON_STR__GLOBAL_GUIDELINES_KEY] = self.global_guidelines
        if self.custom_metrics:
            output_dict[JSON_STR__CUSTOM_METRICS_KEY] = [metric.name for metric in self.custom_metrics]

        return output_dict

    def get_eval_item_eval_config(self, question_id: str) -> ItemEvaluationConfig:
        """Returns the evaluation config for a specific question_id.

        If the request does not have a specific config, it defaults to
        using the global config to generate the ItemEvaluationConfig.

        Note that global evaluation configs are not allowed to have
        both global and per-item assessment configs. This is enforced by
        the `__post_init__` method of this class.

        Args:
            question_id (str): The question ID to get the config for.

        Returns:
            ItemEvaluationConfig: The config for evaluating a given request.
        """
        if not self.per_item_assessments:
            assessment_configs = self.global_assessment_configs
            custom_metrics = self.custom_metrics
        elif question_id not in self.per_item_assessments:
            raise error_utils.ValidationError(
                f"No per-item assessment configs found for question (question_id=`{question_id}`)."
            )
        else:
            assessment_configs = [
                assessment
                for assessment in self.per_item_assessments[question_id]
                if isinstance(assessment, assessment_config.AssessmentConfig)
            ]
            custom_metrics = [
                assessment
                for assessment in self.per_item_assessments[question_id]
                if isinstance(assessment, agent_custom_metrics.CustomMetric)
            ]
        return ItemEvaluationConfig(
            is_default_config=self.is_default_config,
            assessment_configs=assessment_configs,
            custom_metrics=custom_metrics,
            global_guidelines=self.global_guidelines,
        )


def default_config() -> str:
    """Returns the default config (in YAML)"""
    return """
builtin_assessments:
  - safety
  - groundedness
  - correctness
  - relevance_to_query
  - chunk_relevance
  - context_sufficiency
  - guideline_adherence
"""


def default_config_dict() -> Dict[str, Any]:
    """Returns the default config as a dictionary"""
    return yaml.safe_load(default_config())


def unnecessary_metrics_with_expected_response_or_expected_facts() -> Set[str]:
    """
    Returns a list of unnecessary metrics to not run when expected response or expected facts are
    provided. In this case, we can skip relevance to query and chunk relevance because their ground
    truth counterparts, correctness and context sufficiency, are more informative.
    """
    return {
        assessment_config.RELEVANCE_TO_QUERY.assessment_name,
        assessment_config.CHUNK_RELEVANCE.assessment_name,
    }


def metrics_requiring_ground_truth_or_expected_facts() -> Set[str]:
    """
    Returns a list of unnecessary metrics to not run when no ground truth, or expected facts, or
    grading notes are provided. In this case, we can skip correctness and context sufficiency
    because they require ground truth or expected facts (or grading notes for correctness). Instead,
    we run their less informative counterparts, relevance to query and chunk relevance.
    """
    return {
        assessment_config.CORRECTNESS.assessment_name,
        assessment_config.CONTEXT_SUFFICIENCY.assessment_name,
    }
