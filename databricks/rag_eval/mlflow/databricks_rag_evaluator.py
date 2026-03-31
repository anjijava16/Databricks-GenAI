import json
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import pandas as pd
from jinja2 import Environment, PackageLoader, select_autoescape
from mlflow import environment_variables as mlflow_env_vars
from mlflow import models as mlflow_models
from mlflow.models import evaluation as mlflow_evaluation
from mlflow.models.evaluation import artifacts as mlflow_artifacts
from mlflow.utils import mlflow_tags

from databricks.rag_eval import (
    context,
    env_vars,
    evaluator_plugin,
    session,
)
from databricks.rag_eval.config import evaluation_config
from databricks.rag_eval.evaluation import harness, per_run_metrics
from databricks.rag_eval.mlflow import mlflow_utils
from databricks.rag_eval.utils import error_utils, workspace_url_resolver


def _log_pandas_df_artifact(pandas_df, artifact_name):
    """
    Logs a pandas DataFrame as a JSON artifact, then returns an EvaluationArtifact object.
    """
    mlflow_artifact_name = f"{artifact_name}.json"
    mlflow.log_table(pandas_df, mlflow_artifact_name)
    return mlflow_artifacts.JsonEvaluationArtifact(
        uri=mlflow.get_artifact_uri(mlflow_artifact_name),
    )


# Used to display summary and instructions to the user after evaluation is complete.
_AI_ICON_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" aria-hidden="true" focusable="false" class=""><path fill="currentColor" fill-rule="evenodd" d="M10.726 8.813 13.199 8l-2.473-.813a3 3 0 0 1-1.913-1.913L8 2.801l-.813 2.473a3 3 0 0 1-1.913 1.913L2.801 8l2.473.813a3 3 0 0 1 1.913 1.913L8 13.199l.813-2.473a3 3 0 0 1 1.913-1.913Zm2.941.612c1.376-.452 1.376-2.398 0-2.85l-2.473-.813a1.5 1.5 0 0 1-.956-.956l-.813-2.473c-.452-1.376-2.398-1.376-2.85 0l-.813 2.473a1.5 1.5 0 0 1-.956.956l-2.473.813c-1.376.452-1.376 2.398 0 2.85l2.473.813a1.5 1.5 0 0 1 .956.957l.813 2.472c.452 1.376 2.398 1.376 2.85 0l.813-2.473a1.5 1.5 0 0 1 .957-.956l2.472-.813Z" clip-rule="evenodd"></path></svg>
"""


def _display_summary_and_usage_instructions(run_id: str):
    """
    Displays summary of the evaluation result, errors and warnings if any,
    and instructions on what to do after running `mlflow.evaluate`.
    """

    run = mlflow.get_run(run_id)
    if mlflow_tags.MLFLOW_DATABRICKS_WORKSPACE_URL in run.data.tags:
        # Include Databricks URLs in the displayed message.
        workspace_url = run.data.tags[mlflow_tags.MLFLOW_DATABRICKS_WORKSPACE_URL]
        resolver = workspace_url_resolver.WorkspaceUrlResolver(workspace_url)
        eval_results_url = resolver.resolve_url_for_mlflow_evaluation_results(run.info)

        # Create a Jinja2 environment and load the template
        env = Environment(
            loader=PackageLoader("databricks.rag_eval", "templates"),
            autoescape=select_autoescape(["html"]),
        )
        template = env.get_template("eval_output.html")

        # Render the template with the data
        rendered_html = template.render(
            {
                "summary_text": None,
                "eval_results_url": eval_results_url,
            }
        )

        context.get_context().display_html(rendered_html)
    else:
        print(
            """Evaluation completed.

Metrics and evaluation results can be viewed from the MLflow run page.
To compare evaluation results across runs, view the "Evaluations" tab of the experiment.

Get aggregate metrics: `result.metrics`.
Get per-row evaluation results: `result.tables['eval_results']`.
`result` is the `EvaluationResult` object returned by `mlflow.evaluate`.
"""
        )


def _log_evaluation_input_artifacts(
    config: evaluation_config.GlobalEvaluationConfig,
):
    """
    Logs the configuration to MLflow.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir, "eval_config.json")
        config_str = json.dumps(config.to_dict())
        path.write_text(config_str)
        mlflow.log_artifact(str(path))


class DatabricksRagEvaluator(mlflow_evaluation.base.ModelEvaluator):
    @classmethod
    def can_evaluate(cls, *, model_type, evaluator_config, **kwargs) -> bool:
        """
        See parent class docstring.
        """
        return model_type == evaluator_plugin.MODEL_TYPE

    @context.eval_context
    def evaluate(
        self,
        *,
        model_type,
        dataset,
        run_id,
        evaluator_config: Optional[Dict[str, Any]] = None,
        model=None,
        custom_metrics=None,
        extra_metrics=None,
        custom_artifacts=None,
        baseline_model=None,
        predictions=None,
        **kwargs,
    ):
        """
        Runs Databricks RAG evaluation on the provided dataset.

        The following arguments are supported:
        - model_type: Must be the same as evaluator_plugin.MODEL_TYPE
        - dataset
        - run_id

        For more details, see parent class docstring.
        """
        try:
            if evaluator_config is None:
                evaluator_config = {}

            config = evaluation_config.GlobalEvaluationConfig.from_mlflow_evaluate_args(evaluator_config, extra_metrics)

            eval_dataset = mlflow_utils.mlflow_dataset_to_evaluation_dataset(dataset)

            # Set batch size to the context
            session.current_session().set_session_batch_size(len(eval_dataset.eval_items))
            run_info = mlflow.get_run(run_id)
            experiment_id = run_info.info.experiment_id

            eval_results = harness.run(
                model=model,
                eval_dataset=eval_dataset,
                config=config,
                experiment_id=experiment_id,
                run_id=run_id,
            )

            mlflow_per_run_metrics = per_run_metrics.generate_per_run_metrics(eval_results, config=config)

            search_traces_df = mlflow.search_traces(experiment_ids=[experiment_id], run_id=run_id)
            search_traces_artifact = _log_pandas_df_artifact(search_traces_df, "eval_results")
            artifacts = {"eval_results": search_traces_artifact}

            if env_vars.AGENT_EVAL_LOG_ARTIFACTS_TO_MLFLOW_ENABLED.get():
                _log_evaluation_input_artifacts(config)
                # TODO(ML-54543): Remove this once we have migrated to the new eval results artifact
                eval_results_df = pd.DataFrame([result.to_pd_series() for result in eval_results])
                eval_results_artifact = _log_pandas_df_artifact(eval_results_df, "legacy_eval_results")
                artifacts["legacy_eval_results"] = eval_results_artifact

            mlflow.log_metrics(mlflow_per_run_metrics)

            # Check for failed scorers and log a warning
            failed_scorers = set()
            for result in eval_results:
                for assessment in result.assessment_results:
                    if hasattr(assessment, "rating") and assessment.rating.error_message is not None:
                        failed_scorers.add(assessment.assessment_name)
                for metric in result.metric_results:
                    if metric.metric_value.error is not None:
                        failed_scorers.add(metric.metric_value.name)

            if failed_scorers:
                failed_scorers_str = ", ".join(sorted(failed_scorers))
                warnings.warn(
                    f"Some scorers failed during evaluation: {failed_scorers_str}. "
                    f"Please check the evaluation result page for more details."
                )

            _display_summary_and_usage_instructions(run_id)

            if mlflow_env_vars.MLFLOW_MAX_TRACES_TO_DISPLAY_IN_NOTEBOOK.unset():
                # Note that this environment variable is evaluated post-run, so we cannot set it
                # back to the original value within this function.
                mlflow_env_vars.MLFLOW_MAX_TRACES_TO_DISPLAY_IN_NOTEBOOK.set("50")

            # TODO(ML-54543): Replace the traces here with those returned from search_traces_df
            #                 and remove unnecessary `mlflow.get_trace` calls.
            traces = [result.eval_item.trace for result in eval_results]
            mlflow.tracing.display.get_display_handler().display_traces(traces)

            return mlflow_models.EvaluationResult(metrics=mlflow_per_run_metrics, artifacts=artifacts)

        except error_utils.ValidationError as e:
            # Scrub trace for user-facing validation errors
            raise error_utils.ValidationError(str(e)) from None
