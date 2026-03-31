from typing import TYPE_CHECKING

import mlflow

import databricks.sdk
from databricks.agents.utils.endpoint_utils import _get_endpoint_experiment_id_opt
from databricks.agents.utils.mlflow_utils import _is_mlflow_3_or_above, get_model_info, get_workspace_id

if TYPE_CHECKING:
    from databricks.agents.review_app import ReviewApp


def _get_experiment_id_for_deployment(model_name: str, model_version: str, endpoint_name: str) -> str:
    workspace_client = databricks.sdk.WorkspaceClient()
    endpoint = workspace_client.serving_endpoints.get(endpoint_name)
    # If the endpoint has an experiment, use this experiment ID.
    # All new agent deployments will have an endpoint experiment
    if experiment_id := _get_endpoint_experiment_id_opt(endpoint):
        return experiment_id
    # For existing endpoints that haven't yet been updated to have
    # an "endpoint experiment", fall back to previous logic for
    # getting the experiment ID from the model version.
    client = mlflow.MlflowClient()
    model_version_info = client.get_model_version(model_name, model_version)
    mlflow_model_info = get_model_info(model_name, model_version)
    if _is_mlflow_3_or_above(mlflow_model_info.mlflow_version):
        # In MLflow 3.x, model logging is not always done under a run.
        return client.get_logged_model(model_version_info.model_id).experiment_id
    # In MLflow 2.x, model logging is always done under a run.
    return client.get_run(model_version_info.run_id).info.experiment_id


def get_review_app_v2_from_model_version(model_name: str, model_version: str, endpoint_name: str) -> "ReviewApp":
    from databricks.agents.review_app import get_review_app

    experiment_id = _get_experiment_id_for_deployment(
        model_name=model_name,
        model_version=model_version,
        endpoint_name=endpoint_name,
    )
    # This is idempotent, acts as get_or_create.
    review_app_v2 = get_review_app(experiment_id)
    # This is idempotent as long as the agent_name and model_serving_endpoint are stable.
    review_app_v2.add_agent(
        agent_name=model_name,
        model_serving_endpoint=endpoint_name,
    )
    return review_app_v2


def get_review_app_v2_url(review_app_v2: "ReviewApp") -> str:
    base_url = f"{review_app_v2.url}/chat"
    workspace_id = get_workspace_id()
    if workspace_id is not None:
        base_url += f"?o={workspace_id}"
    return base_url
