import mlflow.entities


def _normalize(url: str) -> str:
    """
    Normalize the URL by removing the protocol and trailing slashes.
    """
    url = url.removeprefix("https://").removeprefix("http://").rstrip("/")

    # Account for a bug in mlflow where the URL is prefixed with "https//"
    url = url.removeprefix("https//")
    return url


class WorkspaceUrlResolver:
    """
    A class to resolve URLs for different entities in a workspace.
    """

    def __init__(self, workspace_url):
        """
        :param instance_name: Databricks workspace instance name (e.g. e2-dogfood.staging.cloud.databricks.com)
        :param workspace_id: ID of this workspace
        """
        self._workspace_url = _normalize(workspace_url)

    def get_full_url(self, path):
        return f"https://{self._workspace_url}/{path.lstrip('/')}/"

    def resolve_url_for_mlflow_run(self, info: mlflow.entities.RunInfo) -> str:
        """Resolve the URL for a MLflow run."""
        path = f"ml/experiments/{info.experiment_id}/runs/{info.run_id}"
        return self.get_full_url(path)

    def resolve_url_for_mlflow_evaluation_results(self, info: mlflow.entities.RunInfo) -> str:
        """Resolve the URL for the evaluation results UI inside the MLflow run."""
        path = f"ml/experiments/{info.experiment_id}/evaluation-runs?selectedRunUuid={info.run_id}"
        return self.get_full_url(path)

    def resolve_url_for_mlflow_experiment(self, info: mlflow.entities.RunInfo) -> str:
        """Resolve the URL for a MLflow experiment."""
        path = f"ml/experiments/{info.experiment_id}"
        return self.get_full_url(path)

    def resolve_url_for_mlflow_experiment_eval_view(self, info: mlflow.entities.RunInfo) -> str:
        """Resolve the URL for the Evaluation tab of an MLflow experiment."""
        return self.resolve_url_for_mlflow_experiment(info) + "?compareRunsMode=ARTIFACT"
