from typing import Any, Dict, Optional

from databricks.sdk import WorkspaceClient


def call_endpoint(
    *,
    method: str,
    route: str,
    json_body: Optional[Dict[str, Any]] = None,
):
    call_kwargs = {}
    if method.lower() == "get":
        call_kwargs["query"] = json_body
    else:
        call_kwargs["body"] = json_body

    # NOTE: This calls internal Databricks SDK APIs, but MLflow relies on the
    # same ones. See https://github.com/mlflow/mlflow/blob/087e1d56b5690e475571e61b86966d8892eefdf3/mlflow/utils/rest_utils.py#L121-L121
    # TODO: switch to public SDK APIs once available
    client = WorkspaceClient()
    raw_response = client.api_client.do(method=method, path=f"/api/2.0/agents/{route}", raw=True, **call_kwargs)
    return raw_response["contents"]._response.json()


def construct_endpoint_url(workspace_url, endpoint_name, workspace_id):
    if endpoint_name is None or workspace_url is None:
        return None
    base_url = f"{workspace_url}/ml/endpoints/{endpoint_name}/"
    if workspace_id is not None:
        base_url += f"?o={workspace_id}"
    return base_url
