from databricks import version
from databricks.rag_eval.clients import databricks_api_client
from databricks.rag_eval.clients.databricks_api_client import raise_for_status
from databricks.rag_eval.utils import request_utils

CLIENT_VERSION_HEADER = "databricks-agents-sdk-version"


def _get_default_headers() -> dict[str, str]:
    """
    Constructs the default request headers.
    """
    headers = {
        CLIENT_VERSION_HEADER: version.VERSION,
    }
    return request_utils.add_traffic_id_header(headers)


class MLFlowClient(databricks_api_client.DatabricksAPIClient):
    """
    Client to interact with the mlflow service.
    """

    def __init__(self):
        super().__init__(version="2.0")

    def link_traces_to_run(
        self,
        *,
        run_id: str,
        trace_ids: list[str],
    ) -> None:
        """
        Link traces to a run.
        """
        request_json = {
            "run_id": run_id,
            "trace_ids": trace_ids,
        }
        with self.get_default_request_session(headers=_get_default_headers()) as session:
            resp = session.post(
                url=self.get_method_url("/mlflow/traces/link-to-run"),
                json=request_json,
            )

        raise_for_status(resp)
