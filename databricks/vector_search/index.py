import json
import time
import datetime
import math
import deprecation
from typing import Optional, List, Dict, Any, Union
from databricks.vector_search.utils import OAuthTokenUtils
from databricks.vector_search.utils import RequestUtils
from databricks.vector_search.utils import UrlUtils
from databricks.vector_search.utils import CredentialStrategy
from databricks.vector_search.reranker import DatabricksReranker, Reranker
from mlflow.utils import databricks_utils
from databricks.vector_search.utils import (
    authentication_warning,
    get_model_serving_invoker_credentials,
)


class VectorSearchIndex:
    """
    VectorSearchIndex is a helper class that represents a Vector Search Index.

    Those who wish to use this class should not instantiate it directly, but rather use the VectorSearchClient class.
    """

    def __init__(
        self,
        workspace_url: str,
        index_url: str,
        name: str,
        endpoint_name: str,
        mlserving_endpoint_name: Optional[str] = None,
        personal_access_token: Optional[str] = None,
        service_principal_client_id: Optional[str] = None,
        service_principal_client_secret: Optional[str] = None,
        azure_tenant_id: Optional[str] = None,
        azure_login_id: Optional[str] = None,
        # whether or not credentials were explicitly passed in by user in client or inferred by client
        # via mlflow utilities. If passed in by user, continue to use user credentials. If not, can
        # attempt automatic auth refresh for model serving.
        use_user_passed_credentials: bool = False,
        credential_strategy: Optional[CredentialStrategy] = None,
        get_reranker_url_callable: Optional[callable] = None,
        mlserving_endpoint_name_for_query: Optional[str] = None,
        total_retries: int = 3,
        backoff_factor: float = 1,
        backoff_jitter: float = 0.2,
    ):
        """
        Initialize a VectorSearchIndex instance.

        :param str workspace_url: The URL of the Databricks workspace.
        :param str index_url: The direct URL to the vector search index endpoint.
        :param str name: The name of the vector search index.
        :param str endpoint_name: The name of the vector search endpoint.
        :param str mlserving_endpoint_name: The name of the model serving endpoint used for embedding generation during ingestion.
        :param str personal_access_token: Personal access token for authentication.
        :param str service_principal_client_id: Service principal client ID for authentication.
        :param str service_principal_client_secret: Service principal client secret for authentication.
        :param str azure_tenant_id: Azure tenant ID for Azure-based authentication.
        :param str azure_login_id: Azure login ID (Databricks Azure Application ID) for authentication.
        :param bool use_user_passed_credentials: Whether credentials were explicitly provided by the user (True) or inferred automatically (False).
        :param CredentialStrategy credential_strategy: The credential strategy to use for authentication.
        :param callable get_reranker_url_callable: A callable function to retrieve the reranker-compatible index URL when needed.
        :param str mlserving_endpoint_name_for_query: The name of the model serving endpoint to use for queries (if different from ingestion endpoint).
        :param int total_retries: Total number of retries for requests. Defaults to 3.
        :param float backoff_factor: Backoff factor for retry delays. Defaults to 1.
        :param float backoff_jitter: Random jitter proportion (0-1) to add to backoff delays. Defaults to 0.2.
        """
        self.workspace_url = workspace_url
        self.name = name
        self.endpoint_name = endpoint_name
        self.personal_access_token = personal_access_token
        self.service_principal_client_id = service_principal_client_id
        self.service_principal_client_secret = service_principal_client_secret
        self.index_url = _get_index_url(
            index_url,
            self.workspace_url,
            self.name,
            self.personal_access_token,
            self.service_principal_client_id,
            self.service_principal_client_secret,
        )
        self._index_url_ensure_reranker_compatible = None
        self._get_reranker_url_callable = get_reranker_url_callable
        self.azure_tenant_id = azure_tenant_id
        self.azure_login_id = azure_login_id
        self._control_plane_oauth_token = None
        self._control_plane_oauth_token_expiry_ts = None
        self._read_oauth_token = None
        self._read_oauth_token_expiry_ts = None
        self._write_oauth_token = None
        self._write_oauth_token_expiry_ts = None
        self._use_user_passed_credentials = use_user_passed_credentials
        # Initialize `mlserving_endpoint_id` as `_get_token_for_request` (a dependency of `_get_mlserving_endpoint_id`)
        # may check the nullability of `mlserving_endpoint_id`
        self.mlserving_endpoint_id = self._get_mlserving_endpoint_id(
            mlserving_endpoint_name
        )
        self.mlserving_endpoint_id_for_query = self._get_mlserving_endpoint_id(
            mlserving_endpoint_name_for_query
        )
        self.credential_strategy = credential_strategy
        self._warned_on_deprecated_columns_to_rerank = False
        self.total_retries = total_retries
        self.backoff_factor = backoff_factor
        self.backoff_jitter = backoff_jitter

    def _get_mlserving_endpoint_id(self, mlserving_endpoint_name):
        if mlserving_endpoint_name is None:
            return None
        resp = RequestUtils.issue_request(
            url=f"{self.workspace_url}/api/2.0/serving-endpoints/{mlserving_endpoint_name}",
            method="GET",
            token=self._get_token_for_request(control_plane=True),
        )
        if resp.get("route_optimized", False):
            return resp["id"]
        else:
            return None

    def _get_token_for_request(self, write=False, control_plane=False, index_url=None):
        try:
            # automatically refresh auth if not passed in by user and in model serving environment
            if (
                not self._use_user_passed_credentials
                and databricks_utils.is_in_databricks_model_serving_environment()
            ):
                if (
                    self.credential_strategy
                    == CredentialStrategy.MODEL_SERVING_USER_CREDENTIALS
                ):
                    _, token = get_model_serving_invoker_credentials()
                    return token
                else:
                    return databricks_utils.get_databricks_host_creds().token
        except Exception as e:
            # Faile to read credentials from model serving environment failed and we will default
            # to cached vector search token
            pass

        if self.personal_access_token:  # PAT flow
            return self.personal_access_token
        if (index_url is None and self.workspace_url in self.index_url) or (index_url is not None and self.workspace_url in index_url):
            control_plane = True
        if (
            control_plane
            and self._control_plane_oauth_token
            and self._control_plane_oauth_token_expiry_ts
            and self._control_plane_oauth_token_expiry_ts - 100 > time.time()
        ):
            return self._control_plane_oauth_token
        if (
            write
            and not control_plane
            and self._write_oauth_token
            and self._write_oauth_token_expiry_ts
            and self._write_oauth_token_expiry_ts - 100 > time.time()
        ):
            return self._write_oauth_token
        if (
            not write
            and not control_plane
            and self._read_oauth_token
            and self._read_oauth_token_expiry_ts
            and self._read_oauth_token_expiry_ts - 100 > time.time()
        ):
            return self._read_oauth_token
        if self.service_principal_client_id and self.service_principal_client_secret:
            if control_plane:
                authorization_details = []
            elif self.mlserving_endpoint_id:
                authorization_details = json.dumps(
                    [
                        {
                            "type": "unity_catalog_permission",
                            "securable_type": "table",
                            "securable_object_name": self.name,
                            "operation": (
                                "WriteVectorIndex" if write else "ReadVectorIndex"
                            ),
                        },
                        {
                            "type": "workspace_permission",
                            "object_type": "serving-endpoints",
                            "object_path": "/serving-endpoints/"
                            + self.mlserving_endpoint_id,
                            "actions": ["query_inference_endpoint"],
                        },
                    ]
                )
            else:
                authorization_details = json.dumps(
                    [
                        {
                            "type": "unity_catalog_permission",
                            "securable_type": "table",
                            "securable_object_name": self.name,
                            "operation": (
                                "WriteVectorIndex" if write else "ReadVectorIndex"
                            ),
                        }
                    ]
                )
            oauth_token_data = (
                OAuthTokenUtils.get_oauth_token(
                    workspace_url=self.workspace_url,
                    service_principal_client_id=self.service_principal_client_id,
                    service_principal_client_secret=self.service_principal_client_secret,
                    authorization_details=authorization_details,
                )
                if not self.azure_tenant_id
                else OAuthTokenUtils.get_azure_oauth_token(
                    workspace_url=self.workspace_url,
                    service_principal_client_id=self.service_principal_client_id,
                    service_principal_client_secret=self.service_principal_client_secret,
                    authorization_details=authorization_details,
                    azure_tenant_id=self.azure_tenant_id,
                    azure_login_id=self.azure_login_id,
                )
            )
            if control_plane:
                self._control_plane_oauth_token = oauth_token_data["access_token"]
                self._control_plane_oauth_token_expiry_ts = time.time() + float(
                    oauth_token_data["expires_in"]
                )
                return self._control_plane_oauth_token
            if write:
                self._write_oauth_token = oauth_token_data["access_token"]
                self._write_oauth_token_expiry_ts = time.time() + float(
                    oauth_token_data["expires_in"]
                )
                return self._write_oauth_token
            self._read_oauth_token = oauth_token_data["access_token"]
            self._read_oauth_token_expiry_ts = time.time() + float(
                oauth_token_data["expires_in"]
            )
            return self._read_oauth_token
        raise Exception("You must specify service principal or PAT token")

    def upsert(self, inputs):
        """
        Upsert data into the index.

        :param inputs: List of dictionaries to upsert into the index.
        """
        assert type(inputs) == list, "inputs must be of type: List of dictionaries"
        assert all(
            type(i) == dict for i in inputs
        ), "inputs must be of type: List of dicts"
        upsert_payload = {"inputs_json": json.dumps(inputs)}
        return RequestUtils.issue_request(
            url=f"{self.index_url}/upsert-data",
            token=self._get_token_for_request(write=True),
            method="POST",
            json=upsert_payload,
        )

    def delete(self, primary_keys):
        """
        Delete data from the index.

        :param primary_keys: List of primary keys to delete from the index.
        """
        assert type(primary_keys) == list, "inputs must be of type: List"
        delete_payload = {"primary_keys": primary_keys}
        return RequestUtils.issue_request(
            url=f"{self.index_url}/delete-data",
            token=self._get_token_for_request(write=True),
            method="DELETE",
            json=delete_payload,
        )

    def describe(self):
        """
        Describe the index. This returns metadata about the index.
        """
        return RequestUtils.issue_request(
            url=f"{self.workspace_url}/api/2.0/vector-search/indexes/{self.name}",
            token=self._get_token_for_request(control_plane=True),
            method="GET",
        )

    def sync(self):
        """
        Sync the index. This is used to sync the index with the source delta table.
        This only works with managed delta sync index with pipeline type="TRIGGERED".
        """
        return RequestUtils.issue_request(
            url=f"{self.workspace_url}/api/2.0/vector-search/indexes/{self.name}/sync",
            token=self._get_token_for_request(control_plane=True),
            method="POST",
        )

    def similarity_search(
        self,
        columns: List[str],
        query_text: Optional[str] = None,
        query_vector: Optional[List[float]] = None,
        filters: Optional[Union[str, Dict[str, Any]]] = None,
        num_results: int = 5,
        debug_level: int = 0,
        score_threshold: Optional[float] = None,
        query_type: Optional[str] = None,
        columns_to_rerank: Optional[List[str]] = None,
        disable_notice: bool = False,
        reranker: Optional[Reranker] = None,
        total_retries: int = 3,
        backoff_factor: float = 1,
        backoff_jitter: float = 0.2,
        *,
        _query_experiment_config: Optional[Dict[str, Any]] = None,
    ):
        """similarity_search(columns: List[str], query_text: Optional[str] = None, query_vector: Optional[List[float]] = None, filters: Optional[Union[str, Dict[str, Any]]] = None, num_results: int = 5, debug_level: int = 0, score_threshold: Optional[float] = None, query_type: Optional[str] = None, columns_to_rerank: Optional[List[str]] = None, disable_notice: bool = False, reranker: Optional[Reranker] = None, total_retries: int = 3, backoff_factor: float = 1, backoff_jitter: float = 0.2)

        Perform a similarity search on the index. This returns the top K results that are most similar to the query.

        :param columns: List of column names to return in the results.
        :param query_text: Query text to search for.
        :param query_vector: Query vector to search for.
        :param filters: Filters to apply to the query.
        :param num_results: Number of results to return.
        :param debug_level: Debug level to use for the query.
        :param score_threshold: Score threshold to use for the query. If reranker is used, the score threshold is applied before reranking.
        :param query_type: Query type of this query. Choices are "ANN" and "HYBRID".
        :param columns_to_rerank: (Deprecated) List of column names to use for reranking the results.
            Use the ``reranker`` parameter instead.
        :param disable_notice: Whether to disable the notice message.
        :param reranker: Optional reranker to apply on the top results. Pass an instance of
            :class:`databricks.vector_search.reranker.DatabricksReranker` with
            ``columns_to_rerank=[...]``. The reranker reorders the initial results using
            the specified text columns.
        :type reranker: Optional[:class:`databricks.vector_search.reranker.Reranker`]
        :param total_retries: Total number of retries for the request. Set to 0 to disable retries.
        :param backoff_factor: Backoff factor to apply between retry attempts. The delay between retries
            is calculated as {backoff_factor} * (2 ** (retry_count - 1)) seconds. For example, with
            backoff_factor=1, delays are 0.5s, 1s, 2s, 4s, etc.
        :param backoff_jitter: Random jitter to add to backoff delays to avoid thundering herd problem.
            Value between 0 and 1 representing the proportion of jitter to apply.

        Example:
            Use the Databricks reranker to improve the ordering of hybrid search results:

            .. code-block:: python

                from databricks.vector_search.reranker import DatabricksReranker

                results = index.similarity_search(
                    query_text=\"How to create a Vector Search index\",
                    columns=[\"id\", \"text\", \"parent_doc_summary\", \"date\"],
                    # The final number of results to return. The reranker will automatically overfetch 50 documents and rerank them.
                    num_results=10,
                    query_type=\"hybrid\",
                    # Needed for debug info to get any warnings and time to rerank the results.
                    debug_level=1,
                    # The text reranked will be concatenated and if it is longer than 2000 characters, it will be truncated.
                    # Include shorter, important columns first.
                    reranker=DatabricksReranker(columns_to_rerank=[\"parent_doc_summary\", \"text\", \"other_column\"]),
                )
                # Check if reranking was successful and how much additional time it took to rerank the results.
                if "warnings" in results['debug_info']:
                    print(results['debug_info']['warnings'])
                else:
                    print(f"Reranking was successful and took {results['debug_info']['reranker_time']}ms")
        """
        authentication_warning(
            not self._use_user_passed_credentials,
            self.personal_access_token,
            disable_notice,
        )
        if columns_to_rerank and reranker is not None:
            raise ValueError(
                "The arguments `columns_to_rerank` and `reranker` cannot both be provided."
            )
        if columns_to_rerank:
            if not self._warned_on_deprecated_columns_to_rerank:
                print(
                    "[NOTICE] The argument `columns_to_rerank` is deprecated. Use the `reranker` argument instead: `from databricks.vector_search.reranker import DatabricksReranker; index.similarity_search(..., reranker=DatabricksReranker(columns_to_rerank=[...]))`."
                )
            self._warned_on_deprecated_columns_to_rerank = True
        if reranker is not None:
            # Move everything to `columns_to_rerank` since it works in both old and new deployment.
            # TODO: Move everything to `reranker` once the new deployment is fully rolled out.
            columns_to_rerank = reranker.columns_to_rerank

        if isinstance(filters, str):
            filter_string = filters
            filters_json = None
        else:
            filter_string = None
            filters_json = json.dumps(filters) if filters else None
        json_data = {
            "num_results": num_results,
            "columns": columns,
            "filters_json": filters_json,
            "filter_string": filter_string,
            "debug_level": debug_level,
        }
        if query_text:
            json_data["query"] = query_text
            json_data["query_text"] = query_text
        if query_vector:
            json_data["query_vector"] = query_vector
        if score_threshold:
            json_data["score_threshold"] = score_threshold
        if query_type:
            json_data["query_type"] = query_type
        if columns_to_rerank:
            json_data["columns_to_rerank"] = columns_to_rerank
            if self._index_url_ensure_reranker_compatible is None:
                # Use the callable to get the reranker-compatible URL
                if self._get_reranker_url_callable:
                    index_url_raw = self._get_reranker_url_callable()
                    self._index_url_ensure_reranker_compatible = _get_index_url(
                        index_url_raw,
                        self.workspace_url,
                        self.name,
                        self.personal_access_token,
                        self.service_principal_client_id,
                        self.service_principal_client_secret,
                    )
                else:
                    self._index_url_ensure_reranker_compatible = self.index_url
            query_url = self._index_url_ensure_reranker_compatible
        else:
            query_url = self.index_url
        if _query_experiment_config:
            json_data["query_experiment_config"] = _query_experiment_config

        response = RequestUtils.issue_request(
            url=f"{query_url}/query",
            token=self._get_token_for_request(index_url=query_url),
            method="GET",
            json=json_data,
            total_retries=total_retries,
            backoff_factor=backoff_factor,
            backoff_jitter=backoff_jitter,
        )

        out_put = response
        while response["next_page_token"]:
            response = self.__get_next_page(query_url, response["next_page_token"])
            out_put["result"]["row_count"] += response["result"]["row_count"]
            out_put["result"]["data_array"] += response["result"]["data_array"]

        out_put.pop("next_page_token", None)
        return out_put

    def wait_until_ready(
        self,
        verbose=False,
        timeout=datetime.timedelta(hours=24),
        wait_for_updates=False,
    ):
        """
        Wait for the index to be online.

        :param bool verbose: Whether to print status messages.
        :param datetime.timedelta timeout: The time allowed until we timeout with an Exception.
        :param bool wait_for_updates: If true, the index will also wait for any updates to be completed.
        """

        def get_index_state():
            return self.describe()["status"]["detailed_state"]

        def is_index_state_ready(index_state):
            if "ONLINE" not in index_state:
                return False
            if not wait_for_updates:
                # It is enough to wait for any online state.
                return True
            # Now check if current online state is an update state.
            return index_state in [
                "ONLINE",
                "ONLINE_NO_PENDING_UPDATE",
                "ONLINE_DIRECT_ACCESS",
            ]

        start_time = datetime.datetime.now()
        sleep_time_seconds = 30
        # Online states all contain `ONLINE`.
        # Provisioning states all contain `PROVISIONING`
        # Offline states all contain `OFFLINE`.
        index_state = get_index_state()
        while (
            not is_index_state_ready(index_state)
            and datetime.datetime.now() - start_time < timeout
        ):
            if "OFFLINE" in index_state:
                raise Exception(f"Index {self.name} is offline")
            if verbose:
                running_time = int(
                    math.floor((datetime.datetime.now() - start_time).total_seconds())
                )
                print(
                    f"Index {self.name} is in state {index_state}. Time: {running_time}s."
                )
            time.sleep(sleep_time_seconds)
            index_state = get_index_state()
        if verbose:
            print(f"Index {self.name} is in state {index_state}.")
        if not is_index_state_ready(index_state):
            raise Exception(
                f"Index {self.name} did not become online within timeout of {timeout.total_seconds()}s."
            )

    def scan(self, num_results=10, last_primary_key=None):
        """
        Given all the data in the index sorted by primary key, this returns the next
        `num_results` data after the primary key specified by `last_primary_key`.
        If last_primary_key is None , it returns the first `num_results`.

        Please note if there's ongoing updates to the index, the scan results may not be consistent.

        :param num_results: Number of results to return.
        :param last_primary_key: last primary key from previous pagination, it will be used as the exclusive starting primary key.
        """
        json_data = {
            "num_results": num_results,
            "endpoint_name": self.endpoint_name,
        }
        if last_primary_key:
            json_data["last_primary_key"] = last_primary_key

        url = self.index_url + "/scan"

        return RequestUtils.issue_request(
            url=url, token=self._get_token_for_request(), method="GET", json=json_data
        )

    @deprecation.deprecated(
        deprecated_in="0.36",
        removed_in="0.37",
        current_version="0.36",
        details="Use the scan function instead",
    )
    def scan_index(self, num_results=10, last_primary_key=None):
        return self.scan(num_results, last_primary_key)

    def __get_next_page(self, index_url, page_token):
        """
        Get the next page of results from a page token.
        """
        json_data = {
            "page_token": page_token,
            "endpoint_name": self.endpoint_name,
        }
        url = index_url + "/query-next-page"

        return RequestUtils.issue_request(
            url=url, token=self._get_token_for_request(), method="GET", json=json_data
        )


def _get_index_url(
    index_url,
    workspace_url,
    index_name,
    personal_access_token,
    service_principal_client_id,
    service_principal_client_secret,
):
    cp_url = workspace_url + f"/api/2.0/vector-search/indexes/{index_name}"
    if personal_access_token and not (
        service_principal_client_id and service_principal_client_secret
    ):
        return cp_url
    elif index_url:
        return UrlUtils.add_https_if_missing(index_url)
    else:
        # Fallback to CP
        return cp_url
