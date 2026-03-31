import time
import json
import datetime
import math
from typing import Optional, Dict, Any
from databricks.vector_search.exceptions import InvalidInputException, NotFound
from databricks.vector_search.utils import OAuthTokenUtils
from databricks.vector_search.utils import RequestUtils
from databricks.vector_search.index import VectorSearchIndex
from mlflow.utils import databricks_utils
from databricks.vector_search.utils import CredentialStrategy
from databricks.vector_search.utils import (
    authentication_warning,
    get_model_serving_invoker_credentials,
    is_in_model_serving_environment,
)


MIN_QPS_RESET_TO_DEFAULT = -1


class VectorSearchClient:
    """
    A client for interacting with the Vector Search service.

    This client provides methods for managing endpoints and indexes in the Vector Search service.
    """

    def __init__(
        self,
        workspace_url=None,
        personal_access_token=None,
        service_principal_client_id=None,
        service_principal_client_secret=None,
        azure_tenant_id=None,
        azure_login_id="2ff814a6-3304-4ab8-85cb-cd0e6f879c1d",  # Databricks Azure Application ID in AZURE_PUBLIC environment
        disable_notice=False,
        credential_strategy=None,
    ):
        """
        Initialize the VectorSearchClient.

        :param str workspace_url: The URL of the Databricks workspace (e.g., "https://your-workspace.databricks.com").
                                  Optional if running inside a Databricks notebook (auto-detected).
        :param str personal_access_token: Personal access token for authentication. Used with workspace_url for PAT auth.
                                         Optional if running inside a Databricks notebook (auto-detected).
        :param str service_principal_client_id: OAuth client ID of the service principal. Required for service principal auth.
        :param str service_principal_client_secret: OAuth client secret of the service principal. Required for service principal auth.
        :param str azure_tenant_id: Azure tenant ID for Azure-based authentication. Required only for Azure service principal auth.
        :param str azure_login_id: Azure login ID (Databricks Azure Application ID) for authentication.
                                   Defaults to "2ff814a6-3304-4ab8-85cb-cd0e6f879c1d" (AZURE_PUBLIC).
                                   See all login IDs: https://github.com/databricks/databricks-sdk-py/blob/main/databricks/sdk/environments.py
        :param bool disable_notice: Whether to disable authentication notice messages. Default is False.
        :param CredentialStrategy credential_strategy: Credential strategy for specialized authentication scenarios.
                                                       Use CredentialStrategy.MODEL_SERVING_USER_CREDENTIALS to authenticate
                                                       with the calling user's credentials in Model Serving environments.

        .. note::
           **Authentication Methods**

           The client supports multiple authentication methods. Choose one based on your environment:

           **1. Auto-detection (Recommended for Databricks Notebooks)**
               When no credentials are provided, the client automatically detects credentials from the
               Databricks notebook environment. This is the simplest method when running inside Databricks.

               Example::

                   from databricks.vector_search.client import VectorSearchClient

                   # Credentials are automatically detected from notebook context
                   client = VectorSearchClient()

           **2. Personal Access Token (PAT)**
               Use a personal access token for user-based authentication. Requires both workspace_url
               and personal_access_token.

               Example::

                   from databricks.vector_search.client import VectorSearchClient

                   client = VectorSearchClient(
                       workspace_url="https://your-workspace.databricks.com",
                       personal_access_token="dapi..."
                   )

           **3. Service Principal (OAuth M2M) - AWS & GCP**
               Use OAuth Machine-to-Machine authentication with a service principal for automated workflows.
               Requires workspace_url, service_principal_client_id, and service_principal_client_secret.

               Example::

                   from databricks.vector_search.client import VectorSearchClient

                   client = VectorSearchClient(
                       workspace_url="https://your-workspace.databricks.com",
                       service_principal_client_id="your-client-id",
                       service_principal_client_secret="your-client-secret"
                   )

           **4. Azure Service Principal**
               For Azure Databricks, use Azure AD authentication with additional azure_tenant_id.

               Example::

                   from databricks.vector_search.client import VectorSearchClient

                   client = VectorSearchClient(
                       workspace_url="https://adb-1234567890123456.7.azuredatabricks.net",
                       service_principal_client_id="your-azure-app-id",
                       service_principal_client_secret="your-azure-app-secret",
                       azure_tenant_id="your-azure-tenant-id"
                   )

           **5. Model Serving with User Credentials**
               When calling from within Databricks Model Serving, use the invoking user's credentials
               for fine-grained access control.

               Example::

                   from databricks.vector_search.client import VectorSearchClient
                   from databricks.vector_search.utils import CredentialStrategy

                   client = VectorSearchClient(
                       credential_strategy=CredentialStrategy.MODEL_SERVING_USER_CREDENTIALS
                   )

           **Additional Notes:**

           - Service principal authentication requires workspace_url to be explicitly provided
           - Personal access token authentication also requires workspace_url
           - When using auto-detection in notebooks, credentials are inferred from the execution context
           - Azure authentication requires both azure_tenant_id and standard service principal credentials
        """
        self.workspace_url = workspace_url
        self.personal_access_token = personal_access_token
        self.service_principal_client_id = service_principal_client_id
        self.service_principal_client_secret = service_principal_client_secret
        self.azure_tenant_id = azure_tenant_id
        self.azure_login_id = azure_login_id
        self._is_notebook_pat = False
        self._control_plane_oauth_token = None
        self._control_plane_oauth_token_expiry_ts = None
        self.credential_strategy = credential_strategy

        # whether or not credentials are explicitly passed in by user in client or inferred by client
        # via mlflow utilities. If passed in by user, continue to use user credentials in index object.
        # If not, can attempt automatic auth refresh for model serving.
        self._using_user_passed_credentials = bool(
            (self.service_principal_client_id and self.service_principal_client_secret) or self.personal_access_token
        )

        if not self._using_user_passed_credentials:
            if (
                self.credential_strategy == CredentialStrategy.MODEL_SERVING_USER_CREDENTIALS
                and is_in_model_serving_environment()
            ):
                host, token = get_model_serving_invoker_credentials()
            else:
                host_creds = databricks_utils.get_databricks_host_creds()
                host, token = host_creds.host, host_creds.token

            self.workspace_url = self.workspace_url or host

            if self.personal_access_token is None:
                self._is_notebook_pat = True
                self.personal_access_token = token

        self.validate(disable_notice=disable_notice)

    def validate(self, disable_notice=False):
        """
        Validate the client configuration.

        :param bool disable_notice: Whether to disable the authentication notice message.
        """
        if not (
            self.personal_access_token or (self.service_principal_client_id and self.service_principal_client_secret)
        ):
            raise InvalidInputException(
                "Please specify either personal access token or service principal client ID and secret."
            )

        # Validate user-passed credentials require workspace_url
        if self._using_user_passed_credentials and not self.workspace_url:
            raise InvalidInputException(
                "Workspace URL is required when Personal Access Token or Service Principal credentials are passed in"
            )

        authentication_warning(self._is_notebook_pat, self.personal_access_token, disable_notice)

    def _get_token_for_request(self):
        if not self._using_user_passed_credentials and is_in_model_serving_environment():
            if self.credential_strategy == CredentialStrategy.MODEL_SERVING_USER_CREDENTIALS:
                _, token = get_model_serving_invoker_credentials()
                return token
            else:
                return databricks_utils.get_databricks_host_creds().token
        if self.personal_access_token:
            return self.personal_access_token
        if (
            self._control_plane_oauth_token
            and self._control_plane_oauth_token_expiry_ts
            and self._control_plane_oauth_token_expiry_ts - 100 > time.time()
        ):
            return self._control_plane_oauth_token
        if self.service_principal_client_id and self.service_principal_client_secret:
            authorization_details = []
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
            self._control_plane_oauth_token = oauth_token_data["access_token"]
            self._control_plane_oauth_token_expiry_ts = time.time() + int(oauth_token_data["expires_in"])
            return self._control_plane_oauth_token
        raise Exception("You must specify service principal or PAT token")

    def create_endpoint(
        self,
        name: str,
        endpoint_type: str = "STANDARD",
        budget_policy_id: Optional[str] = None,
        usage_policy_id: Optional[str] = None,
        min_qps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Create an endpoint.

        :param str name: The name of the endpoint.
        :param str endpoint_type: The type of the endpoint. Must be STANDARD or STORAGE_OPTIMIZED.
        :param str budget_policy_id: The id of the budget policy to assign to the endpoint (deprecated, use usage_policy_id).
        :param str usage_policy_id: The id of the usage policy to assign to the endpoint.
        :param int min_qps: Minimum queries per second for the endpoint. Must be a positive integer.
                            Capacity is automatically adjusted at index creation/sync time based on
                            this throughput requirement. Optional.

        .. note::
           **BETA FEATURE**: The min_qps parameter is in beta and subject to change.
           It enables automatic capacity scaling based on desired throughput.
           Currently only supported for STANDARD endpoints.

        .. warning::
           Only STANDARD endpoints support min_qps. Storage Optimized endpoints will
           return an error if min_qps is specified.

        **Example**::

            from databricks.vector_search.client import VectorSearchClient

            client = VectorSearchClient()

            # Create endpoint with min_qps
            endpoint = client.create_endpoint(
                name="high-throughput-endpoint",
                endpoint_type="STANDARD",
                min_qps=1000
            )

            # Check the scaling info
            scaling_info = endpoint.get("endpoint", {}).get("scaling_info", {})
            print(f"Requested min QPS: {scaling_info.get('requested_min_qps')}")
        """
        # Validate mutual exclusivity
        if budget_policy_id and usage_policy_id:
            raise InvalidInputException(
                "Cannot specify both budget_policy_id and usage_policy_id. Please use only one."
            )

        # Validate min_qps
        if min_qps is not None:
            if not isinstance(min_qps, int):
                raise InvalidInputException("min_qps must be an integer.")
            if min_qps <= 0:
                raise InvalidInputException("min_qps must be a positive integer.")

        json_data = {"name": name, "endpoint_type": endpoint_type}

        # Use usage_policy_id if provided, otherwise fall back to budget_policy_id
        policy_id = usage_policy_id or budget_policy_id
        if policy_id:
            # For now, always send as budget_policy_id to maintain backend compatibility
            # When backend is updated, we can conditionally use the appropriate field
            json_data["budget_policy_id"] = policy_id

        # Add min_qps if provided
        if min_qps is not None:
            json_data["min_qps"] = min_qps

        return RequestUtils.issue_request(
            url=f"{self.workspace_url}/api/2.0/vector-search/endpoints",
            token=self._get_token_for_request(),
            method="POST",
            json=json_data,
        )

    def get_endpoint(self, name):
        """
        Get an endpoint.

        :param str name: The name of the endpoint.
        :raises NotFound: If the endpoint does not exist.
        :raises PermissionDenied: If user lacks permission to view the endpoint.
        """
        return RequestUtils.issue_request(
            url=f"{self.workspace_url}/api/2.0/vector-search/endpoints/{name}",
            token=self._get_token_for_request(),
            method="GET",
        )

    def endpoint_exists(self, name):
        """
        Check if an endpoint exists.

        This method provides a cleaner alternative to catching NotFound exceptions
        when checking for resource existence.

        :param str name: The name of the endpoint.
        :return: True if the endpoint exists, False otherwise.
        :rtype: bool

        **Example**::

            from databricks.vector_search.client import VectorSearchClient

            client = VectorSearchClient()
            if client.endpoint_exists("my-endpoint"):
                endpoint = client.get_endpoint("my-endpoint")
            else:
                endpoint = client.create_endpoint("my-endpoint")
        """
        try:
            self.get_endpoint(name)
            return True
        except NotFound:
            return False

    def list_endpoints(self):
        """
        List all endpoints.
        """
        return RequestUtils.issue_request(
            url=f"{self.workspace_url}/api/2.0/vector-search/endpoints",
            token=self._get_token_for_request(),
            method="GET",
        )

    def delete_endpoint(self, name):
        """
        Delete an endpoint.

        :param str name: The name of the endpoint.
        """
        return RequestUtils.issue_request(
            url=f"{self.workspace_url}/api/2.0/vector-search/endpoints/{name}",
            token=self._get_token_for_request(),
            method="DELETE",
        )

    def create_endpoint_and_wait(
        self,
        name: str,
        endpoint_type: str = "STANDARD",
        budget_policy_id: Optional[str] = None,
        usage_policy_id: Optional[str] = None,
        min_qps: Optional[int] = None,
        verbose: bool = False,
        timeout: datetime.timedelta = datetime.timedelta(minutes=60),
    ) -> None:
        """
        Create an endpoint and wait for it to be online.

        :param str name: The name of the endpoint.
        :param str endpoint_type: The type of the endpoint. Must be STANDARD or STORAGE_OPTIMIZED.
        :param str budget_policy_id: The id of the budget policy to assign to the endpoint (deprecated, use usage_policy_id).
        :param str usage_policy_id: The id of the usage policy to assign to the endpoint.
        :param int min_qps: Minimum queries per second for the endpoint. Must be a positive integer.
                            Capacity is automatically adjusted at index creation/sync time based on
                            this throughput requirement. Optional.
        :param bool verbose: Whether to print status messages.
        :param datetime.timedelta timeout: The time allowed until we timeout with an Exception.

        .. note::
           **BETA FEATURE**: The min_qps parameter is in beta and subject to change.
           It enables automatic capacity scaling based on desired throughput.
           Currently only supported for STANDARD endpoints.

        .. warning::
           Only STANDARD endpoints support min_qps. Storage Optimized endpoints will
           return an error if min_qps is specified.
        """
        self.create_endpoint(name, endpoint_type, budget_policy_id, usage_policy_id, min_qps)
        self.wait_for_endpoint(name, verbose, timeout)

    def wait_for_endpoint(self, name, verbose=False, timeout=datetime.timedelta(minutes=60)):
        """
        Wait for an endpoint to be online.

        :param str name: The name of the endpoint.
        :param bool verbose: Whether to print status messages.
        :param datetime.timedelta timeout: The time allowed until we timeout with an Exception.
        """

        def get_endpoint_state():
            endpoint = self.get_endpoint(name)
            endpoint_state = endpoint["endpoint_status"]["state"]
            return endpoint_state

        start_time = datetime.datetime.now()
        sleep_time_seconds = min(30, timeout.total_seconds())
        # Possible states are "ONLINE", "OFFLINE", and "PROVISIONING".
        endpoint_state = get_endpoint_state()
        while endpoint_state != "ONLINE" and datetime.datetime.now() - start_time < timeout:
            if endpoint_state == "OFFLINE":
                raise Exception(f"Endpoint {name} is OFFLINE.")
            if verbose:
                running_time = int(math.floor((datetime.datetime.now() - start_time).total_seconds()))
                print(f"Endpoint {name} is in state {endpoint_state}. Time: {running_time}s.")
            time.sleep(sleep_time_seconds)
            endpoint_state = get_endpoint_state()
        if endpoint_state == "ONLINE":
            if verbose:
                print(f"Endpoint {name} is ONLINE.")
        else:
            raise Exception(f"Endpoint {name} did not become ONLINE within timeout of {timeout.total_seconds()}s.")

    def update_endpoint_budget_policy(self, name, budget_policy_id=None, usage_policy_id=None):
        """
        Update an endpoint's budget/usage policy.

        :param str name: The name of the endpoint.
        :param str budget_policy_id: The id of the budget policy to assign to the endpoint (deprecated, use usage_policy_id).
        :param str usage_policy_id: The id of the usage policy to assign to the endpoint.
        """
        # Validate mutual exclusivity
        if budget_policy_id and usage_policy_id:
            raise InvalidInputException(
                "Cannot specify both budget_policy_id and usage_policy_id. Please use only one."
            )

        # Use usage_policy_id if provided, otherwise fall back to budget_policy_id
        policy_id = usage_policy_id or budget_policy_id
        if not policy_id:
            raise InvalidInputException("Either budget_policy_id or usage_policy_id must be specified.")

        # For now, always send as budget_policy_id to maintain backend compatibility
        json_data = {"budget_policy_id": policy_id}
        return RequestUtils.issue_request(
            url=f"{self.workspace_url}/api/2.0/vector-search/endpoints/{name}/budget-policy",
            token=self._get_token_for_request(),
            method="PATCH",
            json=json_data,
        )

    def update_endpoint_usage_policy(self, name, usage_policy_id):
        """
        Update an endpoint's usage policy (alias for update_endpoint_budget_policy).

        :param str name: The name of the endpoint.
        :param str usage_policy_id: The id of the usage policy to assign to the endpoint.
        """
        return self.update_endpoint_budget_policy(name, usage_policy_id=usage_policy_id)

    def update_endpoint(self, name: str, min_qps: Optional[int] = None) -> Dict[str, Any]:
        """
        Update an endpoint's configuration.

        :param str name: The name of the endpoint. Required.
        :param int min_qps: Minimum queries per second for the endpoint. Must be a positive integer,
                            or -1 to reset to default (non-high-QPS mode). Use
                            ``MIN_QPS_RESET_TO_DEFAULT`` for clarity. Optional.

        .. note::
           **BETA FEATURE**: The min_qps parameter is in beta and subject to change.
           It enables automatic capacity scaling based on desired throughput.
           Currently only supported for STANDARD endpoints.

        .. warning::
           Only STANDARD endpoints support min_qps. Storage Optimized endpoints will
           return an error if min_qps is specified.

        **Example**::

            from databricks.vector_search.client import VectorSearchClient
            from databricks.vector_search.client import MIN_QPS_RESET_TO_DEFAULT

            client = VectorSearchClient()

            # Update existing endpoint with min_qps
            response = client.update_endpoint(name="my-endpoint", min_qps=500)

            # Reset endpoint to default (non-high-QPS mode)
            response = client.update_endpoint(name="my-endpoint", min_qps=MIN_QPS_RESET_TO_DEFAULT)

            # Check scaling state
            scaling_info = response.get("endpoint", {}).get("scaling_info", {})
            state = scaling_info.get("state")

            if state == "SCALING_CHANGE_APPLIED":
                print("Scaling configuration applied")
            elif state == "SCALING_CHANGE_IN_PROGRESS":
                print("Scaling change in progress, capacity will be adjusted at next index sync")
        """
        # Validate name
        if not name:
            raise InvalidInputException("Endpoint name must be specified.")

        # Validate min_qps: positive integer or -1 (reset to default)
        if min_qps is not None:
            if not isinstance(min_qps, int):
                raise InvalidInputException("min_qps must be an integer.")
            if min_qps == 0 or min_qps < -1:
                raise InvalidInputException(
                    "min_qps must be a positive integer or -1 (to reset to default)."
                )

        # Build request body with only specified fields
        json_data = {}
        if min_qps is not None:
            json_data["min_qps"] = min_qps

        return RequestUtils.issue_request(
            url=f"{self.workspace_url}/api/2.0/vector-search/endpoints/{name}",
            token=self._get_token_for_request(),
            method="PATCH",
            json=json_data,
        )

    def list_indexes(self, name):
        """
        List all indexes for an endpoint.

        :param str name: The name of the endpoint.
        """
        return RequestUtils.issue_request(
            url=f"{self.workspace_url}/api/2.0/vector-search/endpoints/{name}/indexes",
            token=self._get_token_for_request(),
            method="GET",
        )

    def create_delta_sync_index(
        self,
        endpoint_name,
        index_name,
        primary_key,
        source_table_name,
        pipeline_type,
        embedding_dimension=None,
        embedding_vector_column=None,
        embedding_source_column=None,
        embedding_model_endpoint_name=None,
        sync_computed_embeddings=False,
        columns_to_sync=None,
        model_endpoint_name_for_query=None,
        budget_policy_id=None,
        usage_policy_id=None,
    ):
        """
        Create a delta sync index.

        :param str columns_to_sync: The columns that would be synced to the vector index with the primary key and vector column always being synced. If the field is not defined, all columns will be synced.
        :param str endpoint_name: The name of the endpoint.
        :param str index_name: The name of the index.
        :param str primary_key: The primary key of the index.
        :param str source_table_name: The name of the source table.
        :param str pipeline_type: The type of the pipeline. Must be CONTINUOUS or TRIGGERED.
        :param int embedding_dimension: The dimension of the embedding vector.
        :param str embedding_vector_column: The name of the embedding vector column.
        :param str embedding_source_column: The name of the embedding source column.
        :param str embedding_model_endpoint_name: The name of the embedding model endpoint.
        :param bool sync_computed_embeddings: Whether to automatically sync the vector index contents and computed embeddings to a new UC table,
                                             table name will be ${index_name}_writeback_table.
        :param str model_endpoint_name_for_query: When set, queries will use this embedding model instead of the embedding_model_endpoint_name. If unset, queries continue to use embedding_model_endpoint_name.
        :param str budget_policy_id: The budget policy ID to associate with this index for cost tracking and management (deprecated, use usage_policy_id).
        :param str usage_policy_id: The usage policy ID to associate with this index for cost tracking and management.
        """
        # Validate mutual exclusivity
        if budget_policy_id and usage_policy_id:
            raise InvalidInputException(
                "Cannot specify both budget_policy_id and usage_policy_id. Please use only one."
            )
        assert pipeline_type, "Pipeline type cannot be None. Please use CONTINUOUS/TRIGGERED as the pipeline type."
        json_data = {
            "name": index_name,
            "index_type": "DELTA_SYNC",
            "primary_key": primary_key,
            "delta_sync_index_spec": {
                "source_table": source_table_name,
                "pipeline_type": pipeline_type.upper(),
            },
        }
        if embedding_vector_column:
            assert embedding_dimension, "Embedding dimension must be specified if source column is used"
            assert not sync_computed_embeddings, "Sync computed embedding is not supported with embedding vector column"
            json_data["delta_sync_index_spec"]["embedding_vector_columns"] = [
                {
                    "name": embedding_vector_column,
                    "embedding_dimension": embedding_dimension,
                }
            ]
        elif embedding_source_column:
            assert embedding_model_endpoint_name, "You must specify Embedding Model Endpoint"
            embedding_source_config = {
                "name": embedding_source_column,
                "embedding_model_endpoint_name": embedding_model_endpoint_name,
            }
            if model_endpoint_name_for_query:
                embedding_source_config["model_endpoint_name_for_query"] = model_endpoint_name_for_query
            json_data["delta_sync_index_spec"]["embedding_source_columns"] = [embedding_source_config]
            if sync_computed_embeddings:
                json_data["delta_sync_index_spec"]["embedding_writeback_table"] = f"{index_name}_writeback_table"

        if columns_to_sync:
            json_data["delta_sync_index_spec"]["columns_to_sync"] = columns_to_sync

        # Use usage_policy_id if provided, otherwise fall back to budget_policy_id
        policy_id = usage_policy_id or budget_policy_id
        if policy_id:
            # For now, always send as budget_policy_id to maintain backend compatibility
            json_data["delta_sync_index_spec"]["budget_policy_id"] = policy_id

        resp = RequestUtils.issue_request(
            url=f"{self.workspace_url}/api/2.0/vector-search/endpoints/{endpoint_name}/indexes",
            token=self._get_token_for_request(),
            method="POST",
            json=json_data,
        )

        index_url = resp.get("status", {}).get("index_url")
        mlserving_endpoint_name = self._get_mlserving_endpoint_name_from_resp(resp)
        mlserving_endpoint_name_for_query = self._get_mlserving_endpoint_name_for_query_from_resp(resp)

        def get_reranker_url():
            """Callable to fetch reranker-compatible URL when needed."""
            reranker_resp = RequestUtils.issue_request(
                url=self._get_index_url(endpoint_name, resp["name"]),
                token=self._get_token_for_request(),
                method="GET",
                params={"ensure_reranker_compatible": "true"},
            )
            return reranker_resp.get("status", {}).get("index_url")

        return VectorSearchIndex(
            workspace_url=self.workspace_url,
            index_url=index_url,
            personal_access_token=self.personal_access_token,
            service_principal_client_id=self.service_principal_client_id,
            service_principal_client_secret=self.service_principal_client_secret,
            name=resp["name"],
            endpoint_name=endpoint_name,
            mlserving_endpoint_name=mlserving_endpoint_name,
            azure_tenant_id=self.azure_tenant_id,
            azure_login_id=self.azure_login_id,
            use_user_passed_credentials=self._using_user_passed_credentials,
            credential_strategy=self.credential_strategy,
            get_reranker_url_callable=get_reranker_url,
            mlserving_endpoint_name_for_query=mlserving_endpoint_name_for_query,
        )

    def create_delta_sync_index_and_wait(
        self,
        endpoint_name,
        index_name,
        primary_key,
        source_table_name,
        pipeline_type,
        embedding_dimension=None,
        embedding_vector_column=None,
        embedding_source_column=None,
        embedding_model_endpoint_name=None,
        sync_computed_embeddings=False,
        columns_to_sync=None,
        model_endpoint_name_for_query=None,
        budget_policy_id=None,
        usage_policy_id=None,
        verbose=False,
        timeout=datetime.timedelta(hours=24),
    ):
        """
        Create a delta sync index and wait for it to be ready.

        :param str columns_to_sync: The columns that would be synced to the vector index with the primary key and vector column always being synced. If the field is not defined, all columns will be synced.
        :param str endpoint_name: The name of the endpoint.
        :param str index_name: The name of the index.
        :param str primary_key: The primary key of the index.
        :param str source_table_name: The name of the source table.
        :param str pipeline_type: The type of the pipeline. Must be CONTINUOUS or TRIGGERED.
        :param int embedding_dimension: The dimension of the embedding vector.
        :param str embedding_vector_column: The name of the embedding vector column.
        :param str embedding_source_column: The name of the embedding source column.
        :param str embedding_model_endpoint_name: The name of the embedding model endpoint.
        :param bool verbose: Whether to print status messages.
        :param datetime.timedelta timeout: The time allowed until we timeout with an Exception.
        :param bool sync_computed_embeddings: Whether to automatically sync the vector index contents and computed embeddings to a new UC table,
                                             table name will be ${index_name}_writeback_table.
        :param str model_endpoint_name_for_query: The name of the embedding model endpoint to be used for querying, not ingestion.
        :param str budget_policy_id: The budget policy ID to associate with this index for cost tracking and management (deprecated, use usage_policy_id).
        :param str usage_policy_id: The usage policy ID to associate with this index for cost tracking and management.
        """
        index = self.create_delta_sync_index(
            endpoint_name,
            index_name,
            primary_key,
            source_table_name,
            pipeline_type,
            embedding_dimension,
            embedding_vector_column,
            embedding_source_column,
            embedding_model_endpoint_name,
            sync_computed_embeddings,
            columns_to_sync,
            model_endpoint_name_for_query,
            budget_policy_id,
            usage_policy_id,
        )
        index.wait_until_ready(verbose, timeout)
        return index

    def create_direct_access_index(
        self,
        endpoint_name,
        index_name,
        primary_key,
        embedding_dimension,
        embedding_vector_column,
        schema,
        embedding_model_endpoint_name=None,
        budget_policy_id=None,
        usage_policy_id=None,
    ):
        """
        Create a direct access index.

        :param str endpoint_name: The name of the endpoint.
        :param str index_name: The name of the index.
        :param str primary_key: The primary key of the index.
        :param int embedding_dimension: The dimension of the embedding vector.
        :param str embedding_vector_column: The name of the embedding vector column.
        :param dict schema: The schema of the index.
        :param str embedding_model_endpoint_name: The name of the optional embedding model endpoint to use when querying.
        :param str budget_policy_id: The budget policy id to be applied to the index (deprecated, use usage_policy_id).
        :param str usage_policy_id: The usage policy id to be applied to the index.
        """
        # Validate mutual exclusivity
        if budget_policy_id and usage_policy_id:
            raise InvalidInputException(
                "Cannot specify both budget_policy_id and usage_policy_id. Please use only one."
            )
        assert schema, """
            Schema must be present when creating a direct access index.
            Example schema: {"id": "integer", "text": "string", \
                "text_vector": "array<float>", "bool_val": "boolean", \
                    "float_val": "float", "date_val": "date"}"
        """
        json_data = {
            "name": index_name,
            "index_type": "DIRECT_ACCESS",
            "primary_key": primary_key,
            "direct_access_index_spec": {
                "embedding_vector_columns": [
                    {
                        "name": embedding_vector_column,
                        "embedding_dimension": embedding_dimension,
                    }
                ],
                "schema_json": json.dumps(schema),
            },
        }
        if embedding_model_endpoint_name:
            json_data["direct_access_index_spec"]["embedding_source_columns"] = [
                {"embedding_model_endpoint_name": embedding_model_endpoint_name}
            ]

        # Use usage_policy_id if provided, otherwise fall back to budget_policy_id
        policy_id = usage_policy_id or budget_policy_id
        if policy_id:
            raise Exception("Usage policy is not supported for direct access index")
        resp = RequestUtils.issue_request(
            url=f"{self.workspace_url}/api/2.0/vector-search/endpoints/{endpoint_name}/indexes",
            token=self._get_token_for_request(),
            method="POST",
            json=json_data,
        )

        index_url = resp.get("status", {}).get("index_url")
        mlserving_endpoint_name = self._get_mlserving_endpoint_name_from_resp(resp)

        def get_reranker_url():
            """Callable to fetch reranker-compatible URL when needed."""
            reranker_resp = RequestUtils.issue_request(
                url=self._get_index_url(endpoint_name, resp["name"]),
                token=self._get_token_for_request(),
                method="GET",
                params={"ensure_reranker_compatible": "true"},
            )
            return reranker_resp.get("status", {}).get("index_url")

        return VectorSearchIndex(
            workspace_url=self.workspace_url,
            index_url=index_url,
            personal_access_token=self.personal_access_token,
            service_principal_client_id=self.service_principal_client_id,
            service_principal_client_secret=self.service_principal_client_secret,
            name=resp["name"],
            endpoint_name=endpoint_name,
            mlserving_endpoint_name=mlserving_endpoint_name,
            azure_tenant_id=self.azure_tenant_id,
            azure_login_id=self.azure_login_id,
            use_user_passed_credentials=self._using_user_passed_credentials,
            credential_strategy=self.credential_strategy,
            get_reranker_url_callable=get_reranker_url,
        )

    def _get_index_url(self, endpoint_name, index_name):
        if endpoint_name:
            url = f"{self.workspace_url}/api/2.0/vector-search/endpoints/{endpoint_name}/indexes/{index_name}"
        else:
            url = f"{self.workspace_url}/api/2.0/vector-search/indexes/{index_name}"
        return url

    def _get_mlserving_endpoint_name_from_resp(self, resp):
        return (
            resp.get("delta_sync_index_spec", {})
            .get("embedding_source_columns", [{}])[0]
            .get("embedding_model_endpoint_name", None)
        )

    def _get_mlserving_endpoint_name_for_query_from_resp(self, resp):
        return (
            resp.get("delta_sync_index_spec", {})
            .get("embedding_source_columns", [{}])[0]
            .get("model_endpoint_name_for_query", None)
        )

    def get_index(self, endpoint_name=None, index_name=None):
        """
        Get an index.

        :param Option[str] endpoint_name: The optional name of the endpoint.
        :param str index_name: The name of the index.
        :raises NotFound: If the index does not exist.
        :raises PermissionDenied: If user lacks permission to view the index.
        """
        assert index_name, "Index name must be specified"
        resp = RequestUtils.issue_request(
            url=self._get_index_url(endpoint_name, index_name),
            token=self._get_token_for_request(),
            method="GET",
        )
        index_url = resp.get("status", {}).get("index_url")
        response_endpoint_name = resp.get("endpoint_name")
        mlserving_endpoint_name = self._get_mlserving_endpoint_name_from_resp(resp)
        mlserving_endpoint_name_for_query = self._get_mlserving_endpoint_name_for_query_from_resp(resp)

        def get_reranker_url():
            """Callable to fetch reranker-compatible URL when needed."""
            reranker_resp = RequestUtils.issue_request(
                url=self._get_index_url(endpoint_name, index_name),
                token=self._get_token_for_request(),
                method="GET",
                params={"ensure_reranker_compatible": "true"},
            )
            return reranker_resp.get("status", {}).get("index_url")

        return VectorSearchIndex(
            workspace_url=self.workspace_url,
            index_url=index_url,
            get_reranker_url_callable=get_reranker_url,
            personal_access_token=self.personal_access_token,
            service_principal_client_id=self.service_principal_client_id,
            service_principal_client_secret=self.service_principal_client_secret,
            name=index_name,
            endpoint_name=response_endpoint_name,
            mlserving_endpoint_name=mlserving_endpoint_name,
            azure_tenant_id=self.azure_tenant_id,
            azure_login_id=self.azure_login_id,
            use_user_passed_credentials=self._using_user_passed_credentials,
            credential_strategy=self.credential_strategy,
            mlserving_endpoint_name_for_query=mlserving_endpoint_name_for_query,
        )

    def index_exists(self, endpoint_name=None, index_name=None):
        """
        Check if an index exists.

        This method provides a cleaner alternative to catching NotFound exceptions
        when checking for resource existence.

        :param Option[str] endpoint_name: The optional name of the endpoint.
        :param str index_name: The name of the index.
        :return: True if the index exists, False otherwise.
        :rtype: bool

        **Example**::

            from databricks.vector_search.client import VectorSearchClient

            client = VectorSearchClient()
            if client.index_exists(index_name="my-index"):
                index = client.get_index(index_name="my-index")
            else:
                index = client.create_delta_sync_index(...)
        """
        assert index_name, "Index name must be specified"
        try:
            self.get_index(endpoint_name=endpoint_name, index_name=index_name)
            return True
        except NotFound:
            return False

    def delete_index(self, endpoint_name=None, index_name=None):
        """
        Delete an index.

        :param Option[str] endpoint_name: The optional name of the endpoint.
        :param str index_name: The name of the index.
        """
        assert index_name, "Index name must be specified"
        return RequestUtils.issue_request(
            url=self._get_index_url(endpoint_name, index_name),
            token=self._get_token_for_request(),
            method="DELETE",
        )

    def update_index_budget_policy(self, index_name, budget_policy_id=None, usage_policy_id=None):
        """
        Update the budget/usage policy of an index.

        :param str index_name: The name of the index.
        :param str budget_policy_id: The budget policy id to be applied to the index (deprecated, use usage_policy_id).
        :param str usage_policy_id: The usage policy id to be applied to the index.
        """
        assert index_name, "Index name must be specified"

        # Validate mutual exclusivity
        if budget_policy_id and usage_policy_id:
            raise InvalidInputException(
                "Cannot specify both budget_policy_id and usage_policy_id. Please use only one."
            )

        # Use usage_policy_id if provided, otherwise fall back to budget_policy_id
        policy_id = usage_policy_id or budget_policy_id
        assert policy_id, "Either budget_policy_id or usage_policy_id must be specified"

        # Send as usage_policy_id to the backend (this endpoint already uses usage_policy_id)
        json_data = {"usage_policy_id": policy_id}
        return RequestUtils.issue_request(
            url=f"{self.workspace_url}/api/2.0/vector-search/indexes/{index_name}/usage-policy",
            token=self._get_token_for_request(),
            method="PATCH",
            json=json_data,
        )

    def update_index_usage_policy(self, index_name, usage_policy_id):
        """
        Update the usage policy of an index (alias for update_index_budget_policy).

        :param str index_name: The name of the index.
        :param str usage_policy_id: The usage policy id to be applied to the index.
        """
        return self.update_index_budget_policy(index_name, usage_policy_id=usage_policy_id)
