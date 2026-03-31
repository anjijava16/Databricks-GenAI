import requests
import os
import threading
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from functools import lru_cache
from databricks.vector_search.version import VERSION
from databricks.vector_search.exceptions import (
    BadRequest,
    NotFound,
    PermissionDenied,
    ResourceConflict,
    TooManyRequests,
    VectorSearchException,
)
from enum import Enum
from requests.packages import urllib3
from packaging import version
from typing import Optional


class OAuthTokenUtils:

    @staticmethod
    def get_azure_oauth_token(
        workspace_url,
        azure_tenant_id,
        azure_login_id,
        service_principal_client_id,
        service_principal_client_secret,
        authorization_details=None,
    ):
        assert (
            azure_login_id and azure_tenant_id
        ), "Both azure_login_id and azure_tenant_id must be specified"

        # Currently VectorSearch is only available in AZURE_PUBLIC, hence the url is hardcoded to
        # the active_directory_endpoint in AZURE_PUBLIC, see
        # See https://github.com/databricks/databricks-sdk-py/blob/main/databricks/sdk/environments.py
        # for the list of Azure login ids and active directory endpoint in Azure environments
        active_directory_endpoint = "https://login.microsoftonline.com/"
        aad_url = f"{active_directory_endpoint}/{azure_tenant_id}/oauth2/token"
        aad_response = RequestUtils.issue_request(
            url=aad_url,
            method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "client_credentials",
                "resource": azure_login_id,
                "client_id": service_principal_client_id,
                "client_secret": service_principal_client_secret,
            },
        )
        aad_token = aad_response["access_token"]
        authorization_details = authorization_details or []
        if not authorization_details:
            return aad_response

        # If authorization_details is specified, we need to exchange the AAD token for an OAuth token
        return RequestUtils.issue_request(
            url=workspace_url + "/oidc/v1/token",
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "*/*",
            },
            method="POST",
            data={
                "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                "assertion": aad_token,
                "authorization_details": authorization_details,
            },
        )

    @staticmethod
    def get_oauth_token(
        workspace_url,
        service_principal_client_id,
        service_principal_client_secret,
        authorization_details=None,
    ):
        authorization_details = authorization_details or []
        url = workspace_url + "/oidc/v1/token"
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        data = {
            "grant_type": "client_credentials",
            "scope": "all-apis",
            "authorization_details": authorization_details,
        }
        response = RequestUtils.issue_request(
            url=url,
            auth=(service_principal_client_id, service_principal_client_secret),
            headers=headers,
            method="POST",
            data=data,
        )
        return response


@lru_cache(maxsize=64)
def _cached_get_request_session(
    total_retries,
    backoff_factor,
    backoff_jitter,
    # To create a new Session object for each process, we use the process id as the cache key.
    # This is to avoid sharing the same Session object across processes, which can lead to issues
    # such as https://stackoverflow.com/q/3724900.
    process_id,
):
    session = requests.Session()

    # Check if urllib3 version supports backoff_jitter parameter
    urllib3_version = version.parse(urllib3.__version__)
    supports_backoff_jitter = urllib3_version >= version.parse("2.0.0")
    # Note: The Retry class automatically respects Retry-After headers in 429 and 503 responses.
    # If Retry-After is present, it takes precedence over the backoff_factor calculation.
    if supports_backoff_jitter:
        retry_strategy = Retry(
            total=total_retries,  # Total number of retries
            backoff_factor=backoff_factor,  # A backoff factor to apply between attempts
            backoff_jitter=backoff_jitter,
            status_forcelist=[429, 503, 504],  # HTTP status codes to retry on
        )
    else:
        retry_strategy = Retry(
            total=total_retries,  # Total number of retries
            backoff_factor=backoff_factor,  # A backoff factor to apply between attempts
            status_forcelist=[429, 503, 504],  # HTTP status codes to retry on
        )

    adapter = HTTPAdapter(
        max_retries=retry_strategy, pool_connections=50, pool_maxsize=50
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


class RequestUtils:
    session = _cached_get_request_session(
        total_retries=3, backoff_factor=1, backoff_jitter=0.2, process_id=os.getpid()
    )

    @staticmethod
    def issue_request(
        url: str,
        method: str,
        token: Optional[str] = None,
        params: Optional[dict] = None,
        json: Optional[dict] = None,
        verify: bool = True,
        auth: Optional[tuple] = None,
        data: Optional[dict] = None,
        headers: Optional[dict] = None,
        total_retries: int = 3,
        backoff_factor: float = 1.0,
        backoff_jitter: float = 0.2,
    ):
        headers = headers or dict()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        headers["X-Databricks-Python-SDK-Version"] = VERSION

        # Use custom session with the provided retry parameters
        session = _cached_get_request_session(
            total_retries=total_retries,
            backoff_factor=backoff_factor,
            backoff_jitter=backoff_jitter,
            process_id=os.getpid(),
        )

        response = session.request(
            url=url,
            headers=headers,
            method=method,
            params=params,
            json=json,
            verify=verify,
            auth=auth,
            data=data,
        )

        if not response.ok:
            # Parse error message from JSON response
            error_message = f"HTTP {response.status_code}"
            try:
                error_data = response.json()
                if isinstance(error_data, dict):
                    error_message = error_data.get("message") or error_data.get("error") or str(error_data)
            except Exception:
                error_message = response.text or f"HTTP {response.status_code} error"
            status_code = response.status_code
            if status_code == 400:
                raise BadRequest(error_message, status_code=status_code, response_content=response.content)
            elif status_code == 403:
                raise PermissionDenied(error_message, status_code=status_code, response_content=response.content)
            elif status_code == 404:
                raise NotFound(error_message, status_code=status_code, response_content=response.content)
            elif status_code == 409:
                raise ResourceConflict(error_message, status_code=status_code, response_content=response.content)
            elif status_code == 429:
                raise TooManyRequests(error_message, status_code=status_code, response_content=response.content)
            else:
                # For any other error (500, 503, etc.), raise generic exception with details
                raise VectorSearchException(error_message, status_code=status_code, response_content=response.content)

        return response.json()


class UrlUtils:

    @staticmethod
    def add_https_if_missing(url):
        if not url.startswith("http://") and not url.startswith("https://"):
            url = "https://" + url
        return url


class CredentialStrategy(Enum):
    MODEL_SERVING_USER_CREDENTIALS = 1


def _get_invokers_token_fallback():
    main_thread = threading.main_thread()
    thread_data = main_thread.__dict__
    invokers_token = None
    if "invokers_token" in thread_data:
        invokers_token = thread_data["invokers_token"]
    return invokers_token


def _get_invokers_token_from_mlflowserving():
    try:
        from mlflowserving.scoring_server.agent_utils import fetch_obo_token

        return fetch_obo_token()
    except ImportError:
        return _get_invokers_token_fallback()


def get_model_serving_invoker_credentials():
    host = None
    invokers_token = None

    if is_in_model_serving_environment():
        host = os.environ.get("DATABRICKS_MODEL_SERVING_HOST_URL") or os.environ.get(
            "DB_MODEL_SERVING_HOST_URL"
        )

        invokers_token = _get_invokers_token_from_mlflowserving()

        if invokers_token is None:
            raise RuntimeError(
                "Unable to read Invokers Token in Databricks Model Serving"
            )

    return host, invokers_token


def is_in_model_serving_environment():
    """
    Check whether this is the model serving environment
    Additionally check if the oauth token file path exists
    """
    is_in_model_serving_env = (
        os.environ.get("IS_IN_DB_MODEL_SERVING_ENV")
        or os.environ.get("IS_IN_DATABRICKS_MODEL_SERVING_ENV")
        or "false"
    )
    return is_in_model_serving_env == "true"


def authentication_warning(
    notebook_token=None, personal_access_token=None, disable_notice=False
):
    if not disable_notice:
        if notebook_token:
            print(
                """[NOTICE] Using a notebook authentication token. Recommended for development only. For improved performance, please use Service Principal based authentication. To disable this message, pass disable_notice=True."""
            )
        elif personal_access_token:
            print(
                """[NOTICE] Using a Personal Authentication Token (PAT). Recommended for development only. For improved performance, please use Service Principal based authentication. To disable this message, pass disable_notice=True."""
            )
