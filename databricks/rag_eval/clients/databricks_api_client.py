"""
Base class for concrete clients to various Databricks APIs
"""

import abc
from typing import Dict, Optional, Union

import requests
from requests import adapters
from urllib3.util import retry

import databricks.sdk.config


class DatabricksAPIClient(abc.ABC):
    """
    This is a base client to talk to Databricks API. The child classes of this base client can use the `get_method_url()`
    and `get_default_request_session()` methods provided by `DatabricksAPIClient` to send request to the
    corresponding Databricks API.

    Host resolution and authentication is handled by the databricks-sdk.

    Example Usage:

    class MyClient(DatabricksAPIClient):
      def __init__(self):
        super().__init__(version="2.0")

      def send_request(self):
        with self.get_default_request_session() as s:
            resp = s.post(self.get_method_url("list"), json="request_body: {...}", auth=self.get_auth())
        self.process_response(resp)
    """

    def __init__(self, version: str = "2.0"):
        """
        :param version: The version of the Databricks API, e.g. "2.1", "2.0"
        """
        self._version = version
        self._path_prefix = f"api/{version}"

        # Rely on databricks-sdk for host resolution and authentication
        self._config = databricks.sdk.config.Config()

    def get_method_url(self, method_path: str):
        """
        Returns the URL to invoke a specific method. This is a concatenation of the host, the prefix, and the
        corresponding method path.

        :param method_path: The method path
        """
        return f"{self._config.host}/{self._path_prefix}/{method_path.lstrip('/')}"

    def get_default_request_session(
        self,
        retry_config: Optional[retry.Retry] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> requests.Session:
        """
        Creates a request session with a retry mechanism, headers, and default authentication
        :return: Session object.
        """
        session = self._get_request_session(retry_config)
        if headers is not None:
            session.headers.update(headers)
        session.auth = self._authenticate
        return session

    @classmethod
    def _get_request_session(cls, retry_config: Optional[Union[retry.Retry, int]] = None) -> requests.Session:
        """
        Creates a request session with a retry mechanism.

        :return: Session object.
        """
        if retry_config is None:
            retry_config = adapters.DEFAULT_RETRIES
        adapter = adapters.HTTPAdapter(max_retries=retry_config)
        session = requests.Session()
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        return session

    def _authenticate(self, r: requests.PreparedRequest) -> requests.PreparedRequest:
        """
        Authenticate the request with the provided headers.

        Adapted from https://github.com/databricks/databricks-sdk-py/blob/cbae014ac73b99c659646daa1e0d42f939452567/databricks/sdk/_base_client.py#L100
        """
        if self._config.authenticate:
            headers = self._config.authenticate()
            for k, v in headers.items():
                r.headers[k] = v
        return r


def raise_for_status(resp: requests.Response) -> None:
    """
    Raise an Exception if the response is an error.
    Custom error message is extracted from the response JSON.
    """
    if resp.status_code == requests.codes.ok:
        return
    http_error_msg = ""
    if 400 <= resp.status_code < 500:
        http_error_msg = f"{resp.status_code} Client Error: {resp.reason}\n{resp.text}. "
    elif 500 <= resp.status_code < 600:
        http_error_msg = f"{resp.status_code} Server Error: {resp.reason}\n{resp.text}. "
    raise requests.HTTPError(http_error_msg, response=resp)
