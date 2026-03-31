import hashlib
import time

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.models.model import ModelInfo
from mlflow.protos import databricks_pb2 as db_protos

from databricks.agents.utils.mlflow_utils import get_databricks_uc_registry_uri
from databricks.rag_eval.utils import uc_utils
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import platform as platform_errors
from databricks.sdk.service import serving


class ModelDoesNotExist(Exception):
    """Raised when a model does not exist in the registry."""

    pass


class ModelVersionDoesNotExist(Exception):
    """Raised when a version of a model does not exist in the registry."""

    pass


class ModelHasInvalidSignature(Exception):
    """Raised when a model has an invalid signature."""

    pass


class InvalidServingEndpoint(Exception):
    """Raised when a serving endpoint is invalid."""

    pass


def get_model_version(client: MlflowClient, model_uc_entity: uc_utils.UnityCatalogEntity, version: int) -> ModelInfo:
    """Get the given version of a model from the registry.

    Args:
        client (MlflowClient): The MLflow client to use for retrieving a model.
        model_uc_entity (UnityCatalogEntity): The model's Unity Catalog entity.
        version (int): The model version to retrieve. Must be greater than 0.

    Raises:
        ModelDoesNotExist: When the model does not exist in the registry.
        ModelVersionDoesNotExist: When the model version does not exist in the registry.

    Returns:
        ModelInfo: Information on the model version.
    """
    assert version > 0, "model version must be greater than 0"

    # check if model exists
    try:
        client.get_registered_model(name=model_uc_entity.fullname)
    except MlflowException as err:
        if err.error_code == db_protos.ErrorCode.Name(db_protos.RESOURCE_DOES_NOT_EXIST):
            raise ModelDoesNotExist
        raise

    # check if model version exists
    mlflow.set_registry_uri(get_databricks_uc_registry_uri())
    try:
        model_version = mlflow.models.get_model_info(f"models:/{model_uc_entity.fullname}/{str(version)}")
    except MlflowException as err:
        if err.error_code == db_protos.ErrorCode.Name(db_protos.RESOURCE_DOES_NOT_EXIST):
            raise ModelVersionDoesNotExist
        raise

    return model_version


def hash_to_six_chars(input_string: str) -> str:
    """Hashes a string of arbitrary length using md5 and returns
    the first six characters of the hash.

    We add a unix timestamp salt to the input string as a suffix before
    hashing to ensure that the hash is unique for even identical
    input strings. This is useful for preventing hash collisions.

    Args:
        input_string (str): The string to hash.

    Returns:
        str: The first six characters of the hashed string.
    """
    unix_timestamp = str(time.time())
    bytes_to_hash = (input_string + unix_timestamp).encode("utf-8")
    return hashlib.md5(bytes_to_hash).hexdigest()[:6]


def build_inference_table_name(endpoint_name: str) -> str:
    """Builds the name of the inference table for a given serving endpoint.

    Example:
        build_inference_table_name("test_endpoint_name") -> "test_endpoint_abcdef_traces"

    Args:
        endpoint_name (str): The name of the serving endpoint.

    Returns:
        str: The name of the inference table. Note that it's not fully qualified with catalog and schema.
    """
    suffix = "_traces"
    hashed_endpoint_name = hash_to_six_chars(endpoint_name)
    truncated_endpoint_name = endpoint_name[
        : uc_utils.MAX_UC_ENTITY_NAME_LEN - len(suffix) - len(hashed_endpoint_name) - 1
    ]
    return f"{truncated_endpoint_name}_{hashed_endpoint_name}{suffix}"


def create_model_serving_endpoint(
    client: WorkspaceClient,
    endpoint_name: str,
    config: serving.EndpointCoreConfigInput,
) -> serving.Wait[serving.ServingEndpointDetailed]:
    """Create a new model serving endpoint.

    Assumes the endpoint does not already exist.

    Args:
        client (WorkspaceClient): Databricks workspace client.
        endpoint_name (str): The name of the serving endpoint to create.
        config (EndpointCoreConfigInput): The configuration for the serving endpoint.

    Returns:
        Wait[ServingEndpointDetailed]: The created serving endpoint, which is not immediately available.
    """
    return client.serving_endpoints.create(
        name=endpoint_name,
        config=config,
    )


def delete_model_serving_endpoint(
    client: WorkspaceClient,
    endpoint_name: str,
) -> None:
    """Delete a model serving endpoint.

    Args:
        client (WorkspaceClient): Databricks workspace client.
        endpoint_name (str): The name of the serving endpoint to delete.

    Raises:
        PermissionDenied: When the user does not have permission to delete the serving endpoint.
        ResourceDoesNotExist: When the serving endpoint does not exist.
    """

    try:
        client.serving_endpoints.delete(name=endpoint_name)
    except platform_errors.PermissionDenied:
        raise platform_errors.PermissionDenied(
            f"User does not have permission to delete serving endpoint '{endpoint_name}'"
        )
    except platform_errors.ResourceDoesNotExist:
        raise platform_errors.ResourceDoesNotExist(f"Serving endpoint '{endpoint_name}' does not exist")
