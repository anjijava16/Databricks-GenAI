import logging
from typing import List, Optional

from databricks.agents.sdk_utils.entities import Artifacts, Deployment, Instructions
from databricks.agents.utils.mlflow_utils import get_workspace_url, get_workspace_id
from databricks.agents.utils.rest_utils import call_endpoint, construct_endpoint_url
from databricks.agents.utils.review_app_utils import (
    get_review_app_v2_from_model_version,
    get_review_app_v2_url,
)

_logger = logging.getLogger("agents")
_REVIEW_APP_V2_SEGMENT = "/review-v2/"


def _construct_chain_deployment(chain_deployment: dict) -> Deployment:
    # ToDo (ML-39446): Use automatic proto to py object conversion
    # Do not surface error to user if the backend return missing fields
    endpoint_name = chain_deployment.get("endpoint_name", None)
    workspace_url = get_workspace_url()
    workspace_id = get_workspace_id()
    deployment = Deployment(
        model_name=chain_deployment.get("model_name", None),
        model_version=chain_deployment.get("model_version", None),
        endpoint_name=endpoint_name,
        served_entity_name=chain_deployment.get("served_entity_name", None),
        query_endpoint=chain_deployment.get("query_endpoint", None),
        endpoint_url=construct_endpoint_url(workspace_url, endpoint_name, workspace_id),
        # ToDo (ML-42293): Remove the fallback to rag_app_url once the backend is updated
        review_app_url=chain_deployment.get("review_app_url", chain_deployment.get("rag_app_url", None)),
    )

    if not endpoint_name:
        return deployment

    if _REVIEW_APP_V2_SEGMENT in deployment.review_app_url:
        return deployment

    try:
        # Override the review app V1 URL with the v2 one; get or create a V2
        # review app
        review_app_v2 = get_review_app_v2_from_model_version(
            deployment.model_name, deployment.model_version, deployment.endpoint_name
        )
        deployment.review_app_url = get_review_app_v2_url(review_app_v2)
    except Exception:
        # Fallback to an endpoint-centric URL that doesn't depend on an experiment.
        endpoint_centric_url = f"{workspace_url}/ml/review-v2/chat?endpoint={endpoint_name}"
        if workspace_id is not None:
            endpoint_centric_url += f"&o={workspace_id}"
        deployment.review_app_url = endpoint_centric_url

    return deployment


def _parse_deploy_chain_response(response: dict) -> Deployment:
    # ToDo (ML-39446): Use automatic proto to py object conversion
    # Do not surface error to user if the backend return missing fields
    deployed_chain = response.get("deployed_chain", None)
    return _construct_chain_deployment(deployed_chain) if deployed_chain else None


def _parse_get_chain_deployments_response(response: dict) -> List[Deployment]:
    # ToDo (ML-39446): Use automatic proto to py object conversion
    # Do not surface error to user if the backend return missing fields
    chain_deployments = response.get("chain_deployments", [])
    return [_construct_chain_deployment(x) for x in chain_deployments]


def _parse_list_chain_deployments_response(response: dict) -> List[Deployment]:
    # ToDo (ML-39446): Use automatic proto to py object conversion
    # Do not surface error to user if the backend return missing fields
    chain_deployments = response.get("chain_deployments", [])
    return [_construct_chain_deployment(x) for x in chain_deployments]


def deploy_chain(
    model_name: str,
    model_version: str,
    query_endpoint: str,
    endpoint_name: str,
    served_entity_name: str,
    workspace_url: str,
) -> Deployment:
    request_body = {
        "model_name": model_name,
        "model_version": model_version,
        "query_endpoint": query_endpoint,
        "endpoint_name": endpoint_name,
        "served_entity_name": served_entity_name,
        "workspace_url": workspace_url,
    }
    response = call_endpoint(
        method="POST",
        route="deployments",
        json_body=request_body,
    )

    return _parse_deploy_chain_response(response)


def delete_chain(model_name: str, model_version: Optional[int] = None) -> None:
    route = f"deployments/{model_name}"
    request_body = {"model_name": model_name}
    if model_version:
        route = f"{route}/versions/{model_version}"
        request_body |= {"model_version": model_version}

    call_endpoint(
        method="DELETE",
        route=route,
        json_body=request_body,
    )
    return None


# TODO: add back in params once we've added pagination
# https://github.com/databricks/universe/pull/514434#discussion_r1518324393
def list_chain_deployments():
    response = call_endpoint(
        method="GET",
        route="deployments",
    )

    return _parse_list_chain_deployments_response(response)


def get_chain_deployments(model_name: str, model_version: Optional[str] = None):
    request_body = {"model_name": model_name}
    route = f"deployments/{model_name}"
    if model_version:
        route = f"{route}/versions/{model_version}"
        request_body |= {"model_version": model_version}

    response = call_endpoint(
        method="GET",
        route=route,
        json_body=request_body,
    )
    return _parse_get_chain_deployments_response(response)


def create_review_artifacts(model_name: str, artifacts: List[str]):
    request_body = {
        "model_name": model_name,
        "artifacts": artifacts,
    }
    call_endpoint(
        method="POST",
        route=f"deployments/{model_name}/artifacts",
        json_body=request_body,
    )


def get_review_artifacts(model_name: str):
    request_body = {"model_name": model_name}

    response = call_endpoint(
        method="GET",
        route=f"deployments/{model_name}/artifacts",
        json_body=request_body,
    )
    if "artifacts" not in response:
        return Artifacts(artifact_uris=[])
    return Artifacts(artifact_uris=response["artifacts"])


def set_review_instructions(model_name: str, instructions: str):
    request_body = {
        "model_name": model_name,
        "instructions": instructions,
    }
    call_endpoint(
        method="POST",
        route=f"deployments/{model_name}/instructions",
        json_body=request_body,
    )


def get_review_instructions(model_name: str) -> Instructions:
    request_body = {"model_name": model_name}
    response = call_endpoint(
        method="GET",
        route=f"deployments/{model_name}/instructions",
        json_body=request_body,
    )
    if "instructions" not in response:
        return Instructions(None)
    return Instructions(response["instructions"])
