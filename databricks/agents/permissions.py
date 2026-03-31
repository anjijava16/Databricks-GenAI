import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from mlflow import get_experiment

from databricks.agents.client.rest_client import (
    get_review_artifacts as rest_get_review_artifacts,
)
from databricks.agents.sdk_utils.deployments import _get_deployments
from databricks.agents.sdk_utils.entities import PermissionLevel
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors.platform import (
    NotFound,
    PermissionDenied,
    ResourceDoesNotExist,
    Unauthenticated,
)
from databricks.sdk.service.serving import (
    ServingEndpointAccessControlRequest,
    ServingEndpointPermissionLevel,
    ServingEndpointPermissions,
)
from databricks.sdk.service.workspace import (
    WorkspaceObjectAccessControlRequest,
    WorkspaceObjectPermissionLevel,
    WorkspaceObjectPermissions,
)

_MLFLOW_EXPERIMENT_TYPE_TAG = "mlflow.experimentType"
_MLFLOW_EXPERIMENT_TYPE_VALUE = "MLFLOW_EXPERIMENT"
_NOTEBOOK_EXPERIMENT_TYPE_VALUE = "NOTEBOOK"


@dataclass
class UserPermissionInfo:
    user_type: str
    user_name: str
    permission_level: PermissionLevel

    def _to_dict(self):
        return {
            self.user_type: self.user_name,
            "permission_level": self.permission_level.value,
        }

    def to_experiment_permissions_object(self):
        return WorkspaceObjectAccessControlRequest.from_dict(self._to_dict())

    def to_serving_endpoint_permissions_object(self):
        return ServingEndpointAccessControlRequest.from_dict(self._to_dict())


class UserPermissionAclRequest:
    def __init__(self, user_name, permission_level):
        self.user_name = user_name
        self.permission_level = permission_level
        self.user_type = "user" if "@" in user_name else "group"

    def _build_acl_permission_dict(self):
        acl_permission_dict = {}
        acl_permission_dict["permission_level"] = self.permission_level.value
        if self.user_type == "user":
            acl_permission_dict["user_name"] = self.user_name
        elif self.user_type == "group":
            acl_permission_dict["group_name"] = self.user_name
        return acl_permission_dict

    def to_experiment_permissions_object(self):
        return WorkspaceObjectAccessControlRequest.from_dict(self._build_acl_permission_dict())

    def to_serving_endpoint_permissions_object(self):
        return ServingEndpointAccessControlRequest.from_dict(self._build_acl_permission_dict())


def _get_run_ids_from_artifact_uris(artifact_uris: List[str]) -> List[str]:
    return [re.search(r"runs:/(.*?)/.*", artifact_id).group(1) for artifact_id in artifact_uris]


def _get_experiment_ids(run_ids: List[str]) -> List[str]:
    w = WorkspaceClient()
    experiment_ids = set()
    for run_id in run_ids:
        run_response = w.experiments.get_run(run_id)
        experiment_ids.add(run_response.run.info.experiment_id)
    return list(experiment_ids)


# Given an endpoint, calls it with the appropriate arguments and handles errors
def _call_workspace_api(endpoint, kwargs):
    try:
        return endpoint(**kwargs)
    except Unauthenticated as e:
        raise ValueError("Unable to authenticate to the databricks workspace: " + str(e)) from e
    except PermissionDenied as e:
        raise ValueError(
            "Permission Denied: User does not have valid permissions for setting permssions on the deployment."
        ) from e
    except ResourceDoesNotExist as e:
        raise ValueError("Resource does not exist, please check your inputs: " + str(e)) from e
    except NotFound as e:
        raise ValueError("Invalid Inputs: Passed in parameters are not found. " + str(e)) from e
    except Exception as e:
        raise e


# Get permissions on a given endpoint
def _get_permissions_on_endpoint(endpoint_id: str) -> ServingEndpointPermissions:
    w = WorkspaceClient()
    permissions = _call_workspace_api(w.serving_endpoints.get_permissions, {"serving_endpoint_id": endpoint_id})
    return permissions


# Get permissions on a given experiment
def _get_permissions_on_experiment(
    experiment_id: str,
    experiment_type: str,
) -> WorkspaceObjectPermissions:
    w = WorkspaceClient()
    permissions = _call_workspace_api(
        w.workspace.get_permissions,
        {
            "workspace_object_type": experiment_type,
            "workspace_object_id": experiment_id,
        },
    )
    return permissions


# Given a Permissions Object, and a list of users returns new permissions without the users
def _remove_users_from_permissions_list(permissions, users):
    user_set = set(users)
    acls = permissions.access_control_list
    modified_acls = list(filter(lambda acl: _extract_permission_identifier(acl) not in user_set, acls))
    # No Changes as the user has no permissions on the endpoint
    if len(modified_acls) == len(acls):
        return None
    new_permissions = []
    for acl in modified_acls:
        for permission in acl.all_permissions:
            # "user_name", "group_name" and "service_principal_name" are all keywords used by the permission API later
            if acl.user_name is not None:
                user_type = "user_name"
                user_name = acl.user_name
            elif acl.group_name is not None:
                user_type = "group_name"
                user_name = acl.group_name
            else:
                user_type = "service_principal_name"
                user_name = acl.service_principal_name

            new_permissions.append(UserPermissionInfo(user_type, user_name, permission.permission_level))
    return new_permissions


# For a given a chain model name get all logged trace artifacts and return the corresponding experiment IDs
def _get_experiment_ids_from_trace_artifacts(model_name: str) -> List[str]:
    ml_artifacts = rest_get_review_artifacts(model_name)
    experiment_ids = _get_experiment_ids(_get_run_ids_from_artifact_uris(ml_artifacts.artifact_uris))
    return experiment_ids


# Given an ACL request, returns either a userName, group name or service principal name
# It is guaranteed that only one of these is defined and the rest are marked as None
def _extract_permission_identifier(permission_acl) -> str:
    return permission_acl.user_name or permission_acl.group_name or permission_acl.service_principal_name


# Sets permissions on an endpoint for the list of users
# Permissions is of type [((User_type, username), PermissionLevel)]
def _set_permissions_on_endpoint(
    endpoint_id: str,
    permissions: List[UserPermissionInfo],
):
    if permissions is None:
        return
    acls = []
    for permission in permissions:
        acls.append(permission.to_serving_endpoint_permissions_object())
    # NOTE: this function will overwrite all permissions for the endpoint
    w = WorkspaceClient()
    _call_workspace_api(
        w.serving_endpoints.set_permissions,
        {
            "serving_endpoint_id": endpoint_id,
            "access_control_list": acls,
        },
    )


# Sets permission on experiment
def _set_permissions_on_experiment(
    experiment_id: str,
    experiment_type: str,
    permissions: List[UserPermissionInfo],
):
    if permissions is None:
        return
    acls = []
    for permission in permissions:
        acls.append(permission.to_experiment_permissions_object())
    # NOTE: this function will overwrite all permissions for the experiment
    w = WorkspaceClient()
    _call_workspace_api(
        w.workspace.set_permissions,
        {
            "workspace_object_type": experiment_type,
            "workspace_object_id": experiment_id,
            "access_control_list": acls,
        },
    )


# Update Permissions on Endpoint
def _update_permissions_on_endpoint(
    endpoint_id: str,
    users: List[str],
    permission_level: ServingEndpointPermissionLevel,
):
    acl_requests = [
        UserPermissionAclRequest(user, permission_level).to_serving_endpoint_permissions_object() for user in users
    ]
    w = WorkspaceClient()
    _call_workspace_api(
        w.serving_endpoints.update_permissions,
        {
            "serving_endpoint_id": endpoint_id,
            "access_control_list": acl_requests,
        },
    )


def _update_permissions_on_experiment(
    experiment_ids: List[str],
    users: List[str],
    experiment_type: str,
    permission_level: Optional[WorkspaceObjectPermissionLevel] = None,
):
    acl_requests = [
        UserPermissionAclRequest(user, permission_level).to_experiment_permissions_object() for user in users
    ]
    # NOTE: all experiments must be of the same type (notebooks vs mlflow experiments)
    w = WorkspaceClient()
    for experiment_id in experiment_ids:
        _call_workspace_api(
            w.workspace.update_permissions,
            {
                "workspace_object_type": experiment_type,
                "workspace_object_id": experiment_id,
                "access_control_list": acl_requests,
            },
        )


def _get_endpoint_id_for_deployed_model(model_name: str):
    endpoint_ids = set()
    chain_deployments = _get_deployments(model_name)
    w = WorkspaceClient()
    for deployment in chain_deployments:
        serving_endpoint = _call_workspace_api(w.serving_endpoints.get, {"name": deployment.endpoint_name})
        endpoint_ids.add(serving_endpoint.id)
    return endpoint_ids


def _clear_permissions_for_user_endpoint(endpoint_id: str, clear_users: List[str]):
    # Retrieves all the permissions in the endpoint. Returned list is permission level mapping for all users
    permissions = _get_permissions_on_endpoint(endpoint_id)
    # Filter permissions list such that users in `clear_users` do not have any permissions.
    new_permissions = _remove_users_from_permissions_list(permissions, clear_users)
    # Re sets the permissions for the remaining users
    _set_permissions_on_endpoint(endpoint_id, new_permissions)


def _clear_permission_for_experiments(experiment_ids: List[str], clear_users: List[str], experiment_type: str):
    # NOTE: all experiments must be of the same type (notebooks vs mlflow experiments)
    for experiment_id in experiment_ids:
        # Retrieves all the permissions in the experiment. Returned list is permission level mapping for all users
        experiment_permissions = _get_permissions_on_experiment(experiment_id, experiment_type)
        # Filter permissions list such that users in `clear_users` do not have any permissions.
        new_permissions = _remove_users_from_permissions_list(experiment_permissions, clear_users)
        # Re sets the permisssions for the remaining users
        _set_permissions_on_experiment(experiment_id, experiment_type, new_permissions)


def _filter_experiments_by_type(experiment_ids: List[str]):
    mlflow_experiment_ids = []
    notebook_experiment_ids = []
    for experiment_id in experiment_ids:
        experiment = get_experiment(experiment_id)
        experiment_type = experiment.tags.get(_MLFLOW_EXPERIMENT_TYPE_TAG)
        if experiment_type == _NOTEBOOK_EXPERIMENT_TYPE_VALUE:
            notebook_experiment_ids.append(experiment_id)
        elif experiment_type == _MLFLOW_EXPERIMENT_TYPE_VALUE:
            mlflow_experiment_ids.append(experiment_id)
    return notebook_experiment_ids, mlflow_experiment_ids


def set_permissions(
    model_name: str,
    users: List[str],
    permission_level: PermissionLevel,
):
    """
    Grant or revoke permissions to chat with the agent endpoint associated with the specified model_name in the
    review app. Note: This API updates agent endpoint permissions only, not permissions on the underlying UC model.

    Args:
        model_name: Name of the UC registered model.
        users: List of account user emails or groups.
        permission_level: Permissions level assigned to the list of users.
            Supported permission levels are:

            - `NO_PERMISSIONS`: chat and review privileges revoked for users
            - `CAN_VIEW`: users can list and get metadata for deployed agents
            - `CAN_QUERY`: users can use chat with the agent and provide feedback on their own chats
            - `CAN_REVIEW`: users can provide feedback on review traces
            - `CAN_MANAGE`: users can update existing agent deployments and deploy agents.
    """
    users_set = set(users)
    users = list(users_set)
    endpoint_ids = _get_endpoint_id_for_deployed_model(model_name)
    # Set permissions on Experiments if necessary
    experiment_ids = _get_experiment_ids_from_trace_artifacts(model_name)

    if len(endpoint_ids) == 0:
        raise ValueError("No deployments found for model_name " + model_name)

    if len(experiment_ids) == 0 and permission_level == PermissionLevel.CAN_REVIEW:
        raise ValueError(
            f"Cannot assign review permissions as no review artifacts found for model name: {model_name}. Please create review artifacts using enable_trace_reviews."
        )

    # Set Permissions on Endpoints
    for endpoint_id in endpoint_ids:
        if permission_level == PermissionLevel.NO_PERMISSIONS:
            _clear_permissions_for_user_endpoint(endpoint_id, users)
        elif permission_level == PermissionLevel.CAN_VIEW:
            _update_permissions_on_endpoint(endpoint_id, users, ServingEndpointPermissionLevel.CAN_VIEW)
        elif permission_level == PermissionLevel.CAN_QUERY:
            _update_permissions_on_endpoint(endpoint_id, users, ServingEndpointPermissionLevel.CAN_QUERY)
        elif permission_level == PermissionLevel.CAN_REVIEW:
            _update_permissions_on_endpoint(endpoint_id, users, ServingEndpointPermissionLevel.CAN_QUERY)
        elif permission_level == PermissionLevel.CAN_MANAGE:
            _update_permissions_on_endpoint(endpoint_id, users, ServingEndpointPermissionLevel.CAN_MANAGE)

    # filter experiments into notebook and mlflow experiments
    # NOTE: we will eventually remove notebook experiment handling.
    notebook_experiment_ids, mlflow_experiment_ids = _filter_experiments_by_type(experiment_ids)

    if permission_level == PermissionLevel.NO_PERMISSIONS:
        _clear_permission_for_experiments(notebook_experiment_ids, users, "notebooks")
        _clear_permission_for_experiments(mlflow_experiment_ids, users, "experiments")
    elif permission_level == PermissionLevel.CAN_VIEW:
        # If the user previously had any permissions on the experiment delete them
        _clear_permission_for_experiments(notebook_experiment_ids, users, "notebooks")
        _clear_permission_for_experiments(mlflow_experiment_ids, users, "experiments")
    elif permission_level == PermissionLevel.CAN_QUERY:
        # If the user previously had any permissions on the experiment delete them
        _clear_permission_for_experiments(notebook_experiment_ids, users, "notebooks")
        _clear_permission_for_experiments(mlflow_experiment_ids, users, "experiments")
    elif permission_level == PermissionLevel.CAN_REVIEW:
        _update_permissions_on_experiment(
            notebook_experiment_ids,
            users,
            "notebooks",
            WorkspaceObjectPermissionLevel.CAN_READ,
        )
        _update_permissions_on_experiment(
            mlflow_experiment_ids,
            users,
            "experiments",
            WorkspaceObjectPermissionLevel.CAN_READ,
        )
    elif permission_level == PermissionLevel.CAN_MANAGE:
        # If the user previously had any permissions on the experiment delete them
        _update_permissions_on_experiment(
            notebook_experiment_ids,
            users,
            "notebooks",
            WorkspaceObjectPermissionLevel.CAN_MANAGE,
        )
        _update_permissions_on_experiment(
            mlflow_experiment_ids,
            users,
            "experiments",
            WorkspaceObjectPermissionLevel.CAN_MANAGE,
        )


# Constants for permission mappings for comparison
WORKSPACE_PERMISSION_LEVEL_MAPPING = {
    WorkspaceObjectPermissionLevel.CAN_READ: 0,
    WorkspaceObjectPermissionLevel.CAN_RUN: 1,
    WorkspaceObjectPermissionLevel.CAN_EDIT: 2,
    WorkspaceObjectPermissionLevel.CAN_MANAGE: 3,
}

SERVING_ENDPOINT_PERMISSION_LEVEL_MAPPING = {
    ServingEndpointPermissionLevel.CAN_VIEW: 0,
    ServingEndpointPermissionLevel.CAN_QUERY: 1,
    ServingEndpointPermissionLevel.CAN_MANAGE: 2,
}


def _aggregate_permissions(ids, get_permissions_func, permission_mapping):
    """Aggregate minimum permissions given multiple ids, maintaining usage of specific getters. Returns mapping of user name to the user's minimum permission as a numerical constant.
    - ids: list of ids to aggregate permissions over
    - get_permissions_func: takes in an id and returns permissions for that id
    - permission_mapping: dict mapping permission level to a numerical constant
    """
    min_permissions = {}
    for item_id in ids:
        permissions = get_permissions_func(item_id)
        for acl in permissions.access_control_list:
            permission_id = _extract_permission_identifier(acl)
            current_perm = min_permissions.get(permission_id, float("inf"))
            user_permissions = [permission_mapping.get(p.permission_level, float("inf")) for p in acl.all_permissions]
            min_permissions[permission_id] = min(current_perm, *user_permissions)
    return min_permissions


def _derive_combined_permission_level(endpoint_perm, experiment_perm, experiments_exist):
    """Derive a combined permission level from endpoint and experiment permissions.
    NO_PERMISSIONS -> no permission on endpoints
    CAN_VIEW -> ServingEndpointPermissionLevel.CAN_VIEW on endpoints
    CAN_QUERY -> ServingEndpointPermissionLevel.CAN_QUERY on endpoints
    CAN_REVIEW -> ServingEndpointPermissionLevel.CAN_QUERY (or higher) on endpoints and WorkspaceObjectPermissionLevel.CAN_READ (or higher) on experiments
    CAN_MANAGE -> ServingEndpointPermissionLevel.CAN_MANAGE (or higher) on endpoints and WorkspaceObjectPermissionLevel.CAN_READ (or higher) on experiments
    """

    # - Check whether:
    #  - The endpoint permissions are can manage
    #  - New Experiments will already have can manage through `enable_trace_reviews` and existing experiments would have can manage through `set_permsissions`
    #  - Therefore we don't need to check for experiment permissions
    if endpoint_perm == SERVING_ENDPOINT_PERMISSION_LEVEL_MAPPING[ServingEndpointPermissionLevel.CAN_MANAGE]:
        return PermissionLevel.CAN_MANAGE
    # - Check whether:
    #  - The endpoint permissions are can query
    #  - If experiments exists and the user has permissions of more than a read then return CAN_REVIEW
    if endpoint_perm >= SERVING_ENDPOINT_PERMISSION_LEVEL_MAPPING[ServingEndpointPermissionLevel.CAN_QUERY]:
        if (
            experiments_exist
            and experiment_perm >= WORKSPACE_PERMISSION_LEVEL_MAPPING[WorkspaceObjectPermissionLevel.CAN_READ]
        ):
            return PermissionLevel.CAN_REVIEW
        else:
            return PermissionLevel.CAN_QUERY
    if endpoint_perm == SERVING_ENDPOINT_PERMISSION_LEVEL_MAPPING[ServingEndpointPermissionLevel.CAN_VIEW]:
        return PermissionLevel.CAN_VIEW
    return PermissionLevel.NO_PERMISSIONS


def get_permissions(model_name: str) -> List[Tuple[str, PermissionLevel]]:
    """Compute combined minimum permissions for endpoints and experiments associated with a model."""
    # Fetch endpoints and experiments
    endpoint_ids = _get_endpoint_id_for_deployed_model(model_name)
    experiment_ids = _get_experiment_ids_from_trace_artifacts(model_name)
    experiments_exist = len(experiment_ids) > 0
    notebook_ids, mlflow_ids = _filter_experiments_by_type(experiment_ids)

    # Get minimum permissions for each type
    min_endpoint_permissions = _aggregate_permissions(
        endpoint_ids,
        _get_permissions_on_endpoint,
        SERVING_ENDPOINT_PERMISSION_LEVEL_MAPPING,
    )
    min_notebook_permissions = _aggregate_permissions(
        notebook_ids,
        lambda x: _get_permissions_on_experiment(x, "notebooks"),
        WORKSPACE_PERMISSION_LEVEL_MAPPING,
    )
    min_mlflow_permissions = _aggregate_permissions(
        mlflow_ids,
        lambda x: _get_permissions_on_experiment(x, "experiments"),
        WORKSPACE_PERMISSION_LEVEL_MAPPING,
    )

    # Combine notebook and MLflow experiment permissions
    min_experiment_permissions = {}
    for user, perm in {**min_notebook_permissions, **min_mlflow_permissions}.items():
        min_experiment_permissions[user] = min(min_experiment_permissions.get(user, float("inf")), perm)

    # Combine and evaluate overall permissions
    combined_permissions = {}
    all_users = set(min_endpoint_permissions.keys()).union(min_experiment_permissions.keys())
    for user in all_users:
        endpoint_perm_level = min_endpoint_permissions.get(user, -1)
        experiment_perm_level = min_experiment_permissions.get(user, -1)

        # Derive combined permission level
        combined_permissions[user] = _derive_combined_permission_level(
            endpoint_perm_level, experiment_perm_level, experiments_exist
        )

    return list(combined_permissions.items())
