from databricks.agents.sdk_utils.entities import Deployment
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors.platform import (
    PermissionDenied,
    ResourceDoesNotExist,
)


def _check_view_permissions_on_deployment(deployment: Deployment):
    w = WorkspaceClient()
    try:
        w.serving_endpoints.get(deployment.endpoint_name)
    except PermissionDenied:
        raise ValueError(
            "You do not have the necessary permissions to view this deployment. Please ensure you have CAN_VIEW permissions on this deployment to view it."
        )
    except ResourceDoesNotExist:
        raise ResourceDoesNotExist("This deployment does not exist.")
