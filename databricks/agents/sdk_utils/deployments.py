from typing import List, Optional

import databricks.agents.client.rest_client
from databricks.agents.sdk_utils.entities import Deployment


def _get_deployments(model_name: str, model_version: Optional[int] = None) -> List[Deployment]:
    return databricks.agents.client.rest_client.get_chain_deployments(model_name, model_version)


def get_latest_chain_deployment(model_name: str) -> Deployment:
    chain_deployments = databricks.agents.client.rest_client.get_chain_deployments(model_name)
    if len(chain_deployments) == 0:
        raise ValueError(
            f"Model {model_name} has never been deployed. "
            "Please deploy the model first using the databricks.agents.deploy() API"
        )
    return chain_deployments[-1]
