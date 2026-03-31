# Entities for SDK

from dataclasses import dataclass
from enum import Enum
from typing import List


@dataclass
class Deployment:
    model_name: str
    model_version: str
    endpoint_name: str
    served_entity_name: str
    query_endpoint: str  # URI
    endpoint_url: str  # URI
    review_app_url: str  # URI


@dataclass
class Artifacts:
    # List of artifact uris of the format `runs:/<run_id>/<artifact_path>`
    artifact_uris: List[str]


@dataclass
class Instructions:
    instructions: str


class PermissionLevel(Enum):
    """Permission level for chat and review apps."""

    NO_PERMISSIONS = 1, "Users have no chat and review privileges."
    CAN_VIEW = 2, "Users can list and get metadata for deployed agents."
    CAN_QUERY = (
        3,
        "Users can use chat with the agent and provide feedback on their own chats.",
    )
    CAN_REVIEW = 4, "Users can provide feedback on review traces."
    CAN_MANAGE = 5, "Users can update existing agent deployments and deploy agents."
