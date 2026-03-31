import logging
from collections.abc import Iterable
from typing import Any, Collection, Dict, List, Optional, Sequence, Union

import pandas as pd
from mlflow.deployments import get_deploy_client
from mlflow.tracking import fluent
from requests import HTTPError

from databricks.agents.permissions import (
    ServingEndpointPermissionLevel,
    _clear_permissions_for_user_endpoint,
    _update_permissions_on_endpoint,
)
from databricks.rag_eval import context, entities
from databricks.rag_eval.mlflow import mlflow_utils
from databricks.rag_eval.utils import (
    NO_CHANGE,
    collection_utils,
    spark_utils,
    workspace_url_resolver,
)
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog

_logger = logging.getLogger(__name__)


def _get_managed_evals_client():
    return context.get_context().build_managed_evals_client()


@context.eval_context
def create_evals_table(
    evals_table_name: str,
    *,
    agent_name: Optional[str] = None,
    agent_serving_endpoint: Optional[str] = None,
) -> None:
    """
    Create a new evals table.

    Args:
        evals_table_name: The name of the evals table.
        agent_name: The human-readable name of the agent to display in the UI.
        agent_serving_endpoint: The name of the model serving endpoint that serves the agent.
    """

    exp_id = None
    try:
        exp_id = fluent._get_experiment_id()
    except Exception:
        pass

    _get_managed_evals_client().create_managed_evals_instance(
        instance_id=evals_table_name,
        agent_name=agent_name,
        agent_serving_endpoint=agent_serving_endpoint,
        experiment_ids=([exp_id] if exp_id is not None else []),
    )


@context.eval_context
def delete_evals_table(
    evals_table_name: str,
) -> None:
    """
    Delete an evals table.

    Args:
        evals_table_name: The name of the evals table.
    """
    _get_managed_evals_client().delete_managed_evals_instance(
        instance_id=evals_table_name,
    )


@context.eval_context
def update_eval_labeling_config(
    evals_table_name: str,
    *,
    agent_name: Optional[str] = NO_CHANGE,
    agent_serving_endpoint: Optional[str] = NO_CHANGE,
) -> None:
    """
    Update the configurations for the labeling of an evals table.

    Fields that are not provided will not be updated. Set them to None to clear them.

    Args:
        evals_table_name: The name of the evals table.
        agent_name: The human-readable name of the agent to display in the UI.
        agent_serving_endpoint: The name of the model serving endpoint that serves the agent.
    """
    _get_managed_evals_client().update_managed_evals_instance(
        instance_id=evals_table_name,
        agent_name=agent_name,
        agent_serving_endpoint=agent_serving_endpoint,
    )


@context.eval_context
def get_eval_labeling_config(
    evals_table_name: str,
) -> Dict[str, Any]:
    """
    Get the configurations for the labeling of an evals table.

    Args:
        evals_table_name: The name of the evals table.

    Returns:
        A dictionary containing the configurations for the labeling of the evals table.
    """
    instance = _get_managed_evals_client().get_managed_evals_instance(
        instance_id=evals_table_name,
    )
    return {
        "agent_name": instance.agent_name,
        "agent_serving_endpoint": instance.agent_serving_endpoint,
    }


@context.eval_context
def add_evals(
    evals_table_name: str,
    *,
    evals: Union[List[Dict], pd.DataFrame, "pyspark.sql.DataFrame"],  # noqa: F821
) -> List[str]:
    """
    Add evals to the evals table.

    Input evals should be in Agent Evaluation input schema (https://docs.databricks.com/en/generative-ai/agent-evaluation/evaluation-schema.html#evaluation-input-schema).

    Args:
        evals_table_name: The name of the evals table.
        evals: The evals to add. This can be a list of dictionaries, a pandas DataFrame, or a Spark DataFrame.

    Returns:
        A list of the IDs of the added evals.
    """
    evals = spark_utils.normalize_spark_df(evals)
    if isinstance(evals, pd.DataFrame):
        evals = evals.to_dict(orient="records")
    # at this point we should have normalized to a list of dictionaries

    # Convert numpy ndarray from the evals
    evals = collection_utils.convert_ndarray_to_list(evals)

    tag_set = set()
    for row in evals:
        tags = row.get("tags", [])
        if isinstance(tags, str) or not isinstance(tags, Iterable) or not all(isinstance(tag, str) for tag in tags):
            raise ValueError("Column `tags` must be a list of strings")
        tag_set.update(tags)

    tag_name_to_id = _get_tag_names_to_tag_ids(evals_table_name, tag_set)
    for row in evals:
        row["tag_ids"] = [tag_name_to_id[tag] for tag in row.get("tags", [])]

    return _get_managed_evals_client().add_evals(
        instance_id=evals_table_name,
        evals=evals,
    )


@context.eval_context
def delete_evals(
    evals_table_name: str,
    *,
    eval_ids: List[str],
) -> None:
    """
    Delete evals from the evals table.

    Args:
        evals_table_name: The name of the evals table.
        eval_ids: The IDs of the evals to delete.
    """
    _get_managed_evals_client().delete_evals(
        instance_id=evals_table_name,
        eval_ids=eval_ids,
    )


@context.eval_context
def list_tags(
    evals_table_name: str,
) -> List[str]:
    """
    List all tags in the evals table.

    Args:
        evals_table_name: The name of the evals table.

    Returns:
        A list of tag names.
    """
    full_tags = _get_managed_evals_client().list_tags(instance_id=evals_table_name)
    return [tag["tag_name"] for tag in full_tags]


@context.eval_context
def add_tags(
    evals_table_name: str,
    *,
    tag_names: Collection[str],
) -> None:
    """
    Add tags to the evals table.
    Tags that already exist will be skipped.

    Args:
        evals_table_name: The name of the evals table.
        tag_names: The names of the tags to add.
    """
    existing_tag_names = list_tags(evals_table_name=evals_table_name)
    missing_tag_names = set(tag_names) - set(existing_tag_names)
    if missing_tag_names:
        _get_managed_evals_client().batch_create_tags(
            instance_id=evals_table_name,
            tag_names=missing_tag_names,
        )


@context.eval_context
def tag_evals(
    evals_table_name: str,
    *,
    eval_ids: List[str],
    tag_names: List[str],
):
    """
    Add tags to multiple evals.

    All evals get all new tags. Tags that do not yet exist will be created.

    Args:
        evals_table_name: The name of the evals table.
        eval_ids: The eval IDs to tag.
        tag_names: The tags to apply.
    """
    tag_name_to_tag_ids = _get_tag_names_to_tag_ids(evals_table_name, tag_names)
    tag_ids = [tag_name_to_tag_ids[name] for name in tag_names]

    eval_tags = [entities.EvalTag(eval_id=eval_id, tag_id=tag_id) for eval_id in eval_ids for tag_id in tag_ids]
    _get_managed_evals_client().batch_create_eval_tags(instance_id=evals_table_name, eval_tags=eval_tags)


@context.eval_context
def untag_evals(
    evals_table_name: str,
    *,
    eval_ids: List[str],
    tag_names: List[str],
):
    """
    Remove tags from multiple evals.

    Args:
        evals_table_name: The name of the evals table.
        eval_ids: The eval IDs to untag.
        tag_names: The tags to remove.
    """
    # Don't use _get_tag_names_to_tag_ids as it will create tags if they didn't already exist.
    full_tags = _get_managed_evals_client().list_tags(instance_id=evals_table_name)
    tag_name_to_id = {tag["tag_name"]: tag["tag_id"] for tag in full_tags}
    # If a tag name isn't found, we assume that no eval can have that tag.
    existing_tag_ids = [tag_name_to_id[tag_name] for tag_name in tag_names if tag_name in tag_name_to_id]

    eval_tags = [
        entities.EvalTag(eval_id=eval_id, tag_id=tag_id) for eval_id in eval_ids for tag_id in existing_tag_ids
    ]

    _get_managed_evals_client().batch_delete_eval_tags(instance_id=evals_table_name, eval_tags=eval_tags)


@context.eval_context
def get_ui_links(evals_table_name: str, *, display_html: bool = True) -> List[Dict[str, str]]:
    """
    Get the UI links for the evals table.

    Args:
        evals_table_name: The name of the evals table.
        display_html: Whether to display the UI links as HTML buttons. Defaults to True.

    Returns:
        A list of dictionaries containing the name and URL for each UI page.
    """
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
    workspace_url = workspace_url_resolver.WorkspaceUrlResolver(spark.conf.get("spark.databricks.workspaceUrl"))

    instance = _get_managed_evals_client().get_managed_evals_instance(
        instance_id=evals_table_name,
    )
    if not instance.ui_page_configs:
        return []
    if display_html:
        for page in instance.ui_page_configs:
            button_to_page_html = f"""
            <a href="{workspace_url.get_full_url(page.path)}" target="_blank">
              <button style="color: white; background-color: #0073e6; padding: 10px 24px; cursor: pointer; border: none; border-radius: 4px;">
                {page.display_name}
              </button>
            </a>
            """
            context.get_context().display_html(button_to_page_html)
    return [
        {
            "name": page.display_name,
            "url": f"{workspace_url.get_full_url(page.path)}",
        }
        for page in instance.ui_page_configs
    ]


def _add_suffix_to_table(table_name: str, suffix: str) -> str:
    if table_name.endswith("`"):
        return f"{table_name[:-1]}{suffix}`"
    return f"{table_name}{suffix}"


@context.eval_context
def grant_access(
    evals_table_name: str,
    *,
    user_emails: List[str],
) -> None:
    """Grant access to read and modify the evals table (via Spark and UI) to the specified users.

    If an agent is configured for the labeling UI, this function also grants QUERY access to the model.

    Args:
        evals_table_name: The name of the evals table.
        user_emails: The emails of the users to grant access to.
    """
    deploy_client = get_deploy_client(mlflow_utils.resolve_deployments_target())

    evals_instance = _get_managed_evals_client().get_managed_evals_instance(evals_table_name)
    if evals_instance.version != "EntityStoreEnabled":
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.getOrCreate()
        workspace_client = WorkspaceClient()

        # Temporary shim: ensure delta table has correct permissions. Once EStore becomes the SoT,
        # we no longer need these permission updates.
        for suffix in [
            "_evals",
            "_tags",
            "_evals_tags",
            "_agent_invocations",
            "_assessments",
            "_metadata",
        ]:
            table_name = _add_suffix_to_table(evals_table_name, suffix)
            if not spark.catalog.tableExists(table_name):
                continue
            workspace_client.grants.update(
                securable_type=catalog.SecurableType.TABLE.value,
                full_name=table_name,
                changes=[
                    catalog.PermissionsChange(
                        principal=user_email,
                        add=[catalog.Privilege.SELECT, catalog.Privilege.MODIFY],
                    )
                    for user_email in user_emails
                ],
            )

    instance = _get_managed_evals_client().get_managed_evals_instance(instance_id=evals_table_name)
    agent_serving_endpoint = instance.agent_serving_endpoint
    endpoint_id = _get_updatable_endpoint_id_from_endpoint_name(deploy_client, agent_serving_endpoint)
    if endpoint_id is not None:
        try:
            _update_permissions_on_endpoint(endpoint_id, user_emails, ServingEndpointPermissionLevel.CAN_QUERY)
        except ValueError as e:
            _logger.warning(f"Could not update permissions for `{agent_serving_endpoint}`: {e!r}")
    _get_managed_evals_client().update_eval_permissions(instance_id=evals_table_name, add_emails=user_emails)


@context.eval_context
def sync_evals_to_uc(evals_table_name: str) -> None:
    """
    Sync stored evals to the provided UC table name.

    Args:
        evals_table_name: The name of the evals table.
    """
    _get_managed_evals_client().sync_evals_to_uc(instance_id=evals_table_name)


@context.eval_context
def revoke_access(
    evals_table_name: str,
    *,
    user_emails: List[str],
) -> None:
    """Revoke access to read and modify the evals table (via Spark and UI) to the specified users.

    If an agent is configured for the labeling UI, this function also revokes QUERY access to the model.

    Args:
        evals_table_name: The name of the evals table.
        user_emails: The emails of the users to revoke access from.
    """
    deploy_client = get_deploy_client(mlflow_utils.resolve_deployments_target())

    evals_instance = _get_managed_evals_client().get_managed_evals_instance(evals_table_name)
    if evals_instance.version != "EntityStoreEnabled":
        from pyspark.sql import SparkSession

        workspace_client = WorkspaceClient()

        spark = SparkSession.builder.getOrCreate()

        # Temporary shim: ensure delta table has correct permissions. Once EStore becomes the SoT,
        # we no longer need these permission updates.
        for suffix in [
            "_evals",
            "_tags",
            "_evals_tags",
            "_agent_invocations",
            "_assessments",
            "_metadata",
        ]:
            table_name = _add_suffix_to_table(evals_table_name, suffix)
            if not spark.catalog.tableExists(table_name):
                continue
            remove = [catalog.Privilege.ALL_PRIVILEGES]
            workspace_client.grants.update(
                securable_type=catalog.SecurableType.TABLE.value,
                full_name=table_name,
                changes=[catalog.PermissionsChange(principal=user, remove=remove) for user in user_emails],
            )

    agent_serving_endpoint = evals_instance.agent_serving_endpoint
    endpoint_id = _get_updatable_endpoint_id_from_endpoint_name(deploy_client, agent_serving_endpoint)
    if endpoint_id is not None:
        try:
            _clear_permissions_for_user_endpoint(endpoint_id, clear_users=user_emails)
        except ValueError as e:
            _logger.error(f"Could not update permissions for `{agent_serving_endpoint}`: {e!r}")

    _get_managed_evals_client().update_eval_permissions(instance_id=evals_table_name, remove_emails=user_emails)


def _get_updatable_endpoint_id_from_endpoint_name(deploy_client, agent_serving_endpoint) -> Optional[str]:
    """
    Gets the endpoint ID from the endpoint name.

    Edge cases:
    - endpoint is a FMAPI endpoint -> return None
    - endpoint does not exist -> return None
    - unexpected error -> raise
    """
    if not agent_serving_endpoint:
        return None
    try:
        endpoint = deploy_client.get_endpoint(agent_serving_endpoint)
    except HTTPError as e:
        if e.response.status_code == 404:
            _logger.warning(f"Endpoint `{agent_serving_endpoint}` does not exist.")
            return None
        raise
    if endpoint.get("endpoint_type") == "FOUNDATION_MODEL_API":
        _logger.info(f"Foundation model endpoint found: `{agent_serving_endpoint}`; permission update not needed.")
        return None
    return endpoint["id"]


def _get_tag_names_to_tag_ids(evals_table_name: str, tag_names: Sequence[str]):
    """Maps tag names back to tag id, ensuring existence of tags."""
    # insert any new tags into the system first (add_tag will handle deduping)
    add_tags(evals_table_name=evals_table_name, tag_names=list(tag_names))
    full_tags = _get_managed_evals_client().list_tags(instance_id=evals_table_name)
    tag_name_to_id = {tag["tag_name"]: tag["tag_id"] for tag in full_tags}
    return tag_name_to_id
