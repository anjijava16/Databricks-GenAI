import logging
import uuid
import warnings
from typing import Dict, List, Optional

import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.tracking.default_experiment import DEFAULT_EXPERIMENT_ID
from mlflow.types import ColSpec, DataType, Schema
from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed,
)
from databricks.agents.utils.mlflow_utils import get_workspace_id

# Agents SDK
import databricks.agents
from databricks.agents.client.rest_client import (
    delete_chain as rest_delete_chain,
)
from databricks.agents.client.rest_client import (
    deploy_chain as rest_deploy_chain,
)
from databricks.agents.client.rest_client import (
    list_chain_deployments as rest_list_chain_deployments,
)
from databricks.agents.feedback import _FEEDBACK_MODEL_NAME
from databricks.agents.sdk_utils.deployments import _get_deployments
from databricks.agents.sdk_utils.entities import Deployment
from databricks.agents.utils.endpoint_utils import (
    _MONITOR_EXPERIMENT_ID_TAG,
    _get_endpoint_experiment_id_opt,
)
from databricks.agents.utils.mlflow_utils import (
    _check_model_is_rag_compatible,
    _get_latest_model_version,
    _is_mlflow_3_or_above,
    get_databricks_uc_registry_uri,
    get_model_info,
    get_workspace_url,
)
from databricks.agents.utils.uc import (
    _check_model_name,
    _escape_uc_name,
    _get_catalog_and_schema,
    _sanitize_model_name,
)
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors.platform import (
    BadRequest,
    InvalidParameterValue,
    PermissionDenied,
    ResourceConflict,
    ResourceDoesNotExist,
)
from databricks.sdk.service import catalog
from databricks.sdk.service.serving import (
    AiGatewayConfig,
    AiGatewayInferenceTableConfig,
    EndpointCoreConfigInput,
    EndpointCoreConfigOutput,
    EndpointPendingConfig,
    EndpointTag,
    Route,
    ServedEntityInput,
    ServedEntityOutput,
    ServingEndpointDetailed,
    TrafficConfig,
)

_logger = logging.getLogger("agents")

__DEPLOY_ENV_VARS_WITH_STATIC_VALUES = {
    "ENABLE_LANGCHAIN_STREAMING": "true",
    "ENABLE_MLFLOW_TRACING": "true",
    "RETURN_REQUEST_ID_IN_RESPONSE": "true",
}
_MLFLOW_EXPERIMENT_ID_ENV_VAR = "MLFLOW_EXPERIMENT_ID"
_MAX_ENDPOINT_NAME_LEN = 63
_MAX_SERVED_ENTITY_NAME_LEN = 63


def _compute_tag_updates(
    current_tags: List[EndpointTag],
    user_tags: Optional[Dict[str, str]],
    experiment_id_for_tracing: Optional[str],
) -> tuple[List[EndpointTag], List[str]]:
    """
    Compute which tags to add and remove for an existing endpoint.

    Args:
        current_tags: Current tags on the endpoint
        user_tags: User-provided tags to set
        experiment_id_for_tracing: Experiment ID for monitoring (if available)

    Returns:
        Tuple of (tags_to_add, tags_to_remove)
    """
    tags_to_add = []
    tags_to_remove = []

    curr_tags_dict = {t.key: t.value for t in current_tags or []}

    # Handle user-provided tags
    if user_tags is not None:
        user_tags_set = set((key, value) for key, value in user_tags.items())
        existing_user_tags = {(k, v) for k, v in curr_tags_dict.items() if k != _MONITOR_EXPERIMENT_ID_TAG}

        tags_to_add.extend(EndpointTag(key, value) for key, value in user_tags_set - existing_user_tags)
        tags_to_remove.extend(key for key, _ in existing_user_tags - user_tags_set)

    # Handle MONITOR_EXPERIMENT_ID tag - add if missing and experiment ID is available
    # But don't add it if the user explicitly provided it in their tags
    if (
        _MONITOR_EXPERIMENT_ID_TAG not in curr_tags_dict
        and experiment_id_for_tracing is not None
        and (user_tags is None or _MONITOR_EXPERIMENT_ID_TAG not in user_tags)
    ):
        tags_to_add.append(EndpointTag(_MONITOR_EXPERIMENT_ID_TAG, experiment_id_for_tracing))

    return tags_to_add, tags_to_remove


def _validate_environment_vars(environment_vars: Dict[str, str]) -> None:
    # If environment_vars is not a dictionary, raise an error
    if not isinstance(environment_vars, dict):
        raise ValueError("Argument 'environment_vars' must be a dictionary.")

    errors = []
    for key, value in environment_vars.items():
        # Environment variable names must be uppercase and can only contain letters, numbers, or underscores
        if not isinstance(key, str) or not key.isupper() or not key.isidentifier():
            errors.append(
                f"Environment variable ({key}) is not a valid identifier. Allowed characters are uppercase letters, numbers or underscores. An environment variable cannot start with a number."
            )

        # Environment variable values must be strings
        if not isinstance(value, str):
            errors.append(f"Invalid environment variable. Both key ({key}) and value ({value}) must be strings.")

        # Environment variable values cannot override default values for Agents
        if key in __DEPLOY_ENV_VARS_WITH_STATIC_VALUES and value != __DEPLOY_ENV_VARS_WITH_STATIC_VALUES[key]:
            errors.append(f"Environment variable ({key}) cannot be set to value ({value}).")

    if len(errors) > 0:
        raise ValueError("\n".join(errors))


def get_deployments(model_name: str, model_version: Optional[int] = None) -> List[Deployment]:
    """
    Get chain deployments metadata.

    Args:
        model_name: Name of the UC registered model
        model_version: (Optional) Version numbers for specific agents.

    Returns:
        All deployments for the UC registered model.
    """
    return _get_deployments(model_name, model_version)


def _create_served_model_input(
    model_name,
    version,
    scale_to_zero,
    environment_vars,
    served_entity_name,
    instance_profile_arn=None,
    workload_size="Small",
):
    return ServedEntityInput(
        name=served_entity_name,
        entity_name=model_name,
        entity_version=version,
        workload_size=workload_size,
        scale_to_zero_enabled=scale_to_zero,
        environment_vars=environment_vars,
        instance_profile_arn=instance_profile_arn,
    )


def _create_endpoint_name(model_name):
    prefix = "agents_"
    truncated_model_name = model_name[: _MAX_ENDPOINT_NAME_LEN - len(prefix)]
    sanitized_truncated_model_name = _sanitize_model_name(truncated_model_name)
    return f"agents_{sanitized_truncated_model_name}"


def _create_served_model_name(model_name, version):
    model_version_suffix = f"_{version}"
    truncated_model_name = model_name[: _MAX_SERVED_ENTITY_NAME_LEN - len(model_version_suffix)]
    sanitized_truncated_model_name = _sanitize_model_name(truncated_model_name)
    return f"{sanitized_truncated_model_name}{model_version_suffix}"


def _create_feedback_model_name(model_name: str) -> str:
    catalog_name, schema_name = _get_catalog_and_schema(model_name)
    return f"{catalog_name}.{schema_name}.{_FEEDBACK_MODEL_NAME}"


def _set_up_feedback_model_permissions(feedback_uc_model_name: str) -> None:
    workspace_client = WorkspaceClient()
    permission_changes = [
        catalog.PermissionsChange(
            principal="account users",
            add=[catalog.Privilege.EXECUTE],
        )
    ]
    try:
        # Handle databricks-sdk >= 0.56 API
        workspace_client.grants.update(
            securable_type="FUNCTION",
            full_name=feedback_uc_model_name,
            changes=permission_changes,
        )
    except Exception:
        try:
            # Handle databricks-sdk < 0.56 API
            workspace_client.grants.update(
                securable_type=catalog.SecurableType.FUNCTION,
                full_name=feedback_uc_model_name,
                changes=permission_changes,
            )
        except Exception:
            from pyspark.sql import SparkSession

            # Fall back to legacy logic that uses a SQL command to
            # grant permissions (only works on Databricks compute, etc)
            escaped_feedback_name = _escape_uc_name(feedback_uc_model_name)
            spark = SparkSession.builder.getOrCreate()
            spark.sql(f"GRANT EXECUTE ON FUNCTION {escaped_feedback_name} TO `account users`;")


def _log_feedback_model(feedback_uc_model_name: str) -> None:
    input_schema = Schema(
        [
            ColSpec(DataType.string, "request_id"),
            ColSpec(DataType.string, "source"),
            ColSpec(DataType.string, "text_assessments"),
            ColSpec(DataType.string, "retrieval_assessments"),
        ]
    )
    output_schema = Schema([ColSpec(DataType.string, "result")])
    input_signature = ModelSignature(inputs=input_schema, outputs=output_schema)

    mlflow.set_registry_uri(get_databricks_uc_registry_uri())
    input_example = {
        "request_id": "1",
        "source": "random_source",
        "text_assessments": "text_assessments",
        "retrieval_assessments": "retrieval_assessments",
    }
    with mlflow.start_run(run_name="feedback-model"):
        return mlflow.pyfunc.log_model(
            artifact_path=_FEEDBACK_MODEL_NAME,
            signature=input_signature,
            loader_module="feedback",
            pip_requirements=[
                "mlflow",
            ],
            code_paths=[databricks.agents.feedback.__file__],
            registered_model_name=feedback_uc_model_name,
            input_example=input_example,
        )


def _create_feedback_model(feedback_uc_model_name: str, scale_to_zero: bool) -> ServedEntityInput:
    # only create the feedback model if it doesn't already exist in this catalog.schema
    feedback_model_version = str(_get_latest_model_version(feedback_uc_model_name))
    if feedback_model_version == "0":
        # also adds to UC with version '1'
        _log_feedback_model(feedback_uc_model_name)
        feedback_model_version = str(_get_latest_model_version(feedback_uc_model_name))
        _set_up_feedback_model_permissions(feedback_uc_model_name)
    return _create_served_model_input(
        model_name=feedback_uc_model_name,
        version=feedback_model_version,
        served_entity_name=_FEEDBACK_MODEL_NAME,
        scale_to_zero=scale_to_zero,
        environment_vars=None,
    )


def _create_feedback_model_config(
    uc_model_name: str,
    pending_config: EndpointPendingConfig,
    scale_to_zero: bool = False,
) -> EndpointCoreConfigOutput:
    """
    Parse pending_config to get additional information about the feedback model in order to
    return a config as if the endpoint was successfully deployed with only the feedback model.
    This way we can reuse the update functions that are written for normal endpoint updates.
    """
    feedback_models = []
    feedback_routes = []
    feedback_uc_model_name = _create_feedback_model_name(uc_model_name)

    # Try to find a feedback in pending configs
    found_feedback_model = False
    for model in pending_config.served_entities:
        if model.name == _FEEDBACK_MODEL_NAME:
            found_feedback_model = True
            feedback_models = [
                _create_served_model_input(
                    model_name=feedback_uc_model_name,
                    version=model.entity_version,
                    scale_to_zero=model.scale_to_zero_enabled,
                    environment_vars=None,
                    served_entity_name=_FEEDBACK_MODEL_NAME,
                )
            ]
            feedback_routes = [Route(served_model_name=_FEEDBACK_MODEL_NAME, traffic_percentage=0)]
            break

    # If pending configs does not have a feedback model, create a new one
    if not found_feedback_model:
        feedback_models = [
            _create_feedback_model(feedback_uc_model_name, scale_to_zero),
        ]
        feedback_routes = [
            Route(
                served_model_name=_FEEDBACK_MODEL_NAME,
                traffic_percentage=0,
            ),
        ]

    return EndpointCoreConfigOutput(
        served_entities=feedback_models,
        traffic_config=TrafficConfig(routes=feedback_routes),
        auto_capture_config=pending_config.auto_capture_config,
    )


def _construct_table_name(catalog_name, schema_name, model_name):
    w = WorkspaceClient()
    # remove catalog and schema from model_name and add agents- prefix
    base_name = model_name.split(".")[2]
    suffix = ""

    # try to append suffix
    for index in range(20):
        if index != 0:
            suffix = f"_{index}"

        table_name = f"{base_name[:63 - len(suffix)]}{suffix}"

        full_name = f"{catalog_name}.{schema_name}.{table_name}_payload"
        if not w.tables.exists(full_name=full_name).table_exists:
            return table_name

    # last attempt - append uuid and truncate to 63 characters (max length for table_name_prefix)
    # unlikely to have conflict unless base_name is long
    if len(base_name) > 59:
        return f"{base_name[:59]}_{uuid.uuid4().hex}"[:63]
    return f"{base_name}_{uuid.uuid4().hex}"[:63]


def _create_new_endpoint_config(
    model_name,
    version,
    endpoint_name,
    scale_to_zero=False,
    environment_vars=None,
    instance_profile_arn=None,
    workload_size=None,
    deploy_feedback_model=False,
):
    served_model_name = _create_served_model_name(model_name, version)

    # Build the served entities list
    served_entities = [
        _create_served_model_input(
            model_name=model_name,
            version=version,
            scale_to_zero=scale_to_zero,
            environment_vars=environment_vars,
            served_entity_name=served_model_name,
            instance_profile_arn=instance_profile_arn,
            workload_size=workload_size,
        )
    ]

    # Build the traffic routes list
    routes = [
        Route(
            served_model_name=served_model_name,
            traffic_percentage=100,
        )
    ]

    # Only create feedback model if deploy_feedback_model is True
    if deploy_feedback_model:
        feedback_uc_model_name = _create_feedback_model_name(model_name)
        served_entities.append(_create_feedback_model(feedback_uc_model_name, scale_to_zero))
        routes.append(
            Route(
                served_model_name=_FEEDBACK_MODEL_NAME,
                traffic_percentage=0,
            )
        )

    return EndpointCoreConfigInput(
        name=endpoint_name,
        served_entities=served_entities,
        traffic_config=TrafficConfig(routes=routes),
        auto_capture_config=None,
    )


def _create_ai_gateway_config(model_name: str):
    catalog_name, schema_name = _get_catalog_and_schema(model_name)
    table_name = _construct_table_name(catalog_name, schema_name, model_name)

    return AiGatewayConfig(
        inference_table_config=AiGatewayInferenceTableConfig(
            enabled=True,
            catalog_name=catalog_name,
            schema_name=schema_name,
            table_name_prefix=table_name,
        )
    )


def _update_traffic_config(
    model_name: str, version: str, existing_config: EndpointCoreConfigOutput, deploy_feedback_model: bool
) -> TrafficConfig:
    served_model_name = _create_served_model_name(model_name, version)
    updated_routes = [Route(served_model_name=served_model_name, traffic_percentage=100)]

    found_feedback_model = False
    if existing_config:
        for traffic_config in existing_config.traffic_config.routes:
            if traffic_config.served_model_name == _FEEDBACK_MODEL_NAME:
                found_feedback_model = True
            updated_routes.append(
                Route(
                    served_model_name=traffic_config.served_model_name,
                    traffic_percentage=0,
                )
            )
    if not found_feedback_model and deploy_feedback_model:
        updated_routes.append(
            Route(
                served_model_name=_FEEDBACK_MODEL_NAME,
                traffic_percentage=0,
            )
        )
    return TrafficConfig(routes=updated_routes)


def _update_served_models(
    model_name: str,
    version: str,
    endpoint_name: str,
    existing_config: EndpointCoreConfigOutput,
    scale_to_zero: bool,
    environment_vars: Dict[str, str],
    instance_profile_arn: str,
    workload_size="Small",
    deploy_feedback_model=False,
) -> List[ServedEntityInput]:
    served_model_name = _create_served_model_name(model_name, version)
    updated_served_models = [
        _create_served_model_input(
            model_name=model_name,
            version=version,
            served_entity_name=served_model_name,
            scale_to_zero=scale_to_zero,
            environment_vars=environment_vars,
            instance_profile_arn=instance_profile_arn,
            workload_size=workload_size,
        )
    ]

    found_feedback_model = False
    if existing_config:
        for served_model in existing_config.served_entities:
            if served_model.name == _FEEDBACK_MODEL_NAME:
                found_feedback_model = True
        updated_served_models.extend(existing_config.served_entities)

    if not found_feedback_model and deploy_feedback_model:
        updated_served_models.append(_create_feedback_model(_create_feedback_model_name(model_name), scale_to_zero))
    return updated_served_models


def _update_traffic_config_for_delete(
    updated_served_models: List[ServedEntityOutput], has_feedback_model: bool = True
) -> TrafficConfig:
    updated_routes = []

    # Find the highest version
    max_version_served_model = max(updated_served_models, key=lambda sm: int(sm.entity_version))
    max_version = max_version_served_model.entity_version

    # All routes have traffic_percentage=0 except the new highest version
    for served_model in updated_served_models:
        traffic_percentage = 0
        if served_model.entity_version == max_version:
            traffic_percentage = 100
        updated_routes.append(
            Route(
                served_model_name=served_model.name,
                traffic_percentage=traffic_percentage,
            )
        )

    if has_feedback_model:
        # Append route for feedback model
        updated_routes.append(
            Route(
                served_model_name=_FEEDBACK_MODEL_NAME,
                traffic_percentage=0,
            ),
        )

    return TrafficConfig(routes=updated_routes)


def _construct_query_endpoint(workspace_url, endpoint_name, model_name, version):
    # This is a temporary solution until we can identify the appropriate solution to get
    # the workspace URI in backend. Ref: https://databricks.atlassian.net/browse/ML-39391
    served_model_name = _create_served_model_name(model_name, version)
    base_url = f"{workspace_url}/serving-endpoints/{endpoint_name}/served-models/{served_model_name}/invocations"
    workspace_id = get_workspace_id()
    if workspace_id is not None:
        base_url += f"?o={workspace_id}"
    return base_url


# Retry at most 3 times, waiting 3 seconds in between tries.
# If all tries fail, reraise exception.
@retry(reraise=True, stop=stop_after_attempt(3), wait=wait_fixed(3))
def _get_endpoint_with_retry(
    client: WorkspaceClient,
    endpoint_name: str,
) -> ServingEndpointDetailed:
    """Retrieve the serving endpoint with retry.

    Args:
        client (WorkspaceClient): Databricks workspace client.
        endpoint_name (str): Name of the agent serving endpoint.

    Returns:
        ServingEndpointDetailed: The serving endpoint.
    """
    return client.serving_endpoints.get(endpoint_name)


def _get_experiment_id_for_tracing(serving_endpoint_opt):
    # Try to get the experiment ID to use from tracing from the monitor
    # attached to the serving endpoint, if the serving endpoint already exists and
    # monitoring is enabled
    if serving_endpoint_opt and (
        serving_endpoint_experiment_id := _get_endpoint_experiment_id_opt(serving_endpoint=serving_endpoint_opt)
    ):
        return serving_endpoint_experiment_id

    # Fall back to using the current active experiment for monitoring
    # In a Databricks Notebook, this will be set to the default notebook experiment or a specific experiment set by the user
    # In a non databricks environment, this can either be a specific experiment set by the user or will be None or 0
    current_active_experiment_id = mlflow.tracking.fluent._get_experiment_id()
    if current_active_experiment_id and current_active_experiment_id != DEFAULT_EXPERIMENT_ID:
        return str(current_active_experiment_id)


def deploy(
    model_name: str,
    model_version: int,
    scale_to_zero: bool = False,
    environment_vars: Dict[str, str] = None,
    instance_profile_arn: str = None,
    tags: Dict[str, str] = None,
    workload_size: str = "Small",
    endpoint_name: str = None,
    budget_policy_id: str = None,
    description: Optional[str] = None,
    deploy_feedback_model: bool = False,
    usage_policy_id: str = None,
    **kwargs,
) -> Deployment:
    """
    Deploy new version of the agent.

    Args:
        model_name: Name of the UC registered model.
        model_version: Model version number.
        scale_to_zero: Flag to scale the endpoint to zero when not in use. With scale to zero, \
            the compute resources may take time to come up so the app may not be ready instantly. Defaults to False.
        environment_vars: Dictionary of environment variables used to provide configuration for the endpoint. Defaults to {}.
        instance_profile_arn: Instance profile ARN to use for the endpoint. Defaults to None.
        tags: Dictionary of tags to attach to the deployment. Defaults to None.
        endpoint_name: Name of the agent serving endpoint to deploy to. If unspecified, an agent endpoint name will be
                       generated. The agent endpoint name must be the same across all model versions for a particular
                       agent model_name.
        budget_policy_id: **DEPRECATED.** Use ``usage_policy_id`` instead. ID of the serverless budget policy to associate
                          with the endpoint. Only honored the first time a particular agent endpoint is deployed; will be
                          ignored on subsequent calls to deploy() against the same endpoint.
        description: Description of the endpoint. When updating an existing endpoint, this defaults to None (i.e., no update),
                     but when creating a new endpoint, this defaults to an empty string.
        deploy_feedback_model: Whether to deploy the feedback model alongside the agent. Defaults to True.
        usage_policy_id: ID of the serverless usage policy to associate with the endpoint. Only honored the first time
                         a particular agent endpoint is deployed; will be ignored on subsequent calls to deploy() against
                         the same endpoint. If both ``usage_policy_id`` and ``budget_policy_id`` are provided,
                         ``usage_policy_id`` takes precedence.
    Returns:
        Chain deployment metadata.
    """
    if budget_policy_id is not None:
        warnings.warn(
            "The 'budget_policy_id' parameter is deprecated and will be removed in a future release. "
            "Please use 'usage_policy_id' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # If usage_policy_id is not provided, use budget_policy_id for backward compatibility
        if usage_policy_id is None:
            usage_policy_id = budget_policy_id

    model_info = get_model_info(model_name, model_version)
    _check_model_is_rag_compatible(model_info)
    _check_model_name(model_name)
    if not _is_mlflow_3_or_above(model_info.mlflow_version):
        print(
            f"The deployed agent depends on MLflow {model_info.mlflow_version}. "
            "You should upgrade to MLflow 3 for newest capabilities such as "
            "real-time tracing support. Some legacy features such as request logs and assessment logs may no longer work. "
            "See https://docs.databricks.com/aws/en/generative-ai/agent-framework/deploy-agent#deploy-an-agent-using-deploy for more information."
        )
    if deploy_feedback_model:
        warnings.warn(
            "The feedback model is deprecated and will longer be supported. Please:\n"
            "  1. Upgrade to MLflow 3.0 or higher for real-time tracing support\n"
            "  2. Use the MLflow 3 assessments API instead of the legacy feedback model\n"
            "  3. Set deploy_feedback_model=False in your deploy() calls\n"
            "For migration guidance, see: https://docs.databricks.com/aws/en/generative-ai/agent-framework/feedback-model"
        )
    else:
        warnings.warn(
            "This endpoint is being deployed without a feedback model, which has been deprecated.\n"
            "For more information, see: https://docs.databricks.com/aws/en/generative-ai/agent-framework/feedback-model"
        )
    if endpoint_name is None:
        endpoint_name = _create_endpoint_name(model_name)
    scale_to_zero = kwargs.get("scale_to_zero", scale_to_zero)
    user_env_vars = kwargs.get("environment_vars", environment_vars if environment_vars is not None else {})
    _validate_environment_vars(user_env_vars)
    instance_profile_arn = kwargs.get("instance_profile_arn", instance_profile_arn)
    tags = kwargs.get("tags", tags)
    workload_size = kwargs.get("workload_size", workload_size)

    environment_vars = {}
    environment_vars.update(user_env_vars)
    environment_vars.update(__DEPLOY_ENV_VARS_WITH_STATIC_VALUES)

    w = WorkspaceClient()
    try:
        serving_endpoint_opt = w.serving_endpoints.get(endpoint_name)
    except ResourceDoesNotExist:
        serving_endpoint_opt = None

    # If the user has not set the MLFLOW_EXPERIMENT_ID explicitly, then try to infer it from the current environment.
    experiment_id_for_tracing_opt = environment_vars.get(
        _MLFLOW_EXPERIMENT_ID_ENV_VAR
    ) or _get_experiment_id_for_tracing(serving_endpoint_opt)
    if _MLFLOW_EXPERIMENT_ID_ENV_VAR not in environment_vars and experiment_id_for_tracing_opt is not None:
        environment_vars[_MLFLOW_EXPERIMENT_ID_ENV_VAR] = experiment_id_for_tracing_opt

    if experiment_id_for_tracing_opt is None:
        _logger.warning(
            "Could not find active experiment or determine experiment ID for "
            "tracing from environment variables. Traces from this deployment will not be "
            "logged real-time to MLflow or monitored. Please ensure you are using MLflow 3.0 or above, "
            "then set an active experiment using mlflow.set_experiment() before "
            "agents.deploy() in order to log traces from this deployment to MLflow for real-time monitoring."
        )

    endpoint_tags = [EndpointTag(key, val) for key, val in tags.items()] if tags else []
    # Set the MONITOR_EXPERIMENT_ID tag if the experiment ID for tracing is available
    # and the user didn't explicitly specify a tag value themselves
    if (not tags or _MONITOR_EXPERIMENT_ID_TAG not in tags) and experiment_id_for_tracing_opt is not None:
        endpoint_tags.append(EndpointTag(_MONITOR_EXPERIMENT_ID_TAG, experiment_id_for_tracing_opt))

    if serving_endpoint_opt is None:
        description = description or ""
        w.serving_endpoints.create(
            name=endpoint_name,
            config=_create_new_endpoint_config(
                model_name,
                model_version,
                endpoint_name,
                scale_to_zero,
                environment_vars,
                instance_profile_arn,
                workload_size,
                deploy_feedback_model,
            ),
            budget_policy_id=usage_policy_id,
            ai_gateway=_create_ai_gateway_config(model_name),
            tags=endpoint_tags,
            description=description,
        )
    else:
        config = serving_endpoint_opt.config
        # TODO: https://databricks.atlassian.net/browse/ML-39649
        # config=None means this endpoint has never successfully deployed before
        # bc we have a dummy feedback model, we know feedback works, so we only want its config
        if config is None and deploy_feedback_model:
            config = _create_feedback_model_config(model_name, serving_endpoint_opt.pending_config, scale_to_zero)

        # ignore pending_config bc we only redeploy models that have successfully deployed before
        # set the traffic config for all currently deployed models to be 0
        updated_traffic_config = _update_traffic_config(model_name, model_version, config, deploy_feedback_model)
        updated_served_models = _update_served_models(
            model_name,
            model_version,
            endpoint_name,
            config,
            scale_to_zero,
            environment_vars,
            instance_profile_arn,
            workload_size,
            deploy_feedback_model,
        )

        # Update endpoint tags if needed (user tags or missing MONITOR_EXPERIMENT_ID tag)
        tags_to_add, tags_to_remove = _compute_tag_updates(
            current_tags=serving_endpoint_opt.tags,
            user_tags=tags,
            experiment_id_for_tracing=experiment_id_for_tracing_opt,
        )

        if tags_to_add or tags_to_remove:
            w.serving_endpoints.patch(
                name=endpoint_name,
                add_tags=tags_to_add or None,
                delete_tags=tags_to_remove or None,
            )
        if description is not None and serving_endpoint_opt.description != description:
            body = {"name": endpoint_name, "description": description}
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
            w.api_client.do(
                method="PATCH",
                path=f"/api/2.0/serving-endpoints/{endpoint_name}/description",
                body=body,
                headers=headers,
            )
        if usage_policy_id is not None and serving_endpoint_opt.budget_policy_id != usage_policy_id:
            body = {"name": endpoint_name, "budget_policy_id": usage_policy_id}
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
            w.api_client.do(
                method="PATCH",
                path=f"/api/2.0/serving-endpoints/{endpoint_name}/budget-policy",
                body=body,
                headers=headers,
            )

        try:
            # Here we don't need to update the ai-gateway config since using the put_ai_gateway throws an error
            # "Inference tables is already enabled". Also the ai-gateway is never part of the pending config,
            # it is associated with the endpoint directly.
            # Read https://github.com/databricks-eng/universe/pull/874814#discussion_r1920845945 for more info
            w.serving_endpoints.update_config(
                name=endpoint_name,
                served_entities=updated_served_models,
                traffic_config=updated_traffic_config,
                auto_capture_config=getattr(config, "auto_capture_config", None),
            )
        except ResourceConflict:
            raise ValueError(f"Endpoint {endpoint_name} is currently updating.")
        except InvalidParameterValue as e:
            if "served_models cannot contain more than 15 elements" in str(e):
                raise ValueError(
                    f"Endpoint {endpoint_name} already has 15 deployed models. Delete one before redeploying."
                )
            else:
                # pass through any other errors
                raise e
        except BadRequest as e:
            if "Cannot create 2+ served entities" in str(e):
                raise ValueError(
                    f"Endpoint {endpoint_name} already serves model {model_name}, version {model_version}."
                )
            else:
                raise e

    workspace_url = get_workspace_url()
    deployment_info = rest_deploy_chain(
        model_name=model_name,
        model_version=model_version,
        query_endpoint=_construct_query_endpoint(workspace_url, endpoint_name, model_name, model_version),
        endpoint_name=endpoint_name,
        served_entity_name=_create_served_model_name(model_name, model_version),
        workspace_url=workspace_url,
    )

    user_message = f"""
    Deployment of {deployment_info.model_name} version {model_version} initiated.  This can take up to 15 minutes and the Review App & Query Endpoint will not work until this deployment finishes.

    View status: {deployment_info.endpoint_url}
    Review App: {deployment_info.review_app_url}"""

    ## TODO (ML-42186) - Change this to Logger
    print(user_message)

    print(f"\nYou can refer back to the links above from the endpoint detail page at {deployment_info.endpoint_url}.")

    print("\nTo set up monitoring for your deployed agent, see:")
    print("https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/production-monitoring")

    return deployment_info


def list_deployments() -> List[Deployment]:
    """
    Returns:
      A list of all Agent deployments.
    """
    return rest_list_chain_deployments()


def _delete_endpoint(workspace_client: WorkspaceClient, endpoint_name: str, model_name: str) -> None:
    # First, delete the classic monitor associated with the endpoint, if any
    # exists
    try:
        # required to avoid eval_context circular dependency issue
        from databricks.rag_eval.monitoring.api import (
            _get_monitor_internal,
            delete_monitor,
        )

        # we make this call to differentiate between failed delete_monitor
        # calls due to no monitor vs failed calls due to other issues
        _get_monitor_internal(endpoint_name=endpoint_name)

        try:
            delete_monitor(endpoint_name=endpoint_name)
        except Exception:
            _logger.warning(
                (
                    f"Failed to delete monitor for serving endpoint '{endpoint_name}'. ",
                    "If you think this is an error, you can try deleting the monitor manually ",
                    "by running `delete_monitor`.",
                ),
                exc_info=True,
            )
    except Exception:
        ...
    # Then, delete the serving endpoint
    try:
        workspace_client.serving_endpoints.delete(endpoint_name)
    except PermissionDenied:
        raise PermissionDenied(f"User does not have permission to delete deployments for model {model_name}.")


def delete_deployment(model_name: str, model_version: Optional[int] = None) -> None:
    """
    Delete an agent deployment.

    Also deletes the associated monitor.

    Args:
        model_name: Name of UC registered model
        version: Model version number. This is optional and when specified, we delete \
            the served model for the particular version.
    """
    _check_model_name(model_name)
    deployments = get_deployments(model_name=model_name, model_version=model_version)
    if len(deployments) == 0:
        raise ValueError(f"Deployments for model {model_name} do not exist.")
    endpoint_name = deployments[0].endpoint_name

    w = WorkspaceClient()
    endpoint = None
    try:
        endpoint = w.serving_endpoints.get(endpoint_name)
    except ResourceDoesNotExist:
        pass
    except PermissionDenied:
        raise PermissionDenied(f"User does not have permission to delete deployments for model {model_name}.")

    if endpoint:
        if model_version is None:
            _delete_endpoint(w, endpoint_name, model_name)
        else:
            # If model_version is specified, then remove only the served model, leave other served entities
            config = endpoint.config
            # Config=None means the deployment never successfully deployed before. In this
            # case, we raise an error that specified model version is in a failed state or
            # does not exist.
            if config is None:
                raise ValueError(
                    f"The deployment for model version {model_version} is in a failed state or does not exist."
                )

            # Check against expected versions in the database to be resilient to endpoint
            # update failures.
            versions = set([deployment.model_version for deployment in _get_deployments(model_name)])
            updated_served_models = [
                served_model
                for served_model in config.served_entities
                if not (
                    # Filter out the specified model version.
                    served_model.entity_name == model_name and served_model.entity_version == str(model_version)
                )
                and (
                    # Filter out any versions that do not exist in the database.
                    served_model.name == _FEEDBACK_MODEL_NAME or served_model.entity_version in versions
                )
            ]

            if len(config.served_entities) == len(updated_served_models):
                raise ValueError(f"The deployment for model version {model_version} does not exist.")
            updated_served_models_without_feedback = [
                served_model
                for served_model in updated_served_models
                if not (served_model.name == _FEEDBACK_MODEL_NAME)
            ]

            if len(updated_served_models_without_feedback) == 0:
                # If there are no more served models remaining, delete the endpoint
                _delete_endpoint(w, endpoint_name, model_name)
            else:
                has_feedback_model = len(updated_served_models) != len(updated_served_models_without_feedback)
                updated_traffic_config = _update_traffic_config_for_delete(
                    updated_served_models_without_feedback, has_feedback_model
                )
                try:
                    w.serving_endpoints.update_config(
                        name=endpoint_name,
                        served_entities=updated_served_models,
                        traffic_config=updated_traffic_config,
                        auto_capture_config=config.auto_capture_config,
                    )
                except PermissionDenied:
                    raise PermissionDenied(
                        f"User does not have permission to delete deployments for model {model_name}. "
                        f"Deployment for model version {model_version} was not deleted."
                    )
                except Exception as e:
                    raise Exception(f"Failed to delete deployment for model {model_name} version {model_version}. {e}")

    # ToDo[ML-42212]: Move this rest call above deleting endpoints after permissions are implemented in the backend
    rest_delete_chain(model_name, model_version)
