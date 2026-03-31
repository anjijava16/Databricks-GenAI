import json
from typing import List, Optional

import mlflow

from databricks.agents.client.rest_client import (
    create_review_artifacts as rest_create_review_artifacts,
)
from databricks.agents.client.rest_client import (
    get_review_instructions as rest_get_review_instructions,
)
from databricks.agents.client.rest_client import (
    set_review_instructions as rest_set_review_instructions,
)
from databricks.agents.permissions import get_permissions, set_permissions
from databricks.agents.sdk_utils.deployments import get_latest_chain_deployment
from databricks.agents.sdk_utils.entities import PermissionLevel
from databricks.sdk import WorkspaceClient

_TRACES_FILE_PATH = "traces.json"
_DATABRICKS_OPTIONS = "databricks_options"
_CONVERSATION_ID = "conversation_id"
_REQUEST_ID = "request_id"
_TIMESTAMP = "timestamp"
_DATABRICKS_REQUEST_ID = "databricks_request_id"
MLFLOW_TRACE_SCHEMA_VERSION = "mlflow.trace_schema.version"


def _get_inference_table_name(serving_endpoint_name, model_name):
    w = WorkspaceClient()
    serving_endpoint = w.serving_endpoints.get(serving_endpoint_name)
    if auto_capture_config := serving_endpoint.config.auto_capture_config:
        catalog_name = auto_capture_config.catalog_name
        schema_name = auto_capture_config.schema_name
        table_name = auto_capture_config.state.payload_table.name
        return f"`{catalog_name}`.`{schema_name}`.`{table_name}`"
    if ai_gateway := serving_endpoint.ai_gateway:
        if inference_table_config := ai_gateway.inference_table_config:
            catalog_name = inference_table_config.catalog_name
            schema_name = inference_table_config.schema_name
            table_name_prefix = inference_table_config.table_name_prefix
            return f"`{catalog_name}`.`{schema_name}`.`{table_name_prefix}_payload`"

    raise ValueError(
        f"The provided {model_name} doesn't have any inference table configured. "
        "Please update the endpoint to capture payloads to an inference table"
    )


def _generate_review_experiment_name(model_name):
    w = WorkspaceClient()
    current_user = w.current_user.me().user_name
    return f"/Users/{current_user}/agents_reviews_{model_name}"


def _convert_split_completion_input_to_chat_completion_input(request_json):
    # Transform a SplitCompletions input to ChatCompletions Input
    # TODO: rework this code once extensible schemas are supported
    query = request_json.pop("query")
    history = request_json.pop("history")
    request_json["messages"] = history + [{"role": "user", "content": query}]


def _convert_str_object_output_to_chat_completion_output(output_json):
    # Transform a StrObject output to ChatCompletions Output
    # TODO: rework this code once extensible schemas are supported
    output_json.update({"choices": [{"message": {"content": output_json.pop("content")}}]})


def _parse_request_logs(request_logs):
    def transform_request_entry(request_logs_row):
        request_json = json.loads(request_logs_row["request"])
        request_json[_CONVERSATION_ID] = request_json.get(_DATABRICKS_OPTIONS, {}).get(_CONVERSATION_ID, "")
        request_json[_REQUEST_ID] = request_logs_row.get(_DATABRICKS_REQUEST_ID, "")
        request_json[_TIMESTAMP] = request_logs_row.get(_TIMESTAMP, "")

        if "query" in request_json and "history" in request_json:
            # If the request is a SplitCompletions request, convert inplace to ChatCompletions
            _convert_split_completion_input_to_chat_completion_input(request_json)

        return request_json

    request_logs["request"] = request_logs.apply(transform_request_entry, axis=1)

    def transform_output_entry(request_logs_row):
        output_json = json.loads(request_logs_row["response"])
        if "content" in output_json:
            # If the response is a StrObject response, convert inplace to ChatCompletions
            _convert_str_object_output_to_chat_completion_output(output_json)
        return output_json

    request_logs["output"] = request_logs.apply(transform_output_entry, axis=1)
    request_logs = request_logs.rename(columns={_DATABRICKS_REQUEST_ID: _REQUEST_ID})
    request_logs = request_logs.drop(columns=["response"], axis=1)

    if "client_request_id" in request_logs:
        request_logs = request_logs.drop(columns=["client_request_id"], axis=1)

    return request_logs


def retrieve_payload_logs(spark_df):
    from pyspark.sql import functions as F

    request_logs = spark_df.filter(
        F.expr(
            "request:dataframe_records[0].text_assessments IS NULL and request:dataframe_records[0].retrieval_assessments IS NULL"
        )
    ).filter(F.col("status_code") == "200")  # Ignore Error requests

    if request_logs.isEmpty():
        return None

    """
    - If the trace object exists and if its V1 or V2 then don't follow this method and use unpacking logic as we need to be backward compatible
        - Filter by successful request logs and then check to se whether there are rows where databricks.trace.info is not null
        - Return True if that Dataframe is not empty
        - Return False if that dataframe is empty
    """
    request_logs_with_trace = request_logs.filter(F.expr("response:databricks_output.trace IS NOT NULL"))
    if not request_logs_with_trace.isEmpty():
        v2_request_logs = request_logs.filter(
            F.expr(f"response:databricks_output.trace['{MLFLOW_TRACE_SCHEMA_VERSION}']==2")
        )
        v1_request_logs = request_logs.filter(
            F.expr(
                f"response:databricks_output.trace['{MLFLOW_TRACE_SCHEMA_VERSION}'] IS NULL and response:databricks_output.trace.info IS NULL"
            )
        )
        if not v1_request_logs.isEmpty() or not v2_request_logs.isEmpty():
            return None

    # If trace object does not exist or if its mlflow trace don't follow unpacking logic
    timestamp_col = (
        F.col("request_time")  # Default to 'request_time' if 'timestamp_ms' is not present
        if "request_time" in request_logs.columns
        else F.timestamp_millis(F.col("timestamp_ms"))  # Use 'timestamp_ms' if present
    )
    return request_logs.withColumn("timestamp", timestamp_col).withColumn(
        "trace",
        F.coalesce(F.expr("response:databricks_output.trace"), F.lit("")),
    )


def enable_trace_reviews(model_name: str, request_ids: Optional[List[str]] = None) -> str:
    """
    Enable the reviewer UI to collect feedback on the conversations from the endpoint inference log.

    Args:
        model_name: The name of the UC Registered Model to use when
            registering the chain as a UC Model Version.
            Example: catalog.schema.model_name
        request_ids: Optional list of request_ids for which the feedback
            needs to be captured.

    :return: URL for the reviewer UI where users can start providing feedback

    Example:

    .. code-block:: python

        from databricks.agents import enable_trace_reviews

        enable_trace_reviews(
            model_name="catalog.schema.chain_model",
            request_ids=["490cf09b-6da6-474f-bc35-ee5ca688ff8", "a4d37810-5cd0-4cbd-aa25-e5ceaf6a448"],
        )

    """
    chain_deployment = get_latest_chain_deployment(model_name)
    serving_endpoint_name = chain_deployment.endpoint_name
    table_full_name = _get_inference_table_name(serving_endpoint_name=serving_endpoint_name, model_name=model_name)

    if request_ids:
        # cast id to int if other type is passed in
        request_ids_str = ", ".join([f"'{id}'" for id in request_ids])
        sql_query = f"SELECT * FROM {table_full_name} WHERE databricks_request_id IN ({request_ids_str})"
    else:
        sql_query = f"SELECT * FROM {table_full_name}"
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
    try:
        spark_df = spark.sql(sql_query)
        if converted_spark_df := retrieve_payload_logs(spark_df):
            df = converted_spark_df.toPandas()
            df = _parse_request_logs(df)
        else:
            raise ValueError(
                "The trace logged used for review is in an unsupported format. "
                "Please delete the endpoint and try again to gather newer format of the trace."
            )
    except Exception as e:
        raise ValueError(f"Failed to fetch the data from the table {table_full_name}. Error: {str(e)}") from e

    review_experiment_name = _generate_review_experiment_name(model_name)
    # get or create the review experiment
    review_experiment = mlflow.get_experiment_by_name(review_experiment_name)
    can_manage_users = []
    if review_experiment:
        review_experiment_id = review_experiment.experiment_id
    else:
        # Get Permissions
        current_permissions = get_permissions(model_name)
        # Filter and get users with can manage permissions
        can_manage_users = [
            permission_tuple[0]
            for permission_tuple in list(
                filter(
                    lambda permission_tuple: permission_tuple[1] == PermissionLevel.CAN_MANAGE,
                    current_permissions,
                )
            )
        ]
        review_experiment_id = mlflow.create_experiment(review_experiment_name)

    with mlflow.start_run(experiment_id=review_experiment_id) as model_run:
        mlflow.log_table(data=df, artifact_file=_TRACES_FILE_PATH)
        artifact_uri = f"runs:/{model_run.info.run_id}/{_TRACES_FILE_PATH}"
        rest_create_review_artifacts(model_name, artifacts=[artifact_uri])
        if len(can_manage_users) > 0:
            # Set Permissions on users again to set experiment permissions
            set_permissions(model_name, can_manage_users, PermissionLevel.CAN_MANAGE)

    return chain_deployment.review_app_url


def set_review_instructions(model_name: str, instructions: str) -> None:
    """
    Set the instructions for the review UI.

    Args:
        model_name: The name of the UC Registered Model to use when
            registering the chain as a UC Model Version.
        instructions: Instructions for the reviewer UI in markdown format.

    Example:

    .. code-block:: python

        from databricks.agents import set_review_instructions

        set_review_instructions(
            model_name="catalog.schema.chain_model",
            instructions="Please provide feedback on the conversations based on your knowledge of UC."
        )

    """
    rest_set_review_instructions(model_name, instructions)


def get_review_instructions(model_name: str) -> str:
    """
    Get the instructions for the review UI.

    Args:
    model_name: The name of the UC Registered Model to use when
        registering the chain as a UC Model Version.
        Example: catalog.schema.model_name

    Returns:
        Instructions for the reviewer UI in markdown format

    Example:

    .. code-block:: python

        from databricks.agents import get_review_instructions

        instructions = get_review_instructions(model_name="catalog.schema.chain_model")
        print(instructions)

    """
    return rest_get_review_instructions(model_name)
