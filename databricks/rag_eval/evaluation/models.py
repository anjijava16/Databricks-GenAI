"""
This module contains helper functions for invoking the model to be evaluated.
"""

import logging
import re
import uuid
from dataclasses import dataclass
from typing import Any, List, Optional

import mlflow
import mlflow.deployments
import mlflow.entities as mlflow_entities
import mlflow.pyfunc.context as pyfunc_context
import mlflow.pyfunc.model as pyfunc_model
import mlflow.tracing.fluent
import mlflow.utils.logging_utils

from databricks import agents
from databricks.rag_eval.evaluation import entities, traces
from databricks.rag_eval.mlflow import mlflow_utils
from databricks.rag_eval.utils import input_output_utils, trace_utils

_logger = logging.getLogger(__name__)


_FAIL_TO_GET_TRACE_WARNING_MSG = re.compile(r"Failed to get trace from the tracking store")


@dataclass
class ModelResult:
    """
    The result of invoking the model.
    """

    response: Optional[str] = None
    raw_model_output: Optional[Any] = None
    retrieval_context: Optional[entities.RetrievalContext] = None
    tool_calls: Optional[List[entities.ToolCallInvocation]] = None
    trace: Optional[mlflow_entities.Trace] = None
    error_message: Optional[str] = None


def invoke_model(
    model: mlflow.pyfunc.PyFuncModel,
    eval_item: entities.EvalItem,
) -> ModelResult:
    """
    Invoke the model with a request to get a model result.

    :param model: The model to invoke.
    :param eval_item: The eval item containing the request.
    :return: The model result.
    """
    model_result = ModelResult()
    # === Prepare the model input ===
    model_input = input_output_utils.to_chat_completion_request(eval_item.raw_request)
    if _is_agent_endpoint(model):
        # For agent endpoints, we set the flag to include trace in the model output
        model_input = input_output_utils.set_include_trace(model_input)

    # === Invoke the model and get the trace ===
    # Use a random UUID as the context ID to avoid conflicts with other evaluations on the same set of questions
    context_id = str(uuid.uuid4())
    with pyfunc_context.set_prediction_context(pyfunc_context.Context(context_id, is_evaluate=True)):
        try:
            model_result.raw_model_output = model.predict(model_input)
        except Exception as e:
            model_result.error_message = f"Fail to invoke the model with {model_input}. {e!r}"

    # Get the trace from the MLflow trace server
    with mlflow.utils.logging_utils.suppress_logs(mlflow.tracing.fluent.__name__, _FAIL_TO_GET_TRACE_WARNING_MSG):
        # If not found, mlflow.get_trace will return None
        model_result.trace = mlflow.get_trace(context_id)

    # Last resort: just forge a trace. Note: this should not happen with mlflow3 but we prefer to not
    # break the evaluation run by throwing an error.
    if model_result.trace is None:
        if model_result.error_message:
            model_result.trace = trace_utils.create_minimal_trace(
                request=model_input, response=model_result.error_message, status="ERROR"
            )
        else:
            model_result.trace = trace_utils.create_minimal_trace(
                request=model_input, response=model_result.raw_model_output
            )

    # === Parse the response from the raw model output ===
    if model_result.raw_model_output is not None:
        try:
            model_result.response = input_output_utils.response_to_string(model_result.raw_model_output)
        except ValueError:
            model_result.response = None

    # === Extract the retrieval context from the trace ===
    model_result.retrieval_context = traces.extract_retrieval_context_from_trace(model_result.trace)

    # Extract tool calls from the trace, or response if trace is not available.
    model_result.tool_calls = traces.extract_tool_calls(response=model_result.response, trace=model_result.trace)

    return model_result


def _is_model_endpoint_wrapper(model: Any) -> bool:
    """
    Check if the model is a wrapper of an endpoint.

    :param model: The model to check
    :return: True if the model is an endpoint wrapper
    """
    # noinspection PyProtectedMember
    return isinstance(model, pyfunc_model._PythonModelPyfuncWrapper) and isinstance(
        model.python_model, pyfunc_model.ModelFromDeploymentEndpoint
    )


def _is_agent_task(task_name: str) -> bool:
    """
    Check if the task name is an agent task.

    :param task_name: The name of the task
    :return: True if the task name is an agent task
    """
    return bool(re.match(r"agent/v\d+/chat", task_name))


def _is_agent_endpoint(model: Any) -> bool:
    if not _is_model_endpoint_wrapper(model):
        return False
    try:
        endpoint = model.python_model.endpoint
        deploy_client = mlflow.deployments.get_deploy_client(mlflow_utils.resolve_deployments_target())
        endpoint_metadata = deploy_client.get_endpoint(endpoint)
        if hasattr(endpoint_metadata, "task"):
            return _is_agent_task(endpoint_metadata.task)

        # Note that `served_models` is being deprecated in favor of `served_entities`
        models = endpoint_metadata.config.get("served_models") or endpoint_metadata.config.get("served_entities")
        if not models:
            return False
        model_name = models[0]["model_name"]
        return len(agents.get_deployments(model_name)) > 0
    except Exception as e:
        _logger.warning("Fail to check if the model is an agent endpoint", e)
        return False
