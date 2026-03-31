"""This module deals with extracting information from traces."""

import json
import logging
from typing import Any, Dict, List, Optional

import mlflow.entities as mlflow_entities
import mlflow.models.dependencies_schemas as mlflow_dependencies_schemas
import mlflow.tracing.constant as mlflow_tracing_constant

from databricks.rag_eval.evaluation import entities
from databricks.rag_eval.utils import input_output_utils, trace_utils

_logger = logging.getLogger(__name__)


_DEFAULT_DOC_URI_COL = "doc_uri"

_ID = "id"  # Generic ID field, used for tool call ID
_MESSAGES = "messages"

_ROLE = "role"
_ASSISTANT_ROLE = "assistant"
_TOOL_ROLE = "tool"

_TOOL_FUNCTION = "function"
_TOOL_FUNCTION_NAME = "name"
_TOOL_FUNCTION_ARGUMENTS = "arguments"
_TOOL_CALLS = "tool_calls"
_TOOL_CALL_ID = "tool_call_id"

# MLflow span attribute keys - define missing constant
# TODO(ML-57093): Remove this at some point. This was necessary due to the following PR
# merged in MLflow 3.3.0: https://github.com/mlflow/mlflow/pull/16762
_CHAT_MESSAGES = "mlflow.chat.messages"


# ================== Retrieval Context ==================
def extract_retrieval_context_from_trace(
    trace: Optional[mlflow_entities.Trace],
) -> Optional[entities.RetrievalContext]:
    """
    Extract the retrieval context from the trace.

    Only consider the last retrieval span in the trace if there are multiple retrieval spans.

    If the trace does not have a retrieval span, return None.

    ⚠️ Warning: Please make sure to not throw exception. If fails, return None.

    :param trace: The trace
    :return: The retrieval context
    """
    if trace is None or trace.data is None:
        return None

    # Only consider the top-level retrieval spans
    top_level_retrieval_spans = _get_top_level_retrieval_spans(trace)
    if len(top_level_retrieval_spans) == 0:
        return None
    # Only consider the last top-level retrieval span
    retrieval_span = top_level_retrieval_spans[-1]

    # Get the retriever schema from the trace info
    retriever_schema = _get_retriever_schema_from_trace(trace.info)
    return _extract_retrieval_context_from_retrieval_span(retrieval_span, retriever_schema)


def _get_top_level_retrieval_spans(
    trace: mlflow_entities.Trace,
) -> List[mlflow_entities.Span]:
    """
    Get the top-level retrieval spans in the trace.
    Top-level retrieval spans are the retrieval spans that are not children of other retrieval spans.

    For example, given the following spans:
    - Span A (Chain)
      - Span B (Retriever)
        - Span C (Retriever)
      - Span D (Retriever)
        - Span E (LLM)
          - Span F (Retriever)
    Span B and Span D are top-level retrieval spans.
    Span C and Span F are NOT top-level retrieval spans because they are children of other retrieval spans.
    """
    if trace.data is None or not trace.data.spans:
        return []

    retrieval_spans = {span.span_id: span for span in trace.search_spans(mlflow_entities.SpanType.RETRIEVER)}

    top_level_retrieval_spans = []

    for span in retrieval_spans.values():
        # Check if this span is a child of another retrieval span
        parent_id = span.parent_id
        while parent_id:
            if parent_id in retrieval_spans.keys():
                # This span is a child of another retrieval span
                break
            parent_span = next((s for s in trace.data.spans if s.span_id == parent_id), None)
            if parent_span is None:
                # The parent span is not found, malformed trace
                break
            parent_id = parent_span.parent_id
        else:
            # If the loop completes without breaking, this is a top-level span
            top_level_retrieval_spans.append(span)

    return top_level_retrieval_spans


def _get_retriever_schema_from_trace(
    trace_info: Optional[mlflow_entities.TraceInfo],
) -> Optional[mlflow_dependencies_schemas.RetrieverSchema]:
    """
    Get the retriever schema from the trace info tags.

    Retriever schema is stored in the trace info tags as a JSON string of list of retriever schemas.
    Only consider the last retriever schema if there are multiple retriever schemas.
    """
    if (
        trace_info is None
        or trace_info.tags is None
        or mlflow_dependencies_schemas.DependenciesSchemasType.RETRIEVERS.value not in trace_info.tags
    ):
        return None
    retriever_schemas = json.loads(
        trace_info.tags[mlflow_dependencies_schemas.DependenciesSchemasType.RETRIEVERS.value]
    )
    # Only consider the last retriever schema
    return (
        mlflow_dependencies_schemas.RetrieverSchema.from_dict(retriever_schemas[-1])
        if isinstance(retriever_schemas, list) and len(retriever_schemas) > 0
        else None
    )


def _extract_retrieval_context_from_retrieval_span(
    span: mlflow_entities.Span,
    retriever_schema: Optional[mlflow_dependencies_schemas.RetrieverSchema],
) -> Optional[entities.RetrievalContext]:
    """Get the retrieval context from a retrieval span."""
    try:
        doc_uri_col = (
            retriever_schema.doc_uri if retriever_schema and retriever_schema.doc_uri else _DEFAULT_DOC_URI_COL
        )
        retriever_outputs = span.attributes.get(mlflow_tracing_constant.SpanAttributeKey.OUTPUTS)
        return entities.RetrievalContext(
            chunks=[
                (
                    entities.Chunk(
                        doc_uri=(chunk.get("metadata", {}).get(doc_uri_col) if chunk else None),
                        content=chunk.get("page_content") if chunk else None,
                    )
                )
                for chunk in retriever_outputs or []
            ],
            span_id=span.span_id,
        )
    except Exception as e:
        _logger.debug(f"Fail to get retrieval context from span: {span}. Error: {e!r}")
        return None


# ================== Model Input/Output ==================
def extract_model_output_from_trace(
    trace: Optional[mlflow_entities.Trace],
) -> Optional[input_output_utils.ModelOutput]:
    """
    Extract the model output from the trace.

    Model output should be recorded in the root span of the trace.

    ⚠️ Warning: Please make sure to not throw exception. If fails, return None.
    """
    if trace is None:
        return None
    root_span = trace_utils.get_root_span(trace)
    if root_span is None:
        return None

    try:
        if root_span.attributes is None or mlflow_tracing_constant.SpanAttributeKey.OUTPUTS not in root_span.attributes:
            return None
        return root_span.attributes[mlflow_tracing_constant.SpanAttributeKey.OUTPUTS]

    except Exception as e:
        _logger.debug(f"Fail to extract model output from the root span: {root_span}. Error: {e!r}")
        return None


# ================== Tool Calls ==================
def _extract_tool_calls_from_messages(
    messages: List[Dict[str, Any]],
) -> List[entities.ToolCallInvocation]:
    """
    Helper to extract the tool calls from a list of messages. Uses the tool call ID to match tool
    calls from assistant messages to tool call results from tool messages. Note that there is no
    notion of tool spans, so we cannot extract the raw tool span.
    :param messages: List of messages
    :return: List of tool call invocations
    """
    assistant_messages_with_tools = [
        message for message in messages if message[_ROLE] == _ASSISTANT_ROLE and message.get(_TOOL_CALLS)
    ]

    tool_call_results = {message[_TOOL_CALL_ID]: message for message in messages if message[_ROLE] == _TOOL_ROLE}

    tool_call_invocations = []
    for message in assistant_messages_with_tools:
        for tool_call in message.get(_TOOL_CALLS) or []:
            tool_call_id = tool_call[_ID]
            tool_call_function = tool_call[_TOOL_FUNCTION]
            tool_call_args = tool_call_function[_TOOL_FUNCTION_ARGUMENTS]
            if isinstance(tool_call_args, str):
                try:
                    tool_call_args = json.loads(tool_call_args)
                except Exception:
                    pass

            tool_call_invocations.append(
                entities.ToolCallInvocation(
                    tool_name=tool_call_function[_TOOL_FUNCTION_NAME],
                    tool_call_args=tool_call_args,
                    tool_call_id=tool_call_id,
                    tool_call_result=tool_call_results.get(tool_call_id),
                )
            )
    return tool_call_invocations


def _extract_tool_calls_from_tool_spans(
    trace: mlflow_entities.Trace,
) -> List[entities.ToolCallInvocation]:
    """
    Helper to extract tool calls from tool spans in the trace. Note that there is no way to connect
    the tool span to the LLM/ChatModel span to get available tools.
    :param trace: The trace
    :return: List of tool call invocations
    """
    tool_spans = trace.search_spans(mlflow_entities.SpanType.TOOL)
    tool_call_invocations = []
    for span in tool_spans:
        span_inputs = span.attributes[mlflow_tracing_constant.SpanAttributeKey.INPUTS]
        span_outputs = span.attributes[mlflow_tracing_constant.SpanAttributeKey.OUTPUTS]

        tool_call_invocations.append(
            entities.ToolCallInvocation(
                tool_name=span.name,
                tool_call_args=span_inputs,
                tool_call_id=(span_outputs or {}).get(_TOOL_CALL_ID),
                tool_call_result=span_outputs,
                raw_span=span,
            )
        )

    return tool_call_invocations


def _extract_tool_calls_from_chat_model_spans(
    trace: mlflow_entities.Trace,
) -> List[entities.ToolCallInvocation]:
    """
    Helper to extract tool calls from chat model spans in the trace. Note that this method
    relies on new fields introducing in mlflow 2.20.0 to extract standardized messages and
    available tools.
    :param trace: The trace
    :return: List of tool call invocations
    """
    chat_model_spans = trace.search_spans(mlflow_entities.SpanType.CHAT_MODEL)

    tool_call_id_to_available_tools = {}
    # These dictionaries store both the invocation/result and respective spans. We store
    # the span such that we can prioritize the result span over invocation span.
    tool_call_id_to_invocation = {}
    tool_call_id_to_result = {}

    for span in chat_model_spans:
        messages = span.attributes.get(_CHAT_MESSAGES) or []
        for idx, message in enumerate(messages):
            # The assistant has generated some tool call invocations
            if message[_ROLE] == _ASSISTANT_ROLE and _TOOL_CALLS in message:
                for tool_call in message[_TOOL_CALLS]:
                    tool_call_id = tool_call[_ID]
                    # We should use the first available span (i.e., don't overwrite this value)
                    if tool_call_id not in tool_call_id_to_invocation:
                        tool_call_id_to_invocation[tool_call_id] = (tool_call, span)

                    # If the tool call invocation is the last message, then this span
                    # contains the available tools that can be invoked. Otherwise,
                    # we cannot assume it came from the same span.
                    available_tools = span.attributes.get(mlflow_tracing_constant.SpanAttributeKey.CHAT_TOOLS)
                    if available_tools and idx == len(messages) - 1:
                        tool_call_id_to_available_tools[tool_call_id] = available_tools

            # The tool has responded with a tool call result
            if message[_ROLE] == _TOOL_ROLE:
                tool_call_id = message[_TOOL_CALL_ID]
                # We should use the first available span (i.e., don't overwrite this value)
                if tool_call_id not in tool_call_id_to_result:
                    tool_call_id_to_result[tool_call_id] = (message, span)

    tool_call_invocations = []
    for tool_call_id, (
        tool_call_invocation,
        invocation_span,
    ) in tool_call_id_to_invocation.items():
        tool_call_function = tool_call_invocation.get(_TOOL_FUNCTION) or {}
        tool_call_args = tool_call_function[_TOOL_FUNCTION_ARGUMENTS]
        tool_call_result, result_span = tool_call_id_to_result.get(tool_call_id, (None, None))

        tool_call_invocations.append(
            entities.ToolCallInvocation(
                tool_name=tool_call_function[_TOOL_FUNCTION_NAME],
                tool_call_args=(json.loads(tool_call_args) if isinstance(tool_call_args, str) else tool_call_args),
                tool_call_id=tool_call_id,
                tool_call_result=tool_call_result,
                # This will prefer the span containing both result + invocation over just
                # the invocation span
                raw_span=result_span or invocation_span,
                available_tools=tool_call_id_to_available_tools.get(tool_call_id),
            )
        )

    return tool_call_invocations


def extract_tool_calls(
    *,
    response: Optional[Dict[str, Any]] = None,
    trace: Optional[mlflow_entities.Trace] = None,
) -> Optional[List[entities.ToolCallInvocation]]:
    """
    Extract tool calls from a response or trace object. The trace is prioritized as it provides more
    metadata about the tool calls and not all tools may be logged in the request.
    :param response: The response object
    :param trace: The trace object
    :return: List of tool call invocations
    """
    try:
        # We prefer extracting from the trace opposed to a response object
        if trace is not None:
            if not isinstance(trace, mlflow_entities.Trace):
                raise ValueError(f"Expected a `mlflow.entities.Trace` object, got {type(trace)}")
            tool_calls_from_traces = _extract_tool_calls_from_chat_model_spans(trace)
            # Try extracting from the chat model spans. If no tools are found, try using the tool spans.
            return tool_calls_from_traces if len(tool_calls_from_traces) else _extract_tool_calls_from_tool_spans(trace)

        if response is not None:
            if not isinstance(response, Dict):
                raise ValueError(f"Expected a dictionary, got {type(response)}")
            if _MESSAGES not in response:
                raise ValueError(f"Invalid response object is missing field '{_MESSAGES}': {response}")
            if not isinstance(response[_MESSAGES], list):
                raise ValueError(f"Expected a list of messages, got {type(response[_MESSAGES])}")

            return _extract_tool_calls_from_messages(response[_MESSAGES])

        raise ValueError("A response or trace object must be provided.")
    except Exception:
        return None
