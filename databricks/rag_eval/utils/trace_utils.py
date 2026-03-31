"""This module provides general helpers for traces with no dependencies on the agent evaluation harness."""

import uuid
from typing import Any, Dict, List, Optional

import mlflow
import mlflow.entities as mlflow_entities
import mlflow.pyfunc.context as pyfunc_context
import mlflow.tracing.constant as mlflow_tracing_constant

from databricks.rag_eval.utils import serialization_utils


def span_is_type(
    span: mlflow_entities.Span,
    span_type: str | List[str],
) -> bool:
    """Check if the span is of a certain span type or one of the span types in the collection"""
    if span.attributes is None:
        return False
    if not isinstance(span_type, List):
        span_type = [span_type]
    return span.attributes.get(mlflow_tracing_constant.SpanAttributeKey.SPAN_TYPE) in span_type


def get_leaf_spans(trace: mlflow_entities.Trace) -> List[mlflow_entities.Span]:
    """Get all leaf spans in the trace."""
    if trace.data is None:
        return []
    spans = trace.data.spans or []
    leaf_spans_by_id = {span.span_id: span for span in spans}
    for span in spans:
        if span.parent_id:
            leaf_spans_by_id.pop(span.parent_id, None)
    return list(leaf_spans_by_id.values())


def get_root_span(trace: mlflow_entities.Trace) -> Optional[mlflow_entities.Span]:
    """Get the root span in the trace."""
    if trace.data is None:
        return None
    spans = trace.data.spans or []
    # Root span is the span that has no parent
    return next((span for span in spans if span.parent_id is None), None)


# ================== Trace Creation/Modification ==================
def _generate_trace_id() -> str:
    """
    Generate a new trace ID. This is a 16-byte hex string.
    """
    return uuid.uuid4().hex


def _generate_span_id() -> str:
    """
    Generate a new span ID. This is a 8-byte hex string.
    """
    return uuid.uuid4().hex[:16]  # OTel span spec says it's only 8 bytes (16 hex chars)


def create_minimal_trace(
    request: Dict[str, Any],
    response: Any,
    retrieval_context: Optional[List[mlflow_entities.Document]] = None,
    status: Optional[str] = "OK",
) -> mlflow_entities.Trace:
    """
    Create a minimal trace object with a single span, based on given request/response. If
    retrieval context is provided, a retrieval span is added.

    :param request: The request object. This is expected to be a JSON-serializable object
    :param response: The response object. This is expected to be a JSON-serializable object, but we cannot guarantee this
    :param retrieval_context: Optional list of documents retrieved during processing
    :param status: The status of the trace
    :return: A trace object.
    """
    # Set the context so that the trace is logged synchronously
    context_id = str(uuid.uuid4())
    with pyfunc_context.set_prediction_context(pyfunc_context.Context(context_id, is_evaluate=True)):
        with mlflow.start_span(
            name="root_span",
            attributes={mlflow_tracing_constant.SpanAttributeKey.SPAN_TYPE: mlflow_entities.SpanType.CHAIN},
        ) as root_span:
            root_span.set_inputs(request)

            # Add retrieval span if retrieval context is provided
            if retrieval_context is not None:
                with mlflow.start_span(
                    name="retrieval_span",
                    attributes={mlflow_tracing_constant.SpanAttributeKey.SPAN_TYPE: mlflow_entities.SpanType.RETRIEVER},
                ) as retrieval_span:
                    retrieval_span.set_outputs([doc.to_dict() for doc in retrieval_context])

            root_span.set_outputs(response)
            root_span.set_status(status)
        return mlflow.get_trace(root_span.trace_id)


def inject_experiment_run_id_to_trace(
    trace: mlflow_entities.Trace, experiment_id: str, run_id: Optional[str] = None
) -> mlflow_entities.Trace:
    """
    Inject the experiment and run ID into the trace metadata.

    :param trace: The trace object
    :param experiment_id: The experiment ID to inject
    :param run_id: The run ID to inject
    :return: The updated trace object
    """
    if trace.info.trace_metadata is None:
        trace.info.trace_metadata = {}

    if run_id is not None:
        trace.info.trace_metadata[mlflow_tracing_constant.TraceMetadataKey.SOURCE_RUN] = run_id

    trace_location = trace.info.trace_location
    if trace_location.type == mlflow_entities.TraceLocationType.MLFLOW_EXPERIMENT:
        if trace_location.mlflow_experiment is not None:
            trace.info.trace_location.mlflow_experiment.experiment_id = experiment_id
        else:
            trace.info.trace_location.mlflow_experiment = mlflow_entities.MlflowExperimentLocation(
                experiment_id=experiment_id
            )
    else:
        raise ValueError(f"Cannot inject experiment ID to trace location type: {trace_location.type}")

    return trace


def clone_trace_to_reupload(trace: mlflow_entities.Trace) -> mlflow_entities.Trace:
    """
    Prepare a trace for cloning by resetting traceId and clearing various fields.
    This has the downstream effect of causing the trace to be recreated with a new trace_id.

    :param trace: The trace to prepare
    :return: The prepared trace
    """
    prepared_trace = serialization_utils.deserialize_trace(serialization_utils.serialize_trace(trace))

    # Since the semantics of this operation are to _clone_ the trace, and assessments are tied to
    # a specific trace, we clear assessments as well.
    prepared_trace.info.assessments = []

    # Tags and metadata also contain references to the source run, trace data artifact location, etc.
    # We clear these as well to ensure that the trace is not tied to the original source of the trace.
    for key in [k for k in prepared_trace.info.tags.keys() if k.startswith("mlflow.")]:
        prepared_trace.info.tags.pop(key)
    prepared_trace.info.tags["mlflow.databricks.sourceTraceId"] = trace.info.trace_id

    for key in [
        k
        for k in prepared_trace.info.trace_metadata.keys()
        if k.startswith("mlflow.")
        and k
        not in [
            mlflow_tracing_constant.TraceMetadataKey.INPUTS,
            mlflow_tracing_constant.TraceMetadataKey.OUTPUTS,
        ]
    ]:
        prepared_trace.info.trace_metadata.pop(key)

    prepared_trace.info.trace_id = "tr-" + str(uuid.uuid4().hex)
    # We skip updating the spans here to avoid issues with blob storage retrievals
    return prepared_trace
