"""Entities for evaluation."""

import dataclasses
import hashlib
import json
from collections import abc
from copy import deepcopy
from typing import Any, Callable, Dict, List, Mapping, Optional, TypeAlias, Union

import mlflow.entities as mlflow_entities
import pandas as pd

from databricks.rag_eval import constants, context, schemas
from databricks.rag_eval.config import (
    assessment_config,
    example_config,
)
from databricks.rag_eval.utils import (
    collection_utils,
    enum_utils,
    input_output_utils,
    serialization_utils,
    trace_utils,
)

ChunkInputData = Union[str, Dict[str, Any]]
RetrievalContextInputData = List[Optional[ChunkInputData]]

_SPAN_OUTPUT_KEY = "span_output_key"
_CHUNK_INDEX_KEY = "chunk_index"

_EXPECTATION_FIELDS = [
    schemas.EXPECTED_RESPONSE_COL,
    schemas.EXPECTED_RETRIEVED_CONTEXT_COL,
    schemas.EXPECTED_FACTS_COL,
    schemas.GUIDELINES_COL,
    # Note we don't include schemas.CUSTOM_EXPECTED_COL here because it's handled separately.
]
_EXCLUDED_METRICS_FROM_LOGGING = [
    schemas.LATENCY_SECONDS_COL,
]


@dataclasses.dataclass
class Chunk:
    doc_uri: Optional[str] = None
    content: Optional[str] = None

    @classmethod
    def from_input_data(cls, input_data: Optional[ChunkInputData]) -> Optional["Chunk"]:
        """
        Construct a Chunk from a dictionary optionally containing doc_uri and content.

        An input chunk of a retrieval context can be:
          - A doc URI; or
          - A dictionary with the schema defined in schemas.CHUNK_SCHEMA
        """
        if input_output_utils.is_none_or_nan(input_data):
            return None
        if isinstance(input_data, str):
            return cls(doc_uri=input_data)
        else:
            return cls(
                doc_uri=input_data.get(schemas.DOC_URI_COL),
                content=input_data.get(schemas.CHUNK_CONTENT_COL),
            )

    def to_dict(self):
        return {
            schemas.DOC_URI_COL: self.doc_uri,
            schemas.CHUNK_CONTENT_COL: self.content,
        }

    def to_mlflow_document(self) -> mlflow_entities.Document:
        return mlflow_entities.Document(
            page_content=self.content,
            metadata={
                "doc_uri": self.doc_uri,
            },
        )


@dataclasses.dataclass
class RetrievalContext:
    chunks: List[Optional[Chunk]]
    span_id: Optional[str] = None

    def concat_chunk_content(self, delimiter: str = constants.DEFAULT_CONTEXT_CONCATENATION_DELIMITER) -> Optional[str]:
        """
        Concatenate the non-empty content of the chunks to a string with the given delimiter.
        Return None if all the contents are empty.
        """
        non_empty_contents = [chunk.content for chunk in self.chunks if chunk is not None and chunk.content]
        return delimiter.join(non_empty_contents) if non_empty_contents else None

    def get_doc_uris(self) -> List[Optional[str]]:
        """Get the list of doc URIs in the retrieval context."""
        return [chunk.doc_uri for chunk in self.chunks if chunk is not None]

    def to_output_dict(self) -> List[Dict[str, str]]:
        """Convert the RetrievalContext to a list of dictionaries with the schema defined in schemas.CHUNK_SCHEMA."""
        return [
            (
                {
                    schemas.DOC_URI_COL: chunk.doc_uri,
                    schemas.CHUNK_CONTENT_COL: chunk.content,
                }
                if chunk is not None
                else None
            )
            for chunk in self.chunks
        ]

    def to_mlflow_documents(self) -> List[mlflow_entities.Document]:
        return [chunk.to_mlflow_document() for chunk in self.chunks if chunk is not None]

    @classmethod
    def from_input_data(cls, input_data: Optional[RetrievalContextInputData]) -> Optional["RetrievalContext"]:
        """
        Construct a RetrievalContext from the input.

        Input can be:
        - A list of doc URIs
        - A list of dictionaries with the schema defined in schemas.CHUNK_SCHEMA
        """
        if input_output_utils.is_none_or_nan(input_data):
            return None
        return cls(chunks=[Chunk.from_input_data(chunk_data) for chunk_data in input_data])


@dataclasses.dataclass
class ToolCallInvocation:
    tool_name: str
    tool_call_args: Dict[str, Any]
    tool_call_id: Optional[str] = None
    tool_call_result: Optional[Dict[str, Any]] = None

    # Only available from the trace
    raw_span: Optional[mlflow_entities.Span] = None
    available_tools: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "tool_call_args": self.tool_call_args,
            "tool_call_id": self.tool_call_id,
            "tool_call_result": self.tool_call_result,
            "raw_span": self.raw_span,
            "available_tools": self.available_tools,
        }

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "ToolCallInvocation":
        return cls(
            tool_name=data["tool_name"],
            tool_call_args=data.get("tool_call_args", {}),
            tool_call_id=data.get("tool_call_id"),
            tool_call_result=data.get("tool_call_result"),
            raw_span=data.get("raw_span"),
            available_tools=data.get("available_tools"),
        )

    @classmethod
    def from_dict(
        cls, tool_calls: Optional[List[Dict[str, Any]] | Dict[str, Any]]
    ) -> Optional["ToolCallInvocation" | List["ToolCallInvocation"]]:
        if tool_calls is None:
            return None
        if isinstance(tool_calls, dict):
            return cls._from_dict(tool_calls)
        elif isinstance(tool_calls, list):
            return [cls._from_dict(tool_call) for tool_call in tool_calls]
        else:
            raise ValueError(f"Expected `tool_calls` to be a `dict` or `List[dict]`, but got: {type(tool_calls)}")


class CategoricalRating(enum_utils.StrEnum):
    """A categorical rating for an assessment."""

    YES = "yes"
    NO = "no"
    UNKNOWN = "unknown"

    @classmethod
    def _missing_(cls, value: str):
        value = value.lower()
        for member in cls:
            if member == value:
                return member
        return cls.UNKNOWN

    @classmethod
    def from_example_rating(cls, rating: example_config.ExampleRating) -> "CategoricalRating":
        """Convert an ExampleRating to a CategoricalRating."""
        match rating:
            case example_config.ExampleRating.YES:
                return cls.YES
            case example_config.ExampleRating.NO:
                return cls.NO
            case _:
                return cls.UNKNOWN


@dataclasses.dataclass
class Rating:
    double_value: Optional[float]
    rationale: Optional[str]
    categorical_value: Optional[CategoricalRating]
    error_message: Optional[str]
    error_code: Optional[str]

    @classmethod
    def value(
        cls,
        *,
        rationale: Optional[str] = None,
        double_value: Optional[float] = None,
        categorical_value: Optional[CategoricalRating | str] = None,
    ) -> "Rating":
        """Build a normal Rating with a categorical value, a double value, and a rationale."""
        if categorical_value is not None and not isinstance(categorical_value, CategoricalRating):
            categorical_value = CategoricalRating(categorical_value)
        return cls(
            double_value=double_value,
            rationale=rationale,
            categorical_value=categorical_value,
            error_message=None,
            error_code=None,
        )

    @classmethod
    def error(cls, error_message: str, error_code: Optional[str | int] = None) -> "Rating":
        """Build an error Rating with an error message and an optional error code."""
        if isinstance(error_code, int):
            error_code = str(error_code)
        return cls(
            double_value=None,
            rationale=None,
            categorical_value=None,
            error_message=error_message,
            error_code=error_code or "UNKNOWN",
        )

    @classmethod
    def flip(cls, rating: "Rating") -> "Rating":
        """Built a Rating with the inverse categorical and float values of the input Rating."""
        if rating.double_value is not None and (rating.double_value < 1.0 or rating.double_value > 5.0):
            raise ValueError(f"Cannot flip the rating of double value: {rating.double_value}.")

        match rating.categorical_value:
            case CategoricalRating.YES:
                flipped_categorical_value = CategoricalRating.NO
                flipped_double_value = 1.0
            case CategoricalRating.NO:
                flipped_categorical_value = CategoricalRating.YES
                flipped_double_value = 5.0
            case CategoricalRating.UNKNOWN:
                flipped_categorical_value = CategoricalRating.UNKNOWN
                flipped_double_value = None
            case None:
                flipped_categorical_value = None
                flipped_double_value = None
            case _:
                raise ValueError(f"Cannot flip the rating of categorical value: {rating.categorical_value}")

        return cls(
            double_value=flipped_double_value,
            rationale=rating.rationale,
            categorical_value=flipped_categorical_value,
            error_message=rating.error_message,
            error_code=rating.error_code,
        )


PositionalRating: TypeAlias = Mapping[int, Rating]
"""
A mapping from position to rating.
Position refers to the position of the chunk in the retrieval context.
It is used to represent the ratings of the chunks in the retrieval context.
"""


@dataclasses.dataclass
class EvalItem:
    """
    Represents a row in the evaluation dataset. It contains information needed to evaluate a question.
    """

    question_id: str
    """Unique identifier for the eval item."""

    raw_request: Any
    """Raw input to the agent when `evaluate` is called. Comes from "request" or "inputs" columns. """

    raw_response: Any
    """Raw output from an agent."""

    has_inputs_outputs: bool = False
    """Whether the eval item used the new inputs/outputs columns, or the old request/response columns."""

    question: Optional[str] = None
    """String representation of the model input that is used for evaluation."""

    answer: Optional[str] = None
    """String representation of the model output that is used for evaluation."""

    retrieval_context: Optional[RetrievalContext] = None
    """Retrieval context that is used for evaluation."""

    ground_truth_answer: Optional[str] = None
    """String representation of the ground truth answer."""

    ground_truth_retrieval_context: Optional[RetrievalContext] = None
    """Ground truth retrieval context."""

    expected_facts: Optional[List[str]] = None
    """List of expected facts to help evaluate the answer."""

    guidelines: Optional[List[str]] = None
    """[INTERNAL ONLY] List of guidelines the provided context must adhere to used for the judge service."""

    named_guidelines: Optional[Dict[str, List[str]]] = None
    """Mapping of name to guidelines the provided context must adhere to."""

    guidelines_context: Optional[Dict[str, str]] = None
    """Mapping of a string (context field name) to string (content) containing context the guidelines can apply to."""

    custom_expected: Optional[Dict[str, Any]] = None
    """Custom expected data to help evaluate the answer."""

    custom_inputs: Optional[Dict[str, Any]] = None
    """Custom expected data to help evaluate the answer."""

    custom_outputs: Optional[Dict[str, Any]] = None
    """Custom expected data to help evaluate the answer."""

    trace: Optional[mlflow_entities.Trace] = None
    """Trace of the model invocation."""

    tool_calls: Optional[List[ToolCallInvocation]] = None
    """List of tool call invocations from an agent."""

    managed_evals_eval_id: Optional[str] = None
    """Unique identifier for the managed-evals eval item."""

    managed_evals_dataset_id: Optional[str] = None
    """Unique identifier for the managed-evals dataset."""

    model_error_message: Optional[str] = None
    """Error message if the model invocation fails."""

    source_id: Optional[str] = None
    """
    The source for this eval row. If source_type is "HUMAN", then user email.
    If source_type is "SYNTHETIC_FROM_DOC", then the doc URI.
    """

    source_type: Optional[str] = None
    """Source of the eval item. e.g. HUMAN, SYNTHETIC_FROM_DOC, PRODUCTION_LOG..."""

    tags: Optional[Dict[str, str]] = None
    """Tags associated with the eval item."""

    @property
    def concatenated_retrieval_context(self) -> Optional[str]:
        """Get the concatenated content of the retrieval context.
        Return None if there is no non-empty retrieval context content."""
        return self.retrieval_context.concat_chunk_content() if self.retrieval_context else None

    @property
    def raw_guidelines(self):
        # When returning the guidelines, ensure they are returned in the format they are given.
        # In other words, revert the default mapping we create when a list of guidelines is provided.
        is_named_guidelines = not (
            self.named_guidelines is not None
            and len(self.named_guidelines) == 1
            and assessment_config.GUIDELINE_ADHERENCE.assessment_name in self.named_guidelines
        )
        return self.guidelines if not is_named_guidelines and self.guidelines is not None else self.named_guidelines

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvalItem":
        """
        Create an EvalItem from a row of MLflow EvaluationDataset data.
        """
        retrieved_context = RetrievalContext.from_input_data(data.get(schemas.RETRIEVED_CONTEXT_COL))

        expected_retrieved_context = RetrievalContext.from_input_data(data.get(schemas.EXPECTED_RETRIEVED_CONTEXT_COL))

        has_inputs_outputs = False
        # Set the question/raw_request
        raw_request = data.get(schemas.REQUEST_COL)
        # Get the raw request from "inputs" if "request" is not present.
        if not raw_request:
            # Parse the "inputs" if it is a VariantVal.
            raw_request = input_output_utils.parse_variant_data(data.get(schemas.INPUTS_COL))
            has_inputs_outputs = True
            if isinstance(raw_request, str):
                try:
                    # Deseralize the "inputs" json string into dict[str, Any].
                    raw_request: dict[str, Any] = json.loads(raw_request)
                except Exception as e:
                    raise ValueError(f"`{schemas.INPUTS_COL}` must be JSON serializable: {type(raw_request)}") from e
        try:
            question = input_output_utils.request_to_string(raw_request)
        except ValueError:
            question = None

        # Set the question id
        question_id = data.get(schemas.REQUEST_ID_COL)
        if input_output_utils.is_none_or_nan(question_id):
            question_id = hashlib.sha256(str(raw_request).encode()).hexdigest()

        # Set the answer/raw_response
        raw_response = data.get(schemas.RESPONSE_COL)
        # Get the raw response from "outputs" if "response" is not present.
        if not raw_response:
            # Parse the "outputs" if it is a VariantVal.
            raw_response = input_output_utils.parse_variant_data(data.get(schemas.OUTPUTS_COL))
            if isinstance(raw_response, str):
                try:
                    # Deseralize the json string into dict[str, Any].
                    raw_response: dict[str, Any] = json.loads(raw_response)
                except Exception as e:
                    raise ValueError(f"`{schemas.OUTPUTS_COL}` must be JSON serializable: {type(raw_response)}") from e
        try:
            answer = input_output_utils.response_to_string(raw_response)
        except ValueError:
            answer = None

        ground_truth_answer = input_output_utils.response_to_string(data.get(schemas.EXPECTED_RESPONSE_COL))

        expected_facts = data.get(schemas.EXPECTED_FACTS_COL)
        expected_facts = list(expected_facts) if not input_output_utils.is_none_or_nan(expected_facts) else None

        named_guidelines = data.get(schemas.GUIDELINES_COL)

        trace = data.get(schemas.TRACE_COL)
        if input_output_utils.is_none_or_nan(trace):
            trace = None
        else:
            trace = serialization_utils.deserialize_trace(trace)

        trace_expectations = {}
        if trace:
            # Note that we do not expect multiple expectations with the same name. However, if we
            # do get multiple, we take the latest one based on the last_update_time_ms field.
            sorted_assessments = sorted(trace.info.assessments or [], key=lambda a: a.last_update_time_ms or 0)
            for assessment in sorted_assessments:
                if assessment.expectation is not None:
                    trace_expectations[assessment.name] = assessment.expectation.value

        # Extract the relevant expected data from "custom_expected" or "expectations".
        custom_expected = input_output_utils.normalize_to_dictionary(deepcopy(data.get(schemas.CUSTOM_EXPECTED_COL)))
        expectations = input_output_utils.normalize_to_dictionary(deepcopy(data.get(schemas.EXPECTATIONS_COL)))

        # Merge the trace expectations with the specified expectations. Expectations from the
        # expectations column take precedence. custom_expected and expectations are mutually
        # exclusive so order does not matter.
        expectations = {**trace_expectations, **expectations, **custom_expected}

        # Extract the built-in expectations from the expectations.
        if ground_truth_answer is None and schemas.EXPECTED_RESPONSE_COL in expectations:
            ground_truth_answer = expectations.pop(schemas.EXPECTED_RESPONSE_COL)
        if expected_facts is None and schemas.EXPECTED_FACTS_COL in expectations:
            expected_facts = expectations.pop(schemas.EXPECTED_FACTS_COL)
        if named_guidelines is None and schemas.GUIDELINES_COL in expectations:
            named_guidelines = expectations.pop(schemas.GUIDELINES_COL)
        if expected_retrieved_context is None and schemas.EXPECTED_RETRIEVED_CONTEXT_COL in expectations:
            expected_retrieved_context = RetrievalContext.from_input_data(
                expectations.pop(schemas.EXPECTED_RETRIEVED_CONTEXT_COL)
            )

        # If the dict is empty, set to None to avoid creating an empty output column.
        custom_expected = expectations or None

        guidelines = None  # These are only used to pass to the judge
        if input_output_utils.is_none_or_nan(named_guidelines):
            guidelines = None
            named_guidelines = None
        elif isinstance(named_guidelines, abc.Iterable) and not isinstance(named_guidelines, Mapping):
            # When an iterable (e.g., list or numpy array) is passed, we can use these guidelines
            # for the judge service. We cannot use a mapping.
            guidelines = list(named_guidelines)
            # Convert an iterable of guidelines to a default mapping
            named_guidelines = {assessment_config.GUIDELINE_ADHERENCE.assessment_name: list(named_guidelines)}

        guidelines_context = data.get(schemas.GUIDELINES_CONTEXT_COL)
        guidelines_context = guidelines_context if not input_output_utils.is_none_or_nan(guidelines_context) else None

        custom_inputs = None
        if isinstance(raw_request, dict):
            custom_inputs = raw_request.get(schemas.CUSTOM_INPUTS_COL)
        elif hasattr(raw_request, "custom_inputs"):
            custom_inputs = raw_request.custom_inputs
        # Deserialize the custom inputs if they are a string
        if isinstance(custom_inputs, str):
            custom_inputs = json.loads(custom_inputs)

        custom_outputs = None
        if isinstance(raw_response, dict):
            custom_outputs = raw_response.get(schemas.CUSTOM_OUTPUTS_COL)
        elif hasattr(raw_response, "custom_outputs"):
            custom_outputs = raw_response.custom_outputs
        # Deserialize the custom outputs if they are a string
        if isinstance(custom_outputs, str):
            custom_outputs = json.loads(custom_outputs)

        tool_calls = data.get(schemas.TOOL_CALLS_COL)
        if isinstance(tool_calls, list):
            tool_calls = [ToolCallInvocation.from_dict(tool_call) for tool_call in tool_calls]
        elif isinstance(tool_calls, dict):
            tool_calls = [ToolCallInvocation.from_dict(tool_calls)]

        source_id = data.get(schemas.SOURCE_ID_COL)
        if input_output_utils.is_none_or_nan(source_id):
            source_id = None
        source_type = data.get(schemas.SOURCE_TYPE_COL)
        if input_output_utils.is_none_or_nan(source_type):
            source_type = None

        if not source_id:
            source = data.get(schemas.SOURCE_COL)
            if not input_output_utils.is_none_or_nan(source):
                if (source.get("human") or {}).get("user_name", None):
                    source_id = source.get("human").get("user_name")
                    source_type = "human"
                elif (source.get("document") or {}).get("doc_uri", None):
                    source_id = source.get("document").get("doc_uri")
                    source_type = "document"
                elif (source.get("trace") or {}).get("trace_id", None):
                    source_id = source.get("trace").get("trace_id")
                    source_type = "trace"

        managed_evals_eval_id = data.get(schemas.MANAGED_EVALS_EVAL_ID_COL)
        managed_evals_dataset_id = data.get(schemas.MANAGED_EVALS_DATASET_ID_COL)
        tags = data.get(schemas.TAGS_COL)
        if input_output_utils.is_none_or_nan(tags):
            tags = None
        # If tags is a list or tuple, convert to dict[str, str].
        if isinstance(tags, (list, tuple)):
            tags = {tag: "true" for tag in tags}
        return cls(
            question_id=question_id,
            question=question,
            raw_request=raw_request,
            has_inputs_outputs=has_inputs_outputs,
            answer=answer,
            raw_response=raw_response,
            retrieval_context=retrieved_context,
            ground_truth_answer=ground_truth_answer,
            ground_truth_retrieval_context=expected_retrieved_context,
            expected_facts=expected_facts,
            guidelines=guidelines,
            named_guidelines=named_guidelines,
            guidelines_context=guidelines_context,
            custom_expected=custom_expected,
            custom_inputs=custom_inputs,
            custom_outputs=custom_outputs,
            trace=trace,
            tool_calls=tool_calls,
            source_id=source_id,
            source_type=source_type,
            managed_evals_eval_id=managed_evals_eval_id,
            managed_evals_dataset_id=managed_evals_dataset_id,
            tags=tags,
        )

    def as_dict(self, *, use_chat_completion_request_format: bool = False) -> Dict[str, Any]:
        """
        Get as a dictionary. Keys are defined in schemas. Exclude None values.

        :param use_chat_completion_request_format: Whether to use the chat completion request format for the request.
        """
        request = self.raw_request or self.question
        if use_chat_completion_request_format:
            request = input_output_utils.to_chat_completion_request(self.question)
        response = self.raw_response or self.answer

        inputs = {
            schemas.REQUEST_ID_COL: self.question_id,
            # input
            schemas.REQUEST_COL: request,
            schemas.CUSTOM_INPUTS_COL: self.custom_inputs,
            # output
            schemas.RESPONSE_COL: response,
            schemas.RETRIEVED_CONTEXT_COL: (
                self.retrieval_context.to_output_dict() if self.retrieval_context else None
            ),
            schemas.CUSTOM_OUTPUTS_COL: self.custom_outputs,
            schemas.TRACE_COL: serialization_utils.serialize_trace(self.trace),
            schemas.TOOL_CALLS_COL: (
                [ToolCallInvocation.to_dict(tool_call) for tool_call in self.tool_calls]
                if self.tool_calls is not None
                else None
            ),
            schemas.MODEL_ERROR_MESSAGE_COL: self.model_error_message,
            # expected
            schemas.EXPECTED_RETRIEVED_CONTEXT_COL: (
                self.ground_truth_retrieval_context.to_output_dict() if self.ground_truth_retrieval_context else None
            ),
            schemas.EXPECTED_RESPONSE_COL: self.ground_truth_answer,
            schemas.EXPECTED_FACTS_COL: self.expected_facts,
            schemas.GUIDELINES_COL: self.raw_guidelines,
            schemas.GUIDELINES_CONTEXT_COL: self.guidelines_context,
            schemas.CUSTOM_EXPECTED_COL: self.custom_expected,
            # source related
            schemas.SOURCE_TYPE_COL: self.source_type,
            schemas.SOURCE_ID_COL: self.source_id,
            schemas.MANAGED_EVALS_EVAL_ID_COL: self.managed_evals_eval_id,
            schemas.MANAGED_EVALS_DATASET_ID_COL: self.managed_evals_dataset_id,
            schemas.TAGS_COL: self.tags,
        }
        return collection_utils.drop_none_values(inputs)


@dataclasses.dataclass
class AssessmentSource:
    source_id: str

    @classmethod
    def builtin(cls) -> "AssessmentSource":
        return cls(
            source_id="databricks",
        )

    @classmethod
    def custom(cls) -> "AssessmentSource":
        return cls(
            source_id="custom",
        )


@dataclasses.dataclass(frozen=True, eq=True)
class AssessmentResult:
    """Holds the result of an assessment."""

    assessment_name: str
    assessment_type: assessment_config.AssessmentType
    assessment_source: AssessmentSource

    def __lt__(self, other):
        if not isinstance(other, AssessmentResult):
            return NotImplemented
        # Compare by assessment_name
        return self.assessment_name < other.assessment_name


@dataclasses.dataclass(frozen=True, eq=True)
class PerRequestAssessmentResult(AssessmentResult):
    """Holds the result of a per-request assessment."""

    rating: Rating
    assessment_type: assessment_config.AssessmentType
    span_id: Optional[str] = None

    def to_mlflow_assessment(
        self,
        assessment_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None,
        root_span_id: Optional[str] = None,
    ) -> mlflow_entities.Feedback:
        """
        Convert a PerRequestAssessmentResult object to a MLflow Assessment object.
        :param assessment_name: The name of the assessment
        :param metadata: Additional metadata to add to the assessment
        :param trace_id: The trace ID of the trace associated
        :param root_span_id: The root span ID; if a more specific span ID is not available, this will be used
        :return: MLflow Assessment object
        """

        source = mlflow_entities.AssessmentSource(
            source_type=mlflow_entities.AssessmentSourceType.LLM_JUDGE,
            source_id=self.assessment_source.source_id or AssessmentSource.builtin().source_id,
        )
        if self.rating.error_message is not None:
            value_kwarg = {
                "error": mlflow_entities.AssessmentError(
                    error_code=self.rating.error_code,
                    error_message=self.rating.error_message,
                ),
                "value": None,
            }
        else:
            value = self.rating.categorical_value or self.rating.double_value
            value_kwarg = {"value": value}

        if trace_id is not None:
            value_kwarg["trace_id"] = trace_id
        if (self.span_id or root_span_id) is not None:
            value_kwarg["span_id"] = self.span_id or root_span_id
        if metadata is not None:
            value_kwarg["metadata"] = metadata

        return mlflow_entities.Feedback(
            name=assessment_name or self.assessment_name,
            source=source,
            rationale=self.rating.rationale,
            **value_kwarg,
        )


@dataclasses.dataclass(frozen=True, eq=True)
class PerChunkAssessmentResult(AssessmentResult):
    """Holds the result of a per-chunk assessment."""

    positional_rating: PositionalRating
    assessment_type: assessment_config.AssessmentType = dataclasses.field(
        init=False, default=assessment_config.AssessmentType.RETRIEVAL
    )
    span_id: Optional[str] = None

    def to_mlflow_assessment(
        self,
        assessment_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None,
        root_span_id: Optional[str] = None,
    ) -> List[mlflow_entities.Feedback]:
        """
        Convert a PerChunkAssessmentResult object to a MLflow Assessment object.
        :param assessment_name: The name of the assessment
        :param metadata: Additional metadata to add to the assessment
        :param trace_id: The trace ID of the trace associated
        :param root_span_id: The root span ID; if a more specific span ID is not available, this will be used
        :return: MLflow Assessment object
        """
        assessments = []
        for position, rating in self.positional_rating.items():
            if rating.error_message is not None:
                value_kwarg = {
                    "error": mlflow_entities.AssessmentError(
                        error_code=rating.error_code, error_message=rating.error_message
                    ),
                    "value": None,
                }
            else:
                value = rating.categorical_value or rating.double_value
                value_kwarg = {"value": value}

            if trace_id is not None:
                value_kwarg["trace_id"] = trace_id
            if (self.span_id or root_span_id) is not None:
                value_kwarg["span_id"] = self.span_id or root_span_id

            source = mlflow_entities.AssessmentSource(
                source_type=mlflow_entities.AssessmentSourceType.LLM_JUDGE,
                source_id=self.assessment_source.source_id or AssessmentSource.builtin().source_id,
            )
            assessments.append(
                mlflow_entities.Feedback(
                    name=assessment_name or self.assessment_name,
                    source=source,
                    rationale=rating.rationale,
                    metadata={_SPAN_OUTPUT_KEY: str(position), **(metadata or {})},
                    **value_kwarg,
                )
            )
        return assessments


@dataclasses.dataclass(frozen=True, eq=True)
class MetricResult:
    """Holds the result of a metric."""

    metric_value: mlflow_entities.Assessment
    legacy_metric: bool = False
    aggregations: Optional[List[Union[str, Callable]]] = None

    @staticmethod
    def make_legacy_metric(metric_name, metric_value, **kwargs):
        """
        Convenience constructor that also sets a legacy flag. "Legacy metric" implies a few things:
        1. When we log evaluations to mlflow using the old API, legacy metrics get logged under
           _metrics.json rather than _assessments.json
        2. They have some special handling for how their column names get generated in the results dataframe.
           Specifically, they get created without the /value, /rationale, /error suffixes.
        """
        return MetricResult(
            metric_value=mlflow_entities.Feedback(
                name=metric_name,
                source=mlflow_entities.AssessmentSource(
                    source_type=mlflow_entities.AssessmentSourceType.CODE,
                    source_id=metric_name,
                ),
                value=metric_value,
                **kwargs,
            ),
            legacy_metric=True,
        )

    def to_mlflow_assessment(
        self,
        assessment_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
    ) -> mlflow_entities.Assessment:
        """
        Convert a MetricResult object to a MLflow Assessment object.

        This function is deprecated; eventually mlflow.log_feedback will directly accept
        mlflow_entities.Assessment objects for feedback and internalize all of the conversions here.

        :param assessment_name: The name of the assessment
        :param metadata: Additional metadata to add to the assessment
        :param trace_id: The trace ID of the trace associated if the trace ID is not already set
        :param span_id: The span ID of the span associated if the span ID is not already set
        :return: MLflow Assessment object
        """
        if trace_id is not None and self.metric_value.trace_id is None:
            self.metric_value.trace_id = trace_id
        if span_id is not None and self.metric_value.span_id is None:
            self.metric_value.span_id = span_id
        if assessment_name is not None:
            self.metric_value.name = assessment_name
        if metadata is not None:
            self.metric_value.metadata.update(metadata or {})
        return self.metric_value


@dataclasses.dataclass
class EvalResult:
    """Holds the result of the evaluation for an eval item."""

    eval_item: EvalItem
    _assessments: List[mlflow_entities.Assessment] = dataclasses.field(default_factory=list)
    # Deprecated: Use eval_item.trace.info.assessments
    assessment_results: List[AssessmentResult] = dataclasses.field(default_factory=list)

    metric_results: List[MetricResult] = dataclasses.field(default_factory=list)
    """A collection of MetricResult."""

    eval_error: Optional[str] = None
    """
    Error message encountered in processing the eval item.
    """

    def __eq__(self, other):
        if not isinstance(other, EvalResult):
            return False
        # noinspection PyTypeChecker
        return (
            self.eval_item == other.eval_item
            and sorted(self._assessments) == sorted(other._assessments)
            and sorted(self.assessment_results) == sorted(other.assessment_results)
            and sorted(self.metric_results, key=lambda m: m.metric_value.name)
            == sorted(other.metric_results, key=lambda m: m.metric_value.name)
            and self.eval_error == other.eval_error
        )

    @property
    def assessments(self) -> List[mlflow_entities.Assessment]:
        """Temporary shim to return assessments in the new format.

        At first, this method will translate the old assessments (V2) to the new format.

        Eventually we will switch over to directly producing mlflow_entities.Assessments,
        and when the switchover is complete, the self._assessments property will simply become
        self.assessments and this @property shim will be dropped.

        These assessments (V3) are destined to be logged to mlflow via log_feedback(), which will
        eventually replace the existing mechanism where assessmentsV2 are written to an
        _assessments.json file.
        """
        converted_assessment_results: list[mlflow_entities.Assessment] = []
        converted_metric_results: list[mlflow_entities.Assessment] = []
        converted_expectations: list[mlflow_entities.Assessment] = []

        trace_id = self.eval_item.trace.info.trace_id
        root_span = trace_utils.get_root_span(self.eval_item.trace)
        root_span_id = root_span.span_id if root_span is not None else None

        for metric in self.metric_results:
            if metric.metric_value.name in _EXCLUDED_METRICS_FROM_LOGGING:
                continue
            converted_metric_results.append(metric.to_mlflow_assessment(trace_id=trace_id, span_id=root_span_id))

        for assessment in self.assessment_results:
            if isinstance(assessment, PerRequestAssessmentResult):
                converted_assessment_results.append(
                    assessment.to_mlflow_assessment(trace_id=trace_id, root_span_id=root_span_id)
                )
            elif isinstance(assessment, PerChunkAssessmentResult):
                converted_assessment_results.extend(
                    assessment.to_mlflow_assessment(trace_id=trace_id, root_span_id=root_span_id)
                )

        def _expectation_obj_to_json_str(expectation_obj: Any) -> str:
            """Convert an arbitrary expectation object to a JSON string."""
            if isinstance(expectation_obj, str):
                return expectation_obj

            try:
                # Convert to JSON string with handling for nested objects
                return json.dumps(expectation_obj, default=lambda o: o.__dict__)
            except:  # noqa: E722
                return str(expectation_obj)

        eval_item_dict = self.eval_item.as_dict()
        expectations = {key: value for key, value in eval_item_dict.items() if key in _EXPECTATION_FIELDS}

        # Unpack custom expectations into individual expectations
        custom_expectations = eval_item_dict.get(schemas.CUSTOM_EXPECTED_COL, {})
        expectations.update(custom_expectations)

        for expectation_name, expectation_value in expectations.items():
            # ExpectationValue values can only hold primitives or a list of primitives. As such,
            # we need to convert objects such as retrieved documents to a JSON string.
            processed_expectation_value = expectation_value
            if isinstance(expectation_value, list) and not all(isinstance(value, str) for value in expectation_value):
                processed_expectation_value = [
                    _expectation_obj_to_json_str(value) for value in expectation_value if value is not None
                ]
            elif isinstance(expectation_value, dict):
                processed_expectation_value = _expectation_obj_to_json_str(expectation_value)

            source_id = context.get_context().get_user_name()
            converted_expectations.append(
                mlflow_entities.Expectation(
                    trace_id=trace_id,
                    span_id=root_span_id,
                    name=expectation_name,
                    source=mlflow_entities.AssessmentSource(
                        source_type=mlflow_entities.AssessmentSourceType.HUMAN,
                        source_id=source_id or "unknown",
                    ),
                    value=processed_expectation_value,
                )
            )

        return self._assessments + converted_assessment_results + converted_metric_results + converted_expectations

    def get_metrics_dict(self) -> Dict[str, Any]:
        """Get the metrics as a dictionary. Keys are defined in schemas."""
        metrics: Dict[str, Any] = {
            metric.metric_value.name: metric.metric_value.feedback.value
            for metric in self.metric_results
            if metric.legacy_metric
        }
        # Remove None values in metrics
        return collection_utils.drop_none_values(metrics)

    def get_assessment_results_dict(self) -> Dict[str, schemas.ASSESSMENT_RESULT_TYPE]:
        """Get the assessment results as a dictionary. Keys are defined in schemas."""
        assessments: Dict[str, schemas.ASSESSMENT_RESULT_TYPE] = {}
        for assessment in self.assessment_results:
            # TODO(ML-45046): remove assessment type lookup in harness, rely on service
            # Get the assessment type from the built-in metrics. If the metric is not found, use the provided assessment type.
            try:
                builtin_assessment_config = (
                    assessment_config.get_builtin_assessment_config_with_service_assessment_name(
                        assessment.assessment_name
                    )
                )
                assessment_type = builtin_assessment_config.assessment_type
            except ValueError:
                assessment_type = assessment.assessment_type

            if (
                isinstance(assessment, PerRequestAssessmentResult)
                and assessment_type == assessment_config.AssessmentType.RETRIEVAL_LIST
            ):
                if assessment.rating.categorical_value is not None:
                    assessments[
                        schemas.get_retrieval_llm_rating_col_name(assessment.assessment_name, is_per_chunk=False)
                    ] = assessment.rating.categorical_value
                if assessment.rating.rationale is not None:
                    assessments[
                        schemas.get_retrieval_llm_rationale_col_name(assessment.assessment_name, is_per_chunk=False)
                    ] = assessment.rating.rationale
                if assessment.rating.error_message is not None:
                    assessments[
                        schemas.get_retrieval_llm_error_message_col_name(assessment.assessment_name, is_per_chunk=False)
                    ] = assessment.rating.error_message
            elif isinstance(assessment, PerRequestAssessmentResult):
                if assessment.rating.categorical_value is not None:
                    assessments[schemas.get_response_llm_rating_col_name(assessment.assessment_name)] = (
                        assessment.rating.categorical_value
                    )
                if assessment.rating.rationale is not None:
                    assessments[schemas.get_response_llm_rationale_col_name(assessment.assessment_name)] = (
                        assessment.rating.rationale
                    )
                if assessment.rating.error_message is not None:
                    assessments[schemas.get_response_llm_error_message_col_name(assessment.assessment_name)] = (
                        assessment.rating.error_message
                    )
            elif isinstance(assessment, PerChunkAssessmentResult):
                # Convert the positional_rating to a list of ratings ordered by position
                # For missing positions, use an error rating. This should not happen in practice.
                ratings_ordered_by_position: List[Rating] = collection_utils.position_map_to_list(
                    assessment.positional_rating,
                    default=Rating.error("Missing rating"),
                )
                if any(rating.categorical_value is not None for rating in ratings_ordered_by_position):
                    assessments[schemas.get_retrieval_llm_rating_col_name(assessment.assessment_name)] = [
                        rating.categorical_value for rating in ratings_ordered_by_position
                    ]
                if any(rating.rationale is not None for rating in ratings_ordered_by_position):
                    assessments[schemas.get_retrieval_llm_rationale_col_name(assessment.assessment_name)] = [
                        rating.rationale for rating in ratings_ordered_by_position
                    ]
                if any(rating.error_message is not None for rating in ratings_ordered_by_position):
                    assessments[schemas.get_retrieval_llm_error_message_col_name(assessment.assessment_name)] = [
                        rating.error_message for rating in ratings_ordered_by_position
                    ]
        for metric in self.metric_results:
            if metric.legacy_metric:
                continue
            if metric.metric_value.feedback.value is not None:
                assessments[f"{metric.metric_value.name}/value"] = metric.metric_value.feedback.value

            if metric.metric_value.rationale is not None:
                assessments[f"{metric.metric_value.name}/rationale"] = metric.metric_value.rationale

            metric_error = metric.metric_value.error
            if metric_error is not None and metric_error.error_message is not None:
                assessments[f"{metric.metric_value.name}/error_message"] = metric_error.error_message

            if metric_error is not None and metric_error.error_code is not None:
                assessments[f"{metric.metric_value.name}/error_code"] = metric_error.error_code
        return assessments

    def get_assessments_dict(self) -> Dict[str, schemas.ASSESSMENT_RESULT_TYPE]:
        """Backwards compatibility shim.

        As we migrate assessments and metrics to directly outputting mlflow.entities.Assessment
        instead of our mix of AssessmentResult and MetricResult, we will want to maintain the
        original output dataframe columns.
        """
        return {}

    def to_pd_series(self) -> pd.Series:
        """Converts the EvalResult to a flattened pd.Series."""
        inputs = self.eval_item.as_dict()
        assessment_results = self.get_assessment_results_dict()
        metrics = self.get_metrics_dict()
        assessments = self.get_assessments_dict()

        # Merge dictionaries and convert to pd.Series
        combined_data = {
            **inputs,
            **assessment_results,
            **metrics,
            **assessments,
        }
        return pd.Series(combined_data)
