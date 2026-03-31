import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Union

from mlflow import entities as mlflow_entities

from databricks.rag_eval import context, schemas
from databricks.rag_eval.config import assessment_config
from databricks.rag_eval.evaluation import entities
from databricks.rag_eval.utils import input_output_utils

USER_DEFINED_ASSESSMENT_NAME_KEY = "_user_defined_assessment_name"

GuidelinesType = Union[str, List[str]]
LegacyGuidelinesType = Union[List[str], Dict[str, List[str]]]


class CallableBuiltinJudge:
    """
    Callable object that can be used to evaluate inputs against
    the LLM judge service with the current assessment config.

    Args:
        config: The assessment config to use for the judge.
        assessment_name: Optional assessment override. If present, the result Assessment will have this name instead of the original built-in judge's name.
    """

    def __init__(
        self,
        config: assessment_config.BuiltinAssessmentConfig,
        assessment_name: Optional[str] = None,
    ):
        self.config = config
        self.assessment_name = assessment_name

    @context.eval_context
    def __call__(
        self,
        *,
        request: Optional[str | Dict[str, Any]] = None,
        response: Optional[str | Dict[str, Any]] = None,
        retrieved_context: Optional[List[Dict[str, Any]] | Any] = None,
        expected_response: Optional[str] = None,
        expected_retrieved_context: Optional[List[Dict[str, Any]]] = None,
        expected_facts: Optional[List[str]] = None,
        guidelines: Optional[Union[LegacyGuidelinesType, GuidelinesType]] = None,
        guidelines_context: Optional[Dict[str, Any]] = None,
    ) -> Union[mlflow_entities.Assessment, List[mlflow_entities.Assessment]]:
        input_output_utils.request_to_string(request)
        input_output_utils.response_to_string(response)

        # Retrieved context is used for documents as well as other types of context (e.g., tool
        # calls). As such, we accept an arbitrary format. If it is not a valid format, we JSON
        # serialize it and wrap it in a document format.
        if not _is_expected_retrieved_context_format(retrieved_context):
            try:
                retrieved_context = [{"content": json.dumps(retrieved_context)}]
            except Exception as e:
                raise ValueError("retrieved_context must be json serializable.") from e

        _validate_string_input("expected_response", expected_response)
        _validate_retrieved_context("expected_retrieved_context", expected_retrieved_context)
        _validate_repeated_string_input("expected_facts", expected_facts)

        # Use appropriate validation based on the config
        if self.config == assessment_config.GUIDELINE_ADHERENCE:
            _validate_legacy_guidelines(guidelines)
        else:
            _validate_guidelines(guidelines)

        _validate_guidelines_context(guidelines_context)

        managed_rag_client = context.get_context().build_managed_rag_client()
        eval_item = entities.EvalItem.from_dict(
            {
                schemas.REQUEST_COL: request,
                schemas.RESPONSE_COL: response,
                schemas.RETRIEVED_CONTEXT_COL: retrieved_context,
                schemas.EXPECTED_RESPONSE_COL: expected_response,
                schemas.EXPECTED_RETRIEVED_CONTEXT_COL: expected_retrieved_context,
                schemas.EXPECTED_FACTS_COL: expected_facts,
                schemas.GUIDELINES_COL: guidelines,
                schemas.GUIDELINES_CONTEXT_COL: (
                    {key: str(value) for key, value in guidelines_context.items()}
                    if guidelines_context is not None
                    else None
                ),
            }
        )

        assessment_kwargs = {
            "metadata": {USER_DEFINED_ASSESSMENT_NAME_KEY: str(self.assessment_name is not None).lower()}
        }

        # Guideline adherence requires special processing due to named guidelines
        if self.config == assessment_config.GUIDELINE_ADHERENCE:
            guideline_adherence_assessment_name = assessment_config.GUIDELINE_ADHERENCE.assessment_name
            is_named_guidelines = not (
                eval_item.named_guidelines is not None
                and len(eval_item.named_guidelines) == 1
                and guideline_adherence_assessment_name in eval_item.named_guidelines
            )

            assessments = []
            for guidelines_name, grouped_guidelines in (eval_item.named_guidelines or {}).items():
                # Replace the named guidelines for each eval with the respective group's guidelines
                guidelines_eval_item = eval_item.as_dict()
                guidelines_eval_item[schemas.GUIDELINES_COL] = grouped_guidelines

                overall_assessment_name = self.assessment_name or assessment_config.GUIDELINE_ADHERENCE.assessment_name
                # Use two-tiered name for named guidelines
                user_facing_assessment_name = (
                    f"{overall_assessment_name}/{guidelines_name}" if is_named_guidelines else overall_assessment_name
                )

                experiment_id = context.get_context().get_mlflow_experiment_id()
                assessment_results = managed_rag_client.get_assessment(
                    eval_item=entities.EvalItem.from_dict(guidelines_eval_item),
                    # Replace the user-facing name for each guidelines assessment
                    config=assessment_config.BuiltinAssessmentConfig(
                        assessment_name=guideline_adherence_assessment_name,
                        user_facing_assessment_name=user_facing_assessment_name,
                        assessment_type=assessment_config.AssessmentType.ANSWER,
                    ),
                    experiment_id=experiment_id,
                )
                if assessment_results:
                    assessments.append(
                        assessment_results[0].to_mlflow_assessment(
                            assessment_name=user_facing_assessment_name,
                            **assessment_kwargs,
                        )
                    )
            return assessments if is_named_guidelines else assessments[0]

        else:
            experiment_id = context.get_context().get_mlflow_experiment_id()

            if self.config == assessment_config.CHUNK_RELEVANCE:
                assessment_results = _get_chunk_relevance_assessments(
                    managed_rag_client=managed_rag_client,
                    eval_item=eval_item,
                    experiment_id=experiment_id,
                )
            else:
                assessment_results = managed_rag_client.get_assessment(
                    eval_item=eval_item,
                    config=self.config,
                    experiment_id=experiment_id,
                )

            return (
                assessment_results[0].to_mlflow_assessment(
                    assessment_name=self.assessment_name,
                    **assessment_kwargs,
                )
                if assessment_results
                else []
            )


def _get_chunk_relevance_assessments(
    managed_rag_client,
    eval_item: entities.EvalItem,
    experiment_id: Optional[str],
) -> List[entities.AssessmentResult]:
    """
    Process chunk relevance assessments in parallel to improve performance.

    This function processes each chunk individually in parallel, allowing the existing retry
    logic in managed_rag_client to handle failures per-chunk instead of retrying the entire batch.

    Args:
        managed_rag_client: Client to call assessment service
        eval_item: Evaluation item with all chunks
        experiment_id: MLflow experiment ID

    Returns:
        List containing a single PerChunkAssessmentResult with all chunk ratings
    """
    if not eval_item.retrieval_context or not eval_item.retrieval_context.chunks:
        return []

    chunks = eval_item.retrieval_context.chunks
    positional_rating = {}

    def process_chunk(position: int, chunk: entities.Chunk) -> tuple[int, Optional[entities.Rating]]:
        """Process a single chunk and return its position and rating."""
        single_chunk_eval_item = entities.EvalItem.from_dict(
            {
                schemas.REQUEST_COL: eval_item.question,
                schemas.RETRIEVED_CONTEXT_COL: [chunk.to_dict()],
            }
        )

        chunk_results = managed_rag_client.get_assessment(
            eval_item=single_chunk_eval_item,
            config=assessment_config.CHUNK_RELEVANCE,
            experiment_id=experiment_id,
        )

        if chunk_results and isinstance(chunk_results[0], entities.PerChunkAssessmentResult):
            per_chunk_result = chunk_results[0].positional_rating
            if per_chunk_result:
                single_chunk_rating = next(iter(per_chunk_result.values()))
                return (position, single_chunk_rating)
        return (position, None)

    with ThreadPoolExecutor() as executor:
        future_to_position = {
            executor.submit(process_chunk, position, chunk): position for position, chunk in enumerate(chunks, start=1)
        }

        for future in as_completed(future_to_position):
            position, rating = future.result()
            if rating is not None:
                positional_rating[position] = rating

    span_id = eval_item.retrieval_context.span_id if eval_item.retrieval_context else None

    return [
        entities.PerChunkAssessmentResult(
            assessment_name=assessment_config.CHUNK_RELEVANCE.assessment_name,
            assessment_source=entities.AssessmentSource.builtin(),
            positional_rating=positional_rating,
            span_id=span_id,
        )
    ]


def _validate_string_input(param_name: str, input_value: Any) -> None:
    if input_value and not isinstance(input_value, str):
        raise ValueError(f"{param_name} must be a string. Got: {type(input_value)}")


def _validate_retrieved_context(param_name: str, retrieved_context: Optional[List[Dict[str, Any]]]) -> None:
    if retrieved_context:
        if not isinstance(retrieved_context, list):
            raise ValueError(f"{param_name} must be a list of dictionaries. Got: {type(retrieved_context)}")
        for context_dict in retrieved_context:
            if not isinstance(context_dict, dict):
                raise ValueError(f"{param_name} must be a list of dictionaries. Got list of: {type(context_dict)}")
            if "content" not in context_dict:
                raise ValueError(f"Each context in {param_name} must have a 'content' key. Got: {context_dict}")
            if set(context_dict.keys()) - {"doc_uri", "content"}:
                raise ValueError(
                    f"Each context in {param_name} must have only 'doc_uri' and 'content' keys. Got: {context_dict}"
                )


def _is_expected_retrieved_context_format(retrieved_context: Optional[List[Dict[str, Any]]] | Any) -> bool:
    if retrieved_context:
        if not isinstance(retrieved_context, list):
            return False
        for context_dict in retrieved_context:
            if (
                not isinstance(context_dict, dict)
                or "content" not in context_dict
                or set(context_dict.keys()) - {"doc_uri", "content"}
            ):
                return False
    return True


def _validate_repeated_string_input(param_name: str, input_value: Any) -> None:
    if input_value is None:
        return
    elif not isinstance(input_value, list):
        raise ValueError(f"{param_name} must be a list. Got: {type(input_value)}")

    for idx, value in enumerate(input_value):
        if not isinstance(value, str):
            raise ValueError(f"{param_name} must be a list of strings. Got: {type(value)} at index: {idx}")


def _validate_legacy_guidelines(guidelines: Optional[LegacyGuidelinesType]) -> None:
    """Validates guidelines for the legacy guideline_adherence judge.

    Accepts List[str] or Dict[str, List[str]]
    """
    if guidelines is None:
        return

    guidelines_is_valid_iterable = input_output_utils.is_valid_guidelines_iterable(guidelines)
    guidelines_is_valid_mapping = input_output_utils.is_valid_guidelines_mapping(guidelines)

    if not (guidelines_is_valid_iterable or guidelines_is_valid_mapping):
        raise ValueError(
            f"Invalid guidelines: {guidelines}. Guidelines must be a list of strings "
            f"or a mapping from a name of guidelines (string) to a list of strings."
        )
    elif guidelines_is_valid_iterable:
        input_output_utils.check_guidelines_iterable_exceeds_limit(guidelines)
    elif guidelines_is_valid_mapping:
        input_output_utils.check_guidelines_mapping_exceeds_limit(guidelines)


def _validate_guidelines(guidelines: Optional[GuidelinesType]) -> None:
    """Validates guidelines for the new guidelines judge.

    Accepts str or List[str].
    """
    if guidelines is None:
        return

    guidelines_is_valid_iterable = input_output_utils.is_valid_guidelines_iterable(guidelines)

    if isinstance(guidelines, str):
        return
    elif guidelines_is_valid_iterable:
        input_output_utils.check_guidelines_iterable_exceeds_limit(guidelines)
    else:
        raise ValueError(f"Invalid guidelines: {guidelines}. Guidelines must be a string or list of strings")


def _validate_guidelines_context(guidelines_context: Optional[Dict[str, Any]]) -> None:
    if guidelines_context is None:
        return

    if not isinstance(guidelines_context, dict):
        raise ValueError(f"guidelines_context must be a dictionary. Got: {type(guidelines_context)}")

    for key in guidelines_context.keys():
        if not isinstance(key, str):
            raise ValueError(f"guidelines_context keys must be strings. Got: {type(key)}")


# use this docstring for the CallableBuiltinJudge class
CALLABLE_BUILTIN_JUDGE_DOCSTRING = """
        {judge_description}

        Args:
            request: Input to the application to evaluate, user’s question or query. For example, “What is RAG?”.
            response: Response generated by the application being evaluated.
            retrieved_context: Retrieval results generated by the retriever in the application being evaluated. 
                It should be a list of dictionaries with the following keys:
                    - doc_uri (Optional): The doc_uri of the context.
                    - content: The content of the context.
            expected_response: Ground-truth (correct) answer for the input request.
            expected_retrieved_context: Array of objects containing the expected retrieved context for the request 
                (if the application includes a retrieval step). It should be a list of dictionaries with the
                following keys:
                    - doc_uri (Optional): The doc_uri of the context.
                    - content: The content of the context.
            expected_facts: Array of strings containing facts expected in the correct response for the input request.
            guidelines: Array of strings containing the guidelines that the response should adhere to.
        Required input arguments:
            {required_args}

        Returns:
            Assessment result for the given input.
        """


# =================== Builtin Judges ===================
def correctness(
    request: str | Dict[str, Any],
    response: str | Dict[str, Any],
    expected_response: Optional[str] = None,
    expected_facts: Optional[List[str]] = None,
    assessment_name: Optional[str] = None,
) -> mlflow_entities.Assessment:
    """
    The correctness LLM judge gives a binary evaluation and written rationale on whether the
    response generated by the agent is factually accurate and semantically similar to the provided
    expected response or expected facts.

    Args:
        request: Input to the application to evaluate, user’s question or query. For example, “What is RAG?”.
        response: Response generated by the application being evaluated.
        expected_response: Ground-truth (correct) answer for the input request.
        expected_facts: Array of strings containing facts expected in the correct response for the input request.
        assessment_name: Optional override for the assessment name.  If present, the output Assessment will use this as the name instead of "correctness"
    Required input arguments:
        request, response, oneof(expected_response, expected_facts)

    Returns:
        Correctness assessment result for the given input.
    """
    return CallableBuiltinJudge(config=assessment_config.CORRECTNESS, assessment_name=assessment_name)(
        request=request,
        response=response,
        expected_response=expected_response,
        expected_facts=expected_facts,
    )


def groundedness(
    request: str | Dict[str, Any],
    response: str | Dict[str, Any],
    retrieved_context: List[Dict[str, Any]] | Any,
    assessment_name: Optional[str] = None,
) -> mlflow_entities.Assessment:
    """
    The groundedness LLM judge returns a binary evaluation and written rationale on whether the
    generated response is factually consistent with the retrieved context.

    Args:
        request: Input to the application to evaluate, user’s question or query. For example, “What is RAG?”.
        response: Response generated by the application being evaluated.
        retrieved_context: Retrieval results generated by the retriever in the application being evaluated.
                It should be a list of dictionaries with the following keys:
                    - doc_uri (Optional): The doc_uri of the context.
                    - content: The content of the context.
        assessment_name: Optional override for the assessment name.  If present, the output Assessment will use this as the name instead of "groundedness"
    Required input arguments:
        request, response, retrieved_context

    Returns:
        Groundedness assessment result for the given input.
    """
    return CallableBuiltinJudge(config=assessment_config.GROUNDEDNESS, assessment_name=assessment_name)(
        request=request,
        response=response,
        retrieved_context=retrieved_context,
    )


def safety(
    request: Optional[str | Dict[str, Any]] = None,
    response: Optional[str | Dict[str, Any]] = None,
    assessment_name: Optional[str] = None,
) -> mlflow_entities.Assessment:
    """
    The safety LLM judge returns a binary rating and a written rationale on whether the generated
    response has harmful or toxic content.

    Args:
        request: Input to the application to evaluate, user’s question or query. For example, “What is RAG?”.
        response: Response generated by the application being evaluated.
        assessment_name: Optional override for the assessment name.  If present, the output Assessment will use this as the name instead of "safety"
    Required input arguments:
        response

    Returns:
        Safety assessment result for the given input.
    """
    if response is None:
        raise ValueError("`response` is required for the safety judge.")

    return CallableBuiltinJudge(config=assessment_config.HARMFULNESS, assessment_name=assessment_name)(
        request=request,
        response=response,
    )


def relevance_to_query(
    request: str | Dict[str, Any],
    response: str | Dict[str, Any],
    assessment_name: Optional[str] = None,
) -> mlflow_entities.Assessment:
    """
    The relevance_to_query LLM judge determines whether the response is relevant to the input request.

    Args:
        request: Input to the application to evaluate, user’s question or query. For example, “What is RAG?”.
        response: Response generated by the application being evaluated.
        assessment_name: Optional override for the assessment name.  If present, the output Assessment will use this as the name instead of "relevance_to_query"
    Required input arguments:
        request, response

    Returns:
        Relevance to query assessment result for the given input.
    """
    return CallableBuiltinJudge(config=assessment_config.RELEVANCE_TO_QUERY, assessment_name=assessment_name)(
        request=request,
        response=response,
    )


def chunk_relevance(
    request: str | Dict[str, Any],
    retrieved_context: List[Dict[str, Any]] | Any,
    assessment_name: Optional[str] = None,
) -> List[mlflow_entities.Assessment]:
    """
    The chunk-relevance-precision LLM judge determines whether the chunks returned by the retriever
    are relevant to the input request. Precision is calculated as the number of relevant chunks
    returned divided by the total number of chunks returned. For example, if the retriever returns
    four chunks, and the LLM judge determines that three of the four returned documents are relevant
    to the request, then llm_judged/chunk_relevance/precision is 0.75.

    Args:
        request: Input to the application to evaluate, user’s question or query. For example, “What is RAG?”.
        retrieved_context: Retrieval results generated by the retriever in the application being evaluated.
                It should be a list of dictionaries with the following keys:
                    - doc_uri (Optional): The doc_uri of the context.
                    - content: The content of the context.
        assessment_name: Optional override for the assessment name.  If present, the output Assessment will use this as the name instead of "chunk_relevance"
    Required input arguments:
        request, retrieved_context

    Returns:
        Chunk relevance assessment result for each of the chunks in the given input.
    """
    return CallableBuiltinJudge(config=assessment_config.CHUNK_RELEVANCE, assessment_name=assessment_name)(
        request=request,
        retrieved_context=retrieved_context,
    )


def context_sufficiency(
    request: str | Dict[str, Any],
    retrieved_context: List[Dict[str, Any]] | Any,
    expected_response: Optional[str] = None,
    expected_facts: Optional[List[str]] = None,
    assessment_name: Optional[str] = None,
) -> mlflow_entities.Assessment:
    """
    The context_sufficiency LLM judge determines whether the retriever has retrieved documents that are
    sufficient to produce the expected response or expected facts.

    Args:
        request: Input to the application to evaluate, user’s question or query. For example, “What is RAG?”.
        expected_response: Ground-truth (correct) answer for the input request.
        retrieved_context: Retrieval results generated by the retriever in the application being evaluated.
                It should be a list of dictionaries with the following keys:
                    - doc_uri (Optional): The doc_uri of the context.
                    - content: The content of the context.
        expected_facts: Array of strings containing facts expected in the correct response for the input request.
        assessment_name: Optional override for the assessment name.  If present, the output Assessment will use this as the name instead of "context_sufficiency"
    Required input arguments:
        request, retrieved_context, oneof(expected_response, expected_facts)

    Returns:
        Context sufficiency assessment result for the given input.
    """
    return CallableBuiltinJudge(config=assessment_config.CONTEXT_SUFFICIENCY, assessment_name=assessment_name)(
        request=request,
        retrieved_context=retrieved_context,
        expected_response=expected_response,
        expected_facts=expected_facts,
    )


def guideline_adherence(
    request: Optional[str | Dict[str, Any]] = None,
    guidelines: Optional[Union[List[str], Dict[str, List[str]]]] = None,
    response: Optional[str | Dict[str, Any]] = None,
    guidelines_context: Optional[Dict[str, Any]] = None,
    assessment_name: Optional[str] = None,
) -> Union[mlflow_entities.Assessment, List[mlflow_entities.Assessment]]:
    """
    .. deprecated::
        The guideline_adherence function is deprecated. Use the guidelines function instead.

    The guideline_adherence LLM judge determines whether the provided context (one of or both of
    guidelines context or the response to the request) adheres to the provided guidelines.

    Args:
        request: Input to the application to evaluate, user’s question or query. For example, “What is RAG?”.
        guidelines: One of the following:
         - Array of strings containing the guidelines that the response or context should adhere to.
         - Mapping of string (named guidelines) to array of strings containing the guidelines the response or context should adhere to.
        response: Response generated by the application being evaluated.
        guidelines_context: Mapping of a string (context field name) to any object (content) containing context the guidelines can apply to. The values in the mapping will be cast to strings.
        assessment_name: Optional override for the assessment name.  If present, the output Assessment will use this as the name instead of "guideline_adherence"
    Required input arguments:
        guidelines, oneof(request, response, guidelines_context)

    Returns:
        Guideline adherence assessment(s) result for the given input. Returns a list when named guidelines are provided.
    """
    if guidelines is None:
        raise ValueError("`guidelines` is required for the guideline_adherence judge.")

    return CallableBuiltinJudge(config=assessment_config.GUIDELINE_ADHERENCE, assessment_name=assessment_name)(
        request=request,
        # TODO(ML-52218): Remove this temporary workaround to pass empty response
        response=response or " ",
        guidelines=guidelines,
        guidelines_context=guidelines_context,
    )


def guidelines(
    guidelines: Union[str, List[str]],
    context: Dict[str, Any],
    assessment_name: Optional[str] = None,
) -> mlflow_entities.Assessment:
    """
    The guidelines LLM judge determines whether the provided context adheres to the provided guidelines.

    Args:
        guidelines: Array of strings containing the guidelines that the context should adhere to.
        context: Mapping of a string (context field name) to any object (content) containing context the guidelines can apply to. The values in the mapping will be cast to strings.
        assessment_name: Optional override for the assessment name. If present, the output Assessment will use this as the name instead of "guidelines"

    Returns:
        Guidelines assessment result for the given input.
    """
    if guidelines is None:
        raise ValueError("`guidelines` is required for the guidelines judge.")

    if context is None:
        raise ValueError("`context` is required for the guidelines judge.")

    return CallableBuiltinJudge(config=assessment_config.GUIDELINES, assessment_name=assessment_name)(
        guidelines=[guidelines] if isinstance(guidelines, str) else guidelines,
        guidelines_context=context,
    )
