from typing import List, TypeVar, Union

######################################################################
# Column/field names used in mlflow EvaluationDataset DataFrames
######################################################################
DOC_URI_COL = "doc_uri"
CHUNK_CONTENT_COL = "content"
TRACE_COL = "trace"
REQUEST_ID_COL = "request_id"
REQUEST_COL = "request"
INPUTS_COL = "inputs"

# Multiturn input field names that should be combined into inputs
PERSONA_COL = "persona"
GOAL_COL = "goal"
CONTEXT_COL = "context"
MULTITURN_INPUT_COLS = [PERSONA_COL, GOAL_COL, CONTEXT_COL]
MULTITURN_IDENTIFIER_COLS = {GOAL_COL}
MULTITURN_INPUT_COLS_SET = frozenset(MULTITURN_INPUT_COLS)
MULTITURN_COLS_STR = ", ".join(MULTITURN_INPUT_COLS)  # For error messages

EXPECTED_RETRIEVED_CONTEXT_COL = "expected_retrieved_context"
EXPECTED_RESPONSE_COL = "expected_response"
RESPONSE_COL = "response"
OUTPUTS_COL = "outputs"
EXPECTATIONS_COL = "expectations"
TAGS_COL = "tags"
RETRIEVED_CONTEXT_COL = "retrieved_context"
GRADING_NOTES_COL = "grading_notes"
EXPECTED_FACTS_COL = "expected_facts"
GUIDELINES_COL = "guidelines"
GUIDELINES_CONTEXT_COL = "guidelines_context"
TOOL_CALLS_COL = "tool_calls"
CUSTOM_EXPECTED_COL = "custom_expected"
CUSTOM_INPUTS_COL = "custom_inputs"
CUSTOM_OUTPUTS_COL = "custom_outputs"
SOURCE_TYPE_COL = "source_type"
SOURCE_ID_COL = "source_id"
SOURCE_COL = "source"
MANAGED_EVALS_EVAL_ID_COL = "managed_evals_eval_id"
MANAGED_EVALS_DATASET_ID_COL = "managed_evals_dataset_id"

# Model error message column
MODEL_ERROR_MESSAGE_COL = "model_error_message"

######################################################################
# Column/field names for the output pandas DataFrame of mlflow.evaluate
######################################################################
_AGENT_PREFIX = "agent/"

LATENCY_SECONDS_COL = _AGENT_PREFIX + "latency_seconds"

GROUND_TRUTH_DOCUMENT_PREFIX = "document_"

_RATINGS_SUFFIX = "/ratings"
_RATIONALES_SUFFIX = "/rationales"
_ERROR_MESSAGES_SUFFIX = "/error_messages"

_RATING_SUFFIX = "/rating"
_RATIONALE_SUFFIX = "/rationale"
_ERROR_MESSAGE_SUFFIX = "/error_message"

######################################################################
# Data types for the output pandas DataFrame of mlflow.evaluate
######################################################################
ASSESSMENT_RESULT_TYPE: TypeVar = TypeVar("ASSESSMENT_RESULT_TYPE", bool, str, None, List[Union[bool, str, None]])
TRACE_METADATA_RUN_ID = "run_id"


######################################################################
# Helper methods to get column names for the output pandas DataFrame of mlflow.evaluate
######################################################################


def get_response_llm_rating_col_name(assessment_name: str) -> str:
    """Returns the column name for the LLM judged response metric rating"""
    return f"{assessment_name}{_RATING_SUFFIX}"


def get_response_llm_rationale_col_name(assessment_name: str) -> str:
    """Returns the column name for the LLM judged response metric rationale"""
    return f"{assessment_name}{_RATIONALE_SUFFIX}"


def get_response_llm_error_message_col_name(assessment_name: str) -> str:
    """Returns the column name for the LLM judged response metric error message"""
    return f"{assessment_name}{_ERROR_MESSAGE_SUFFIX}"


def get_retrieval_llm_rating_col_name(assessment_name: str, is_per_chunk: bool = True) -> str:
    """Returns the column name for the LLM judged retrieval metric rating"""
    suffix = _RATINGS_SUFFIX if is_per_chunk else _RATING_SUFFIX
    return f"{assessment_name}{suffix}"


def get_retrieval_llm_rationale_col_name(assessment_name: str, is_per_chunk: bool = True) -> str:
    """Returns the column name for the LLM judged retrieval metric rationale"""
    suffix = _RATIONALES_SUFFIX if is_per_chunk else _RATIONALE_SUFFIX
    return f"{assessment_name}{suffix}"


def get_retrieval_llm_error_message_col_name(assessment_name: str, is_per_chunk: bool = True) -> str:
    """Returns the column name for the LLM judged retrieval metric error message"""
    suffix = _ERROR_MESSAGES_SUFFIX if is_per_chunk else _ERROR_MESSAGE_SUFFIX
    return f"{assessment_name}{suffix}"


def get_retrieval_llm_metric_col_name(metric_name: str) -> str:
    """Returns the column name for the LLM judged retrieval metric"""
    return f"{metric_name}"
