"""Utilities to validate and manipulate model inputs and outputs."""

import json
from typing import Any, Dict, List, NewType, Optional, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel

from databricks.rag_eval import env_vars
from databricks.rag_eval.utils import collection_utils, error_utils

ModelInput = NewType("ModelInput", Union[Dict[str, Any], str])
ModelOutput = NewType("ModelOutput", Optional[Union[Dict[str, Any], str, List[Dict[str, Any]], List[str]]])

# ChatCompletionRequest fields
_MESSAGES = "messages"
_ROLE = "role"
_CONTENT = "content"
_USER_ROLE = "user"
_CHOICES = "choices"
_MESSAGE = "message"
# SplitChatMessagesRequest fields
_QUERY = "query"
_HISTORY = "history"

_RETURN_TRACE_FLAG = {
    "databricks_options": {
        "return_trace": True,
    }
}


def to_dict(obj: Any) -> Dict:
    if hasattr(obj, "to_dict"):
        return obj.to_dict()

    if isinstance(obj, BaseModel):
        return obj.model_dump()

    # First convert to JSON string with handling for nested objects
    json_str = json.dumps(obj, default=lambda o: o.__dict__)
    # Then convert back to dictionary
    return json.loads(json_str)


def parse_variant_data(data):
    """
    Helper to convert a VariantVal to a Python dictionary. If the data is not a VariantVal,
    it will be returned as is.
    """
    try:
        from pyspark.sql import types as T

        if isinstance(data, T.VariantVal):
            return data.toPython()
    except (AttributeError, ImportError):
        # `pyspark.sql.types.VariantVal` may not be available in all environments, so we catch
        # any exception related to the import of this type.
        pass
    return data


def to_chat_completion_request(data: ModelInput) -> Dict[str, Any]:
    """Converts a model input to a ChatCompletionRequest. The input can be a string or a dict."""
    if isinstance(data, str):
        # For backward compatibility, we convert input strings into ChatCompletionRequests
        # before invoking the model.
        return {
            _MESSAGES: [
                {
                    _ROLE: _USER_ROLE,
                    _CONTENT: data,
                },
            ],
        }
    else:
        return data


def set_include_trace(model_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Set the flag to include trace in the model input.

    :param model_input: The model input
    :return: The model input with the flag set
    """
    return collection_utils.deep_update(model_input, _RETURN_TRACE_FLAG)


def to_chat_completion_response(data: Optional[str]) -> Optional[ModelOutput]:
    """Converts a model output to a ChatCompletionResponse."""
    if data is None:
        return None
    return {
        _CHOICES: [
            {
                _MESSAGE: {
                    _CONTENT: data,
                },
            },
        ],
    }


def request_to_string(data: ModelInput) -> str:
    """
    Extract the user question/query from a model input.

    The following input formats are processed as follows:
    1. str: The user question/query
    2. Dictionary representations of ChatCompletionRequest, ChatModel, or ChatAgent:
        - The last message in the conversation if only one message
        - The serialized message array if multiple valid messages
    3. Dictionary representations of SplitChatMessagesRequest: The `query` field
    4. Anything else is stringified

    This method performs the minimal validations required to extract the input string.
    """
    if is_none_or_nan(data):
        return " "
    if isinstance(data, str):
        return data
    data = parse_variant_data(data)
    data = to_dict(data)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a dictionary, got {type(data)}")
    # ChatCompletionRequest, ChatModel, or ChatAgent input
    if data.get(_MESSAGES) and len(data[_MESSAGES]) > 0:
        messages = data.get(_MESSAGES)
        contents = list(map(lambda message: message.get(_CONTENT), messages))

        # If multiple messages and they all have valid content, serialize the messages array
        if (
            env_vars.AGENT_EVAL_ENABLE_MULTI_TURN_EVALUATION.get()
            and len(contents) > 1
            and all(isinstance(content, str) for content in contents)
        ):
            return json.dumps(messages)
        elif isinstance(contents[-1], str):
            return contents[-1]

    # SplitChatMessagesRequest input
    if _QUERY in data:
        content = data[_QUERY]
        return content if isinstance(content, str) else json.dumps(content, default=lambda o: o.__dict__)

    # for all other input that do not fall into the above categories, stringify the input
    return str(data)


def is_valid_input(data: ModelInput) -> bool:
    """Checks whether an input is considered valid for the purposes of evaluation.

    Valid input formats are described in the docstring for `request_to_string`.
    """
    try:
        return request_to_string(data) is not None
    except ValueError:
        return False


def is_none_or_nan(value: Any) -> bool:
    """Checks whether a value is None or NaN."""
    # isinstance(value, float) check is needed to ensure that pd.isna is not called on an array.
    return value is None or (isinstance(value, float) and pd.isna(value))


def response_to_string(data: ModelOutput) -> Optional[str]:
    """Converts a model output to a string. The following output formats are accepted:
    1. str
    2. Dictionary representations of ChatCompletionResponse or ChatModel
    3. Dictionary representations of ChatAgent - the last message in the conversation
    4. Dictionary representations of StringResponse

    If None is passed in, None is returned.

    This method performs the minimal validations required to extract the output string.
    """
    if is_none_or_nan(data):
        return None
    if isinstance(data, str):
        return data
    if isinstance(data, list) and len(data) > 0:
        # PyFuncModel.predict may wrap the output in a list
        return response_to_string(data[0])
    data = parse_variant_data(data)
    data = to_dict(data)
    if not isinstance(data, Dict):
        raise ValueError(f"Expected a dictionary, got {type(data)}")

    # catch-all to stringify the output
    content = str(data)
    # ChatCompletionResponse or ChatModel output
    if (
        data.get(_CHOICES)
        and len(data[_CHOICES]) > 0
        and isinstance(data[_CHOICES][0], dict)
        and data[_CHOICES][0].get(_MESSAGE) is not None
        and data[_CHOICES][0][_MESSAGE].get(_CONTENT) is not None
    ):
        content = data[_CHOICES][0][_MESSAGE][_CONTENT]
    # ChatAgent output - use the last message in the conversation
    if (
        data.get(_MESSAGES)
        and len(data[_MESSAGES]) > 0
        and isinstance(data[_MESSAGES][-1], dict)
        and data[_MESSAGES][-1].get(_CONTENT) is not None
    ):
        content = data[_MESSAGES][-1][_CONTENT]
    # StringResponse output
    if _CONTENT in data:
        content = data[_CONTENT]

    return content if isinstance(content, str) else json.dumps(content, default=lambda o: o.__dict__)


def normalize_to_dictionary(data: Any) -> Dict[str, Any]:
    """Normalizes a data structure to a dictionary."""
    if is_none_or_nan(data):
        return {}
    elif isinstance(data, str):
        try:
            return json.loads(data)
        except:  # noqa: E722
            pass
    elif isinstance(data, dict):
        return data

    raise ValueError(f"Expected a dictionary or serialized JSON string, got {type(data)}")


def is_valid_output(data: ModelOutput) -> bool:
    """Checks whether an output is considered valid for the purposes of evaluation.

    Valid output formats are described in the docstring for `response_to_string`.
    """
    try:
        response_to_string(data)
        return True
    except ValueError:
        return False


def is_valid_guidelines_iterable(guidelines: Any) -> bool:
    """Checks whether a guidelines object is a valid list of guidelines."""
    return isinstance(guidelines, (list, np.ndarray)) and all(isinstance(guideline, str) for guideline in guidelines)


def check_guidelines_iterable_exceeds_limit(guidelines: List[str]):
    """Checks whether a list of guidelines exceeds the limits for guidelines"""
    if len(guidelines) > env_vars.AGENT_EVAL_MAX_NUM_GUIDELINES.get():
        raise error_utils.ValidationError(
            f"The number of guidelines exceeds the maximum: "
            f"{env_vars.AGENT_EVAL_MAX_NUM_GUIDELINES.get()}. Got {len(guidelines)} guidelines."
            + error_utils.CONTACT_FOR_LIMIT_ERROR_SUFFIX
        )


def is_valid_guidelines_mapping(guidelines: Any) -> bool:
    """Checks whether a guidelines object is a valid mapping of named guidelines."""
    return isinstance(guidelines, dict) and all(
        isinstance(guidelines_group, list) and all(isinstance(guideline, str) for guideline in guidelines_group)
        for guidelines_group in guidelines.values()
    )


def check_guidelines_mapping_exceeds_limit(guidelines: List[str]):
    """Checks whether a mapping of named guidelines exceeds the limits for guidelines"""
    for guidelines_name, grouped_guidelines in guidelines.items():
        if len(grouped_guidelines) > env_vars.AGENT_EVAL_MAX_NUM_GUIDELINES.get():
            raise error_utils.ValidationError(
                f"The number of guidelines for `{guidelines_name}` exceeds the maximum: "
                f"{env_vars.AGENT_EVAL_MAX_NUM_GUIDELINES.get()}. Got {len(grouped_guidelines)}. "
                + error_utils.CONTACT_FOR_LIMIT_ERROR_SUFFIX
            )
