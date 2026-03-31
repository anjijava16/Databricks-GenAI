"""Helper functions for dealing with ratings."""

import re
from typing import Optional

MISSING_INPUTS_ERROR_CODE = 1001
INVALID_INPUT_ERROR_CODE = 1006
CONFLICTING_INPUTS_ERROR_CODE = 1010


def _extract_error_code(error_message: Optional[str]) -> Optional[str]:
    """
    Extract the error code from the error message.
    """
    if error_message is None:
        return None
    match = re.match(r"Error\[(\d+)]", error_message)
    return match.group(1) if match else None


def is_missing_input_error(error_message: Optional[str]) -> bool:
    """
    Check if the error message is due to missing input fields (Error[1001]).
    """
    return _extract_error_code(error_message) == str(MISSING_INPUTS_ERROR_CODE)


def has_conflicting_input_error(error_message: Optional[str]) -> bool:
    """
    Check if the error message is due to conflicting input fields (Error[1010]).
    """
    return _extract_error_code(error_message) == str(CONFLICTING_INPUTS_ERROR_CODE)


def normalize_error_message(error_message: str) -> str:
    """
    Normalize the error message by removing the Reference ID part.

    The Reference ID is a UUID generated for every Armeria service call. We assume any string
    after "Reference ID: " with the character set described in the regex below is a Reference ID.
    """
    return re.sub(r", Reference ID: [\w.,!?;:-]+", "", error_message)
