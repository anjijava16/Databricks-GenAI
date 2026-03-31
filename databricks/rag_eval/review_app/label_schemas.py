"""Label schemas for configuring the Review App."""

from databricks.rag_eval.utils.enum_utils import StrEnum

from .entities import (
    InputCategorical,
    InputCategoricalList,
    InputNumeric,
    InputText,
    InputTextList,
    LabelSchema,
)

EXPECTED_FACTS = "expected_facts"
GUIDELINES = "guidelines"
EXPECTED_RESPONSE = "expected_response"


class LabelSchemaType(StrEnum):
    """Type of label schema."""

    FEEDBACK = "feedback"
    EXPECTATION = "expectation"


__all__ = [
    "EXPECTED_FACTS",
    "GUIDELINES",
    "EXPECTED_RESPONSE",
    "LabelSchemaType",
    "LabelSchema",
    "InputCategorical",
    "InputCategoricalList",
    "InputNumeric",
    "InputText",
    "InputTextList",
]
