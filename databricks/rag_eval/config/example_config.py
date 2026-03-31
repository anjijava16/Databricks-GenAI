import dataclasses
from typing import Optional

from databricks.rag_eval.utils import enum_utils, error_utils


class ExampleRating(enum_utils.StrEnum):
    """
    A rating in an assessment example.
    These are the only valid ratings for the `rating` field in an `AssessmentExample`.
    """

    YES = "yes"
    NO = "no"

    @classmethod
    def _missing_(cls, value: str):
        for member in cls:
            if member == value.strip().lower():
                return member
        raise error_utils.ValidationError(f"Invalid rating for example: '{value}'. Must be one of {cls.values()}.")

    def to_bool(self) -> bool:
        return self == ExampleRating.YES


@dataclasses.dataclass(frozen=True)
class AssessmentExample:
    """
    User-provided example to guide the LLM judge in deciding on the Yes/No value
    for a particular assessment.
    e.g. User provided example yaml:
    ```
    - context: some context
      response: some response
      rating: Yes
      rationale: some rationale

    will be parsed into the following AssessmentExample object:
    ```
    AssessmentExample(
        variables={"context": "some context", "response": "some response"},
        rating=CategoricalRating.YES,
        rationale="some rationale"
    )
    ```
    """

    variables: dict[str, str]
    """
    Mapping from variable name to variable value. Should be validated against corresponding
    assessment config to ensure all required columns are present.
    """

    rating: ExampleRating
    """
    Whether the output should be considered satisfactory or unsatisfactory for the assessment.
    """

    rationale: Optional[str] = None
    """
    Explanation of why the output was given its value.
    """
