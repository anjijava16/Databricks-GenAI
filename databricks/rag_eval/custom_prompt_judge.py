import json
import logging
import re
import traceback
from difflib import unified_diff
from typing import Callable

from mlflow import entities as mlflow_entities

from databricks.rag_eval import context

_logger = logging.getLogger(__name__)

CUSTOM_PROMPT_JUDGE_USE_CASE = "custom_prompt_judge"


def render_prompt(template: str, **kwargs) -> str:
    """
    Render a prompt template with the given kwargs.

    Args:
        template: Template string with {{variable}} placeholders
        **kwargs: Variable values to substitute

    Returns:
        The rendered prompt string with the given kwargs substituted.

    Raises:
        KeyError: If a template variable is not provided in kwargs
    """

    def replacer(match):
        key = match.group(1)
        if key not in kwargs:
            raise KeyError(f"Template variable '{key}' not found in provided arguments")
        return str(kwargs[key])

    return re.sub(r"\{\{(\w+)\}\}", replacer, template)


def parse_llm_judge_output(output: str) -> tuple[str, str]:
    """
    Parse the LLM judge output for the decision and rationale.

    Args:
        output(str): The LLM judge output.

    Returns:
        The LLM judge decision and rationale.
    """
    try:
        parsed_response = json.loads(output)
    except json.JSONDecodeError:
        raise ValueError(f"Custom prompt judge response is not valid JSON: {output}")

    rationale = parsed_response.get("rationale")
    result = parsed_response.get("result")

    if not result:
        raise ValueError(f"Custom prompt judge response does not contain a result: {output}")
    if not rationale:
        rationale = "Rationale not provided by the custom prompt judge."

    return remove_choice_brackets(result), rationale


def _postprocess_user_prompt(prompt: str) -> str:
    suffix = """
    Answer ONLY in JSON and NOT in markdown, following the format:

    {
        "rationale": "Reason for the decision. Start each rationale with `Let's think step by step`."
        "result": "The category chosen."
    }
    """
    return f"{prompt.strip()}\n\n{suffix}"


def extract_choices(prompt: str) -> list[str]:
    """
    Extract choices denoted with [[CHOICE_NAME]] from a prompt.

    Args:
        prompt (str): The prompt text containing choices

    Returns:
        list[str]: A list of extracted choice names, or an empty list if none are found

    Example:
        >>> prompt = '''
        ... [[formal]]: The response is very formal.
        ... [[semi_formal]]: The response is somewhat formal.
        ... [[not_formal]]: The response is not formal.
        ... '''
        >>> extract_choices(prompt)
        ['formal', 'semi_formal', 'not_formal']
    """
    pattern = r"\[\[([\w ]+)\]\]"
    matches = re.findall(pattern, prompt)

    return matches


def remove_choice_brackets(text: str) -> str:
    """
    Remove double square brackets around choices, converting [[CHOICE_NAME]] to CHOICE_NAME.

    Args:
        text (str): Input text containing choices in [[CHOICE_NAME]] format

    Returns:
        str: Text with double square brackets removed from choices

    Example:
        >>> remove_choice_brackets("Choose from [[formal]], [[informal]], or [[neutral]]")
        "Choose from formal, informal, or neutral"
    """
    # Pattern to match text between double square brackets and replace with just the content
    return re.sub(r"\[\[([\w ]+)\]\]", r"\1", text)


def _validate_llm_value(*, value: str, choices: list[str], numeric_values: dict[str, int | float] | None) -> None:
    """
    Validate that the LLM value is one of the provided choices.

    Args:
        value (str): The value to validate.
        choices (list[str]): The list of valid choices.
        numeric_values (dict[str, int | float] | None): Optional mapping from categorical values to numeric scores.

    Raises:
        ValueError: If the value is not in the choices.
    """
    if value not in choices:
        raise ValueError(f"Value '{value}' is not one of the valid choices: {choices}")

    if numeric_values and value not in numeric_values:
        raise ValueError(f"Custom prompt judge output '{value}' not found in numeric values mapping.")


@context.eval_context
def custom_prompt_judge(
    *,
    name: str,
    prompt_template: str,
    numeric_values: dict[str, int | float] | None = None,
) -> Callable[..., mlflow_entities.Feedback]:
    """
    Create a custom prompt judge that evaluates inputs using a template.

    Example prompt template:

    ```
    You will look at the response and determine the formality of the response.

    <request>{{request}}</request>
    <response>{{response}}</response>

    You must choose one of the following categories.

    [[formal]]: The response is very formal.
    [[semi_formal]]: The response is somewhat formal. The response is somewhat formal if the response mentions friendship, etc.
    [[not_formal]]: The response is not formal.
    ```

    Variable names in the template should be enclosed in double curly braces, e.g., `{{request}}`, `{{response}}`.
    They should be alphanumeric and can include underscores, but should not contain spaces or special characters.

    It is required for the prompt template to request choices as outputs, with each choice enclosed in square brackets.
    Choice names should be alphanumeric and can include underscores, but should not contain spaces or special characters.

    Args:
        name (str): Name of the judge, used as the assessment name.
        prompt_template (str): Template string with {{var_name}} placeholders for variable substitution. Should be prompted with choices as outputs.
        numeric_values (dict[str, int | float] | None): Optional mapping from categorical values to numeric scores.
            Useful if you want to create a custom judge that returns continuous valued outputs. Defaults to None.

    Returns:
        A callable that takes keyword arguments mapping to the template variables and returns an mlflow Feedback.
    """
    managed_rag_client = context.get_context().build_managed_rag_client()

    choices = extract_choices(prompt_template)
    _logger.debug(f"Extracted choices from prompt template: {choices}")

    # Validate that choices are provided in the prompt template.
    if not choices:
        raise ValueError(
            "Prompt template must include choices denoted with [[CHOICE_NAME]]. "
            "No choices found in the provided prompt template."
        )

    # Validate that choices match numeric_values keys if provided.
    if numeric_values is not None:
        numeric_values_keys = sorted(list(numeric_values.keys()))
        sorted_choices = sorted(choices)
        if numeric_values_keys != sorted_choices:
            diff = "\n".join(
                unified_diff(
                    numeric_values_keys,
                    sorted_choices,
                    fromfile="numeric_values_keys",
                    tofile="choices",
                )
            )
            raise ValueError(
                f"numeric_values keys must match the choices included in the prompt template.\n"
                f"numeric_values keys: {numeric_values_keys}\n"
                f"choices in prompt: {sorted_choices}\n"
                f"Diff:\n{diff}"
            )

        # Validate that numeric_values values are numeric if provided.
        if not all(isinstance(value, (int, float)) for value in numeric_values.values()):
            raise ValueError("All values in numeric_values must be numeric (int or float).")

    def get_assessment_source() -> mlflow_entities.AssessmentSource:
        return mlflow_entities.AssessmentSource(
            source_type=mlflow_entities.AssessmentSourceType.LLM_JUDGE,
            source_id=f"custom_prompt_judge_{name}",
        )

    def judge(**kwargs) -> mlflow_entities.Feedback:
        """
        Execute the custom judge with the provided arguments.

        Args:
            **kwargs: Arguments to substitute in the template (e.g., request, response)

        Returns:
            Feedback object with the evaluation result
        """

        try:
            # Render prompt template with the given kwargs
            rendered_prompt = render_prompt(prompt_template, **kwargs)
            enhanced_prompt = _postprocess_user_prompt(rendered_prompt)

            _logger.debug(f"Final prompt for custom prompt judge: {enhanced_prompt}")

            # Get experiment ID from context
            experiment_id = context.get_context().get_mlflow_experiment_id()

            # Call the LLM judge.
            llm_res = managed_rag_client.get_chat_completions_result(
                enhanced_prompt, None, experiment_id=experiment_id, use_case=CUSTOM_PROMPT_JUDGE_USE_CASE
            )
            _logger.debug(f"Custom prompt judge response: {llm_res}")

            final_value = None
            rationale = None
            metadata = None
            error = None
            if llm_res.output is not None:
                string_value, rationale = parse_llm_judge_output(llm_res.output)
                _logger.debug(f"Parsed LLM judge string value: {string_value}")
                _logger.debug(f"Parsed LLM judge rationale: {rationale}")

                _validate_llm_value(
                    value=string_value,
                    choices=choices,
                    numeric_values=numeric_values,
                )

                # Map to numeric value if mapping is provided
                final_value = string_value
                if numeric_values:
                    final_value = numeric_values[string_value]
                    metadata = {"string_value": string_value}
            else:
                error = mlflow_entities.AssessmentError(
                    error_code=llm_res.error_code,
                    error_message=llm_res.error_message,
                )

            # Create and return the assessment
            return mlflow_entities.Feedback(
                name=name,
                source=get_assessment_source(),
                rationale=rationale,
                metadata=metadata,
                value=final_value,
                error=error,
            )
        except Exception as e:
            return mlflow_entities.Feedback(
                name=name,
                source=get_assessment_source(),
                value=None,
                error=mlflow_entities.AssessmentError(
                    error_code=type(e).__name__,
                    error_message=str(e),
                    stack_trace=traceback.format_exc(),
                ),
            )

    return judge
