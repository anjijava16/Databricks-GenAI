import dataclasses
import functools
import inspect
import traceback
from typing import Any, Callable, Collection, Dict, List, Literal, Optional, Union

import mlflow.metrics
from mlflow import entities as mlflow_entities
from mlflow import evaluation as mlflow_eval
from mlflow.entities import assessment as mlflow_assessment

from databricks.rag_eval import callable_builtin_judges, schemas
from databricks.rag_eval.evaluation import entities, metrics
from databricks.rag_eval.utils import aggregation_utils, error_utils, input_output_utils


def _make_code_type_assessment_source(metric_name) -> mlflow_entities.AssessmentSource:
    return mlflow_entities.AssessmentSource(
        source_type=mlflow_entities.AssessmentSourceType.CODE,
        source_id=metric_name,
    )


def _get_custom_assessment_name(assessment: mlflow_entities.Assessment, metric_name: str) -> str:
    """Get the name of the custom assessment. Use assessment name if present and not a builtin judge
    name, otherwise use the metric name.

    Args:
        assessment (mlflow_entities.Assessment): The assessment to get the name for.
        metric_name (str): The name of the metric.
    """
    # If the user didn't provide a name, use the metric name
    if assessment.name == mlflow_assessment.DEFAULT_FEEDBACK_NAME:
        return metric_name
    # If the assessment is from a callable builtin judge, use the metric name
    elif (
        assessment.metadata is not None
        and assessment.metadata.get(callable_builtin_judges.USER_DEFINED_ASSESSMENT_NAME_KEY) == "false"
    ):
        return metric_name
    return assessment.name


def _get_full_args_for_custom_metric(eval_item: entities.EvalItem) -> Dict[str, Any]:
    """Get the all available arguments for the custom metrics."""
    return {
        schemas.REQUEST_ID_COL: eval_item.question_id,
        # Here we wrap the raw request in a ChatCompletionRequest object if it's a plain string to be consistent
        # because we have the same wrapping logic when invoking the model.
        # In the long term future, we want to remove this wrapping logic and pass the raw request as is.
        schemas.REQUEST_COL: input_output_utils.to_chat_completion_request(eval_item.raw_request),
        schemas.RESPONSE_COL: eval_item.raw_response,
        schemas.RETRIEVED_CONTEXT_COL: (
            eval_item.retrieval_context.to_output_dict() if eval_item.retrieval_context else None
        ),
        schemas.EXPECTED_RESPONSE_COL: eval_item.ground_truth_answer,
        schemas.EXPECTED_FACTS_COL: eval_item.expected_facts,
        schemas.GUIDELINES_COL: eval_item.raw_guidelines,
        schemas.EXPECTED_RETRIEVED_CONTEXT_COL: (
            eval_item.ground_truth_retrieval_context.to_output_dict()
            if eval_item.ground_truth_retrieval_context
            else None
        ),
        schemas.CUSTOM_EXPECTED_COL: eval_item.custom_expected,
        schemas.CUSTOM_INPUTS_COL: eval_item.custom_inputs,
        schemas.CUSTOM_OUTPUTS_COL: eval_item.custom_outputs,
        schemas.TRACE_COL: eval_item.trace,
        schemas.TOOL_CALLS_COL: eval_item.tool_calls,
    }


def _convert_custom_metric_value(metric_name: str, metric_value: Any) -> List[mlflow_entities.Assessment]:
    """
    Convert the custom metric value to a list of MLflow AssessmentV3 objects.
    Raise an error if the value is not valid.

    Supported metric values:
        - number
        - boolean
        - string
        - AssessmentV2 object
        - List[AssessmentV2]


    If you have a number, boolean, or string:
    @metric
    def custom_metric(request_id, request, response):
        return 0.5

    The assessment will be normalized to:
        mlflow_entities.Assessment(  # This is AssessmentV3
            name="custom_metric",
            source=assessment_source.AssessmentSource(
                source_type=assessment_source.AssessmentSourceType.CODE,
                source_id="custom_metric",
            ),
            feedback=FeedbackValue(value=0.5),
        )

    If you have an assessment or list of assessments:
    @metric
    def custom_metric(request_id, request, response):
        return mlflow.entities.Feedback(  # This is AssessmentV2
            name="custom_assessment",
            value=0.5,
        )

    The assessment will be normalized to:
        mlflow_entities.Assessment(  # This is AssessmentV3
            name="custom_custom_assessment",
            value=0.5,
            source=mlflow.entities.AssessmentSource(
                source_type=mlflow.entities.AssessmentSourceType.CODE,
                source_id="custom_metric",
            ),
        )
    """
    # None is a valid metric value, return an empty list
    if metric_value is None:
        return []

    # Primitives are valid metric values
    if isinstance(metric_value, (int, float, bool, str)):
        return [
            mlflow_entities.Feedback(
                name=metric_name,
                source=_make_code_type_assessment_source(metric_name),
                value=metric_value,
            )
        ]

    if isinstance(metric_value, mlflow_eval.Assessment):
        raise error_utils.ValidationError(
            f"Got unsupported Assessment object from custom metric '{metric_name}'. "
            f"Please use the new `mlflow.entities.Assessment` object instead."
        )

    if isinstance(metric_value, mlflow_entities.Assessment):
        metric_value.name = _get_custom_assessment_name(metric_value, metric_name)
        return [metric_value]

    if isinstance(metric_value, Collection):
        assessments = []
        for item in metric_value:
            if isinstance(item, mlflow_entities.Assessment):
                item.name = _get_custom_assessment_name(item, metric_name)
                assessments.append(item)
            else:
                raise error_utils.ValidationError(
                    f"Got unsupported result from custom metric '{metric_name}'. "
                    f"Expected the metric value to be a number, or a boolean, or a string, or an Assessment, or a list of Assessments. "
                    f"Got {type(item)} in the list. Full list: {metric_value}.",
                )
        return assessments

    raise error_utils.ValidationError(
        f"Got unsupported result from custom metric '{metric_name}'. "
        f"Expected the metric value to be a number, or a boolean, or a string, or an Assessment, or a list of Assessments. "
        f"Got {metric_value}.",
    )


@dataclasses.dataclass
class CustomMetric(metrics.Metric):
    """
    A custom metric that runs a user-defined evaluation function.

    :param name: The name of the metric.
    :param eval_fn: A user-defined function that computes the metric value.
    """

    name: str
    eval_fn: Callable[..., Any]
    aggregations: Optional[List[Union[str, Callable]]] = None

    def run(
        self,
        *,
        eval_item: Optional[entities.EvalItem] = None,
        assessment_results: Optional[List[entities.AssessmentResult]] = None,
    ) -> List[entities.MetricResult]:
        if eval_item is None:
            return []

        kwargs = self._get_kwargs(eval_item)
        try:
            # noinspection PyCallingNonCallable
            metric_value = self.eval_fn(**kwargs)
        except Exception as e:
            error_assessment = mlflow_entities.Feedback(
                name=self.name,
                source=_make_code_type_assessment_source(self.name),
                error=mlflow_entities.AssessmentError(
                    error_code="CUSTOM_METRIC_ERROR",
                    error_message=str(e),
                    stack_trace=traceback.format_exc(),
                ),
            )
            return [
                entities.MetricResult(
                    metric_value=error_assessment,
                )
            ]

        assessments = _convert_custom_metric_value(self.name, metric_value)
        return [
            entities.MetricResult(
                metric_value=assessment,
            )
            for assessment in assessments
        ]

    def __call__(self, *args, **kwargs):
        return self.eval_fn(*args, **kwargs)

    def _get_kwargs(self, eval_item: entities.EvalItem) -> Dict[str, Any]:
        # noinspection PyTypeChecker
        arg_spec = inspect.getfullargspec(self.eval_fn)

        full_args = _get_full_args_for_custom_metric(eval_item)
        # If the metric accepts **kwargs, pass all available arguments
        if arg_spec.varkw:
            return full_args
        kwonlydefaults = arg_spec.kwonlydefaults or {}
        required_args = arg_spec.args + [arg for arg in arg_spec.kwonlyargs if arg not in kwonlydefaults]
        optional_args = list(kwonlydefaults.keys())
        accepted_args = required_args + optional_args
        # Validate that the dataframe can cover all the required arguments
        missing_args = set(required_args) - full_args.keys()
        if missing_args:
            raise TypeError(f"Dataframe is missing arguments {missing_args} to metric {self.name}")
        # Filter the dataframe down to arguments that the metric accepts
        return {k: v for k, v in full_args.items() if k in accepted_args}


def metric(
    eval_fn=None,
    *,
    name: Optional[str] = None,
    aggregations: Optional[
        List[
            Union[
                Literal["min", "max", "mean", "median", "variance", "p90", "p99"],
                Callable,
            ]
        ]
    ] = None,
):
    # noinspection PySingleQuotedDocstring
    '''
    .. deprecated:: 1.6.0
        This function is deprecated and will be removed in a future release.
        Use the new `scorer API <https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/custom-judge/>`_ instead.

    Create a custom agent metric from a user-defined eval function.

    Can be used as a decorator on the eval_fn.

    The eval_fn should have the following signature:

        .. code-block:: python

            def eval_fn(
                *,
                request_id: str,
                request: Union[ChatCompletionRequest, str],
                response: Optional[Any],
                retrieved_context: Optional[List[Dict[str, str]]]
                expected_response: Optional[Any],
                expected_facts: Optional[List[str]],
                guidelines: Optional[Union[List[str], Dict[str, List[str]]]],
                expected_retrieved_context: Optional[List[Dict[str, str]]],
                custom_expected: Optional[Dict[str, Any]],
                custom_inputs: Optional[Dict[str, Any]],
                custom_outputs: Optional[Dict[str, Any]],
                trace: Optional[mlflow.entities.Trace],
                tool_calls: Optional[List[ToolCallInvocation]],
                **kwargs,
            ) -> Optional[Union[int, float, bool]]:
                """
                Args:
                    request_id: The ID of the request.
                    request: The agent's input from your input eval dataset.
                    response: The agent's raw output. Whatever we get from the agent, we will pass it here as is.
                    retrieved_context: Retrieved context, can be from your input eval dataset or from the trace,
                                       we will try to extract retrieval context from the trace;
                                       if you have custom extraction logic, use the `trace` field.
                    expected_response: The expected response from your input eval dataset.
                    expected_facts: The expected facts from your input eval dataset.
                    guidelines: The guidelines from your input eval dataset.
                    expected_retrieved_context: The expected retrieved context from your input eval dataset.
                    custom_expected: Custom expected information from your input eval dataset.
                    custom_inputs: Custom inputs from your input eval dataset.
                    custom_outputs: Custom outputs from the agent's response.
                    trace: The trace object. You can use this to extract additional information from the trace.
                    tool_calls: List of tool call invocations, can be from your agent's response (ChatAgent only)
                                or from the trace. We will prioritize extracting from the trace as it contains
                                additional information such as available tools and from which span the tool was called.
                """

    eval_fn will always be called with named arguments. You only need to declare the arguments you need.
    If kwargs is declared, all available arguments will be passed.

    The return value of the function should be either a number or a boolean. It will be used as the metric value.
    Return None if the metric cannot be computed.

    :param eval_fn: The user-defined eval function.
    :param name: The name of the metric. If not provided, the function name will be used.
    :param aggregations: The aggregations to apply to the metric.
    '''

    def decorator(fn, *, _name=name, _aggregations=aggregations):
        # Use mlflow.metrics.make_metric to validate the metric name
        mlflow.metrics.make_metric(eval_fn=fn, greater_is_better=True, name=_name)
        metric_name = _name or fn.__name__

        # Validate signature of the fn
        arg_spec = inspect.getfullargspec(fn)
        if arg_spec.varargs:
            raise error_utils.ValidationError(
                "The eval_fn should not accept *args.",
            )

        supported_aggregations = list(aggregation_utils._AGGREGATION_TO_AGGREGATE_FUNCTION.keys())
        for aggregation in _aggregations or []:
            if isinstance(aggregation, str) and aggregation not in supported_aggregations:
                raise error_utils.ValidationError(
                    f"Invalid aggregation: {aggregation}. Supported aggregations: {supported_aggregations}.",
                )
        return functools.wraps(fn)(CustomMetric(name=metric_name, eval_fn=fn, aggregations=_aggregations))

    if eval_fn is not None:
        return decorator(eval_fn)

    return decorator


def _analyze_indentation(lines: List[str]) -> (str, int):
    """
    Determine the indentation used in a function body, and the unit.
    If no indentation is used, default to 2 spaces.
    """
    default = (" ", 2)
    # Remove any empty lines
    non_empty_lines = [line for line in lines if line.strip()]

    if not non_empty_lines:
        return default

    # Check the first line that has any indentation
    for line in non_empty_lines:
        # Get the indentation of the current line
        indent_length = len(line) - len(line.lstrip())
        indent_str = line[:indent_length]

        # If this line has indentation
        if indent_str:
            # Determine indentation unit (tabs or spaces)
            indent_unit = "\t" if "\t" in indent_str else " "
            indent_level = indent_str.count(indent_unit)

            return indent_unit, indent_level

    # No indentation found in any line
    return default


def _create_metric_from_function_body(metric_name: str, function_body: str) -> CustomMetric:
    @metric(name=metric_name)
    def wrapped_fn(**kwargs):
        # Build the function definition string
        params = [
            "*",
            schemas.REQUEST_ID_COL,
            schemas.REQUEST_COL,
            schemas.RESPONSE_COL,
            schemas.RETRIEVED_CONTEXT_COL,
            schemas.EXPECTED_RESPONSE_COL,
            schemas.EXPECTED_FACTS_COL,
            schemas.GUIDELINES_COL,
            schemas.EXPECTED_RETRIEVED_CONTEXT_COL,
            schemas.CUSTOM_EXPECTED_COL,
            schemas.CUSTOM_INPUTS_COL,
            schemas.CUSTOM_OUTPUTS_COL,
            schemas.TRACE_COL,
            schemas.TOOL_CALLS_COL,
            "kwargs",
        ]
        params_str = ", ".join(params)
        # Use actual function name for clearer error handling
        function_def = f"def {metric_name}({params_str}):\n"

        # Indent each line of the function body, assuming 0 indentation at the function body root level
        func_lines = function_body.split("\n")
        indent_unit, indent_length = _analyze_indentation(func_lines)
        indented_body = "\n".join(f"{indent_unit*indent_length}{line}" for line in func_lines)

        # Combine the definition and body
        full_function = function_def + indented_body

        # Create a local namespace
        local_namespace = {}

        # Execute the function definition in the local namespace
        exec(full_function, globals(), local_namespace)

        # Return the newly created function
        fn = local_namespace[metric_name]

        cm_kwargs = {
            schemas.REQUEST_ID_COL: kwargs.get(schemas.REQUEST_ID_COL),
            schemas.REQUEST_COL: kwargs.get(schemas.REQUEST_COL),
            schemas.RESPONSE_COL: kwargs.get(schemas.RESPONSE_COL),
            schemas.RETRIEVED_CONTEXT_COL: kwargs.get(schemas.RETRIEVED_CONTEXT_COL),
            schemas.EXPECTED_RESPONSE_COL: kwargs.get(schemas.EXPECTED_RESPONSE_COL),
            schemas.EXPECTED_FACTS_COL: kwargs.get(schemas.EXPECTED_FACTS_COL),
            schemas.GUIDELINES_COL: kwargs.get(schemas.GUIDELINES_COL),
            schemas.EXPECTED_RETRIEVED_CONTEXT_COL: kwargs.get(schemas.EXPECTED_RETRIEVED_CONTEXT_COL),
            schemas.CUSTOM_EXPECTED_COL: kwargs.get(schemas.CUSTOM_EXPECTED_COL),
            schemas.CUSTOM_INPUTS_COL: kwargs.get(schemas.CUSTOM_INPUTS_COL),
            schemas.CUSTOM_OUTPUTS_COL: kwargs.get(schemas.CUSTOM_OUTPUTS_COL),
            schemas.TRACE_COL: kwargs.get(schemas.TRACE_COL),
            schemas.TOOL_CALLS_COL: kwargs.get(schemas.TOOL_CALLS_COL),
        }
        cm_kwargs["kwargs"] = cm_kwargs

        return fn(**cm_kwargs)

    return wrapped_fn
