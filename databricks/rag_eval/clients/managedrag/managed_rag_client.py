import functools
from typing import Dict, List, Optional

import requests
import tenacity
from mlflow.types.llm import ToolDefinition
from requests import HTTPError
from urllib3.util import retry

from databricks import version
from databricks.rag_eval import context, env_vars, session
from databricks.rag_eval.clients import databricks_api_client
from databricks.rag_eval.clients.managedrag import proto_serde
from databricks.rag_eval.config import assessment_config
from databricks.rag_eval.evaluation import (
    custom_metrics as agent_custom_metrics,
)
from databricks.rag_eval.evaluation import (
    entities,
    per_run_metrics,
)
from databricks.rag_eval.utils import request_utils

SESSION_ID_HEADER = "eval-session-id"
BATCH_SIZE_HEADER = "eval-session-batch-size"
CLIENT_VERSION_HEADER = "eval-session-client-version"
CLIENT_NAME_HEADER = "eval-session-client-name"
JOB_ID_HEADER = "eval-session-job-id"
JOB_RUN_ID_HEADER = "eval-session-job-run-id"
MLFLOW_RUN_ID_HEADER = "eval-session-mlflow-run-id"
MONITORING_WHEEL_VERSION_HEADER = "eval-session-monitoring-wheel-version"
CHAT_COMPLETIONS_USE_CASE_HEADER = "chat-completions-use-case"
# List of retryable error codes from the judge service. These errors include both HTTP errors and
# classified judge errors (e.g., 3003 is a rate-limit from LLM proxy)
_CONNECTION_ERROR_CODE = "104"
RATE_LIMIT_ERROR_CODES = [_CONNECTION_ERROR_CODE, "429", "3003", "3004"]
RETRYABLE_JUDGE_ERROR_CODES = [
    "500",
    "502",
    "503",
    "504",
    "2003",
    "3001",
] + RATE_LIMIT_ERROR_CODES


def get_default_retry_config():
    return retry.Retry(
        total=env_vars.AGENT_EVAL_JUDGE_SERVICE_ERROR_RETRIES.get(),
        backoff_factor=env_vars.AGENT_EVAL_JUDGE_SERVICE_BACKOFF_FACTOR.get(),
        status_forcelist=[429, 500, 502, 503, 504],
        backoff_jitter=env_vars.AGENT_EVAL_JUDGE_SERVICE_JITTER.get(),
        allowed_methods=frozenset(["GET", "POST"]),  # by default, it doesn't retry on POST
    )


@functools.lru_cache(maxsize=1)
def _get_immutable_eval_session_headers():
    """
    Construct immutable/long-lived request headers and use an LRU cache to avoid
    expensive spark context calls for every request.
    """
    return {
        CLIENT_VERSION_HEADER: version.VERSION,
        CLIENT_NAME_HEADER: env_vars.RAG_EVAL_EVAL_SESSION_CLIENT_NAME.get(),
        JOB_ID_HEADER: context.get_context().get_job_id(),
        JOB_RUN_ID_HEADER: context.get_context().get_job_run_id(),
    }


def _get_eval_session_headers():
    """Constructs the request headers from the thread-local session."""
    immutable_headers = _get_immutable_eval_session_headers()
    headers = {
        MLFLOW_RUN_ID_HEADER: context.get_context().get_mlflow_run_id(),
        **immutable_headers,
    }
    headers = request_utils.add_traffic_id_header(headers)

    # Pass the internal version of the monitoring wheel if it is set
    current_session = session.current_session()
    if current_session:
        headers[SESSION_ID_HEADER] = current_session.session_id
        if current_session.monitoring_wheel_version:
            headers[MONITORING_WHEEL_VERSION_HEADER] = current_session.monitoring_wheel_version
        if current_session.session_batch_size is not None:
            headers[BATCH_SIZE_HEADER] = str(current_session.session_batch_size)

    return headers


def _result_contains_error_code(
    assessment_result: entities.AssessmentResult,
    error_codes: List[str],
) -> bool:
    """
    Returns True if the given assessment result contains one of the given error codes. For per-chunk
    assessments, at least one rating can contain a given error code to return True.
    """
    if isinstance(assessment_result, entities.PerRequestAssessmentResult):
        return assessment_result.rating.error_code is not None and (assessment_result.rating.error_code in error_codes)
    elif isinstance(assessment_result, entities.PerChunkAssessmentResult):
        return any(
            [
                rating.error_code is not None and (rating.error_code in error_codes)
                for rating in assessment_result.positional_rating.values()
            ]
        )
    return False


def has_any_retryable_judge_errors(
    assessment_results: List[entities.AssessmentResult],
) -> bool:
    """
    Returns True if any of the given assessment results is a retryable judge error.
    """
    return any(
        [
            _result_contains_error_code(assessment_result, RETRYABLE_JUDGE_ERROR_CODES)
            for assessment_result in assessment_results
        ]
    )


def get_num_seconds_to_wait_for_retryable_judge_error(
    state: tenacity.RetryCallState,
) -> float:
    """
    Returns the number of seconds to wait before retrying the judge service. The wait time is
    calculated based on the error code returned by the judge service.
    """
    if state.outcome is None:
        raise RuntimeError("Retry was called without an outcome")

    if _result_contains_error_code(state.outcome.result(), RATE_LIMIT_ERROR_CODES):
        return tenacity.wait_random_exponential(
            exp_base=env_vars.RAG_EVAL_LLM_JUDGE_WAIT_EXP_BASE.get(),
            max=env_vars.RAG_EVAL_LLM_JUDGE_MAX_WAIT.get(),
            min=env_vars.RAG_EVAL_LLM_JUDGE_MIN_WAIT.get(),
        )(state)
    else:
        min_wait = env_vars.RAG_EVAL_LLM_JUDGE_BACKOFF_FACTOR.get()
        max_wait = min_wait + env_vars.RAG_EVAL_LLM_JUDGE_BACKOFF_JITTER.get()
        return tenacity.wait_random(min_wait, max_wait)(state)


class ManagedRagClient(databricks_api_client.DatabricksAPIClient):
    """
    Client to interact with the managed-rag service (/chat-assessments).

    Note: this client reads the session from the current thread and uses it to construct the request headers.
      Make sure to construct this client in the same thread where the session is initialized.
    """

    def __init__(self):
        super().__init__(version="2.0")
        self.proto_serde = proto_serde.ChatAssessmentProtoSerde()

    def get_assessment(
        self,
        eval_item: entities.EvalItem,
        config: assessment_config.BuiltinAssessmentConfig,
        experiment_id: Optional[str] = None,
    ) -> List[entities.AssessmentResult]:
        """
        Retrieves the assessment results from the LLM judge service for the given eval item and requested assessment
        """
        try:
            return self._get_assessment(eval_item, config, experiment_id)
        except tenacity.RetryError as retry_error:
            return retry_error.last_attempt.result()

    @tenacity.retry(
        retry=tenacity.retry_if_result(has_any_retryable_judge_errors),
        wait=get_num_seconds_to_wait_for_retryable_judge_error,
        stop=tenacity.stop_after_attempt(env_vars.RAG_EVAL_LLM_JUDGE_MAX_JUDGE_ERROR_RETRIES.get()),
        reraise=False,
    )
    def _get_assessment(
        self,
        eval_item: entities.EvalItem,
        config: assessment_config.BuiltinAssessmentConfig,
        experiment_id: Optional[str] = None,
    ) -> List[entities.AssessmentResult]:
        assessment_name = config.assessment_name
        request_json = self.proto_serde.construct_assessment_request_json(
            eval_item, assessment_name, experiment_id=experiment_id
        )
        with self.get_default_request_session(headers=_get_eval_session_headers()) as session:
            try:
                resp = session.post(
                    self.get_method_url("/agents/chat-assessments"),
                    json=request_json,
                )
            except requests.exceptions.ConnectionError as e:
                # Handle connection errors (e.g., network issues)
                return self.proto_serde.construct_assessment_error_result(
                    eval_item, config, int(_CONNECTION_ERROR_CODE), e
                )

        # The judge service always attempts to return HTTP 200.
        # Even for rate limit exceeded, context limit exceeded, etc.
        # The response body will contain the error message in the "error" field.
        # The outer retry decorator will check the contents to decide whether to retry.
        if resp.status_code == requests.codes.ok:
            return self.proto_serde.construct_assessment_result(
                resp.json(),
                eval_item,
                config,
            )

        # Sometimes unexpected HTTP errors (500, 503, etc.) are generated at the load balancer level.
        # These are converted to Assessment errors, and the retry decorator will decide whether to retry.
        try:
            resp.raise_for_status()
        except HTTPError as e:
            return self.proto_serde.construct_assessment_error_result(
                eval_item,
                config,
                resp.status_code,
                e,
            )

        return []

    def emit_chat_assessment_usage_event(
        self,
        custom_assessments: List[assessment_config.EvaluationMetricAssessmentConfig],
        num_questions: Optional[int],
    ):
        request_json = self.proto_serde.construct_chat_assessment_usage_event_request_json(
            custom_assessments, num_questions
        )
        # Use default retries. Don't need to use response
        with self.get_default_request_session(headers=_get_eval_session_headers()) as session:
            session.post(
                self.get_method_url("/agents/chat-assessment-usage-events"),
                json=request_json,
            )

    def get_assessment_metric_definitions(
        self, assessment_names: List[str]
    ) -> Dict[str, assessment_config.AssessmentInputRequirementExpression]:
        """Retrieves the metric definitions for the given assessment names."""
        request_json = self.proto_serde.construct_assessment_metric_definition_request_json(assessment_names)

        with self.get_default_request_session(
            get_default_retry_config(), headers=_get_eval_session_headers()
        ) as session:
            resp = session.post(
                self.get_method_url("/agents/chat-assessment-definitions"),
                json=request_json,
            )

        if resp.status_code == requests.codes.ok:
            return self.proto_serde.construct_assessment_input_requirement_expressions(resp.json())
        else:
            try:
                resp.raise_for_status()
            except HTTPError as e:
                raise e

    def get_chat_completions_result(
        self,
        user_prompt: str,
        system_prompt: str | None,
        experiment_id: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        tools: Optional[List[ToolDefinition]] = None,
        use_case: Optional[str] = None,
    ) -> proto_serde.GetChatCompletionsResponse:
        request_json = self.proto_serde.construct_get_chat_completions_request_json(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            experiment_id=experiment_id,
            model=model,
            temperature=temperature,
            tools=tools,
        )
        headers = _get_eval_session_headers()
        if use_case is not None:
            headers[CHAT_COMPLETIONS_USE_CASE_HEADER] = use_case
        with self.get_default_request_session(get_default_retry_config(), headers=headers) as session:
            resp = session.post(
                self.get_method_url("/agents/chat-completions"),
                json=request_json,
            )

        if resp.status_code != requests.codes.ok:
            try:
                resp.raise_for_status()
            except HTTPError as e:
                return proto_serde.GetChatCompletionsResponse(
                    output=None,
                    error_code=str(resp.status_code),
                    error_message=str(e),
                )
        return self.proto_serde.construct_get_chat_completions_result(resp.json())

    def emit_client_error_usage_event(self, error_message: str):
        with self.get_default_request_session(headers=_get_eval_session_headers()) as session:
            session.post(
                self.get_method_url("/agents/evaluation-client-usage-events"),
                json=self.proto_serde.construct_client_usage_events_request_json(
                    usage_events=[self.proto_serde.construct_client_error_usage_event_json(error_message=error_message)]
                ),
            )

    def emit_custom_metric_usage_event(
        self,
        *,
        custom_metrics: List[agent_custom_metrics.CustomMetric],
        eval_count: Optional[int],
        metric_stats: Optional[Dict[str, per_run_metrics.MetricAggregateData]] = None,
    ):
        # Use default retries. Don't need to use response
        with self.get_default_request_session(headers=_get_eval_session_headers()) as session:
            session.post(
                self.get_method_url("/agents/evaluation-client-usage-events"),
                json=self.proto_serde.construct_client_usage_events_request_json(
                    usage_events=[
                        self.proto_serde.construct_custom_metric_usage_event_json(
                            custom_metrics=custom_metrics,
                            eval_count=eval_count,
                            metric_stats=metric_stats,
                        )
                    ]
                ),
            )
