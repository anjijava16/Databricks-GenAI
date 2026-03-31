"""Env vars that can be set for the RAG eval."""

import os

# noinspection PyProtectedMember


# Source:
# https://github.com/mlflow/mlflow/blob/812f1bef02804b7ad875834b35e3677d22323c18/mlflow/environment_variables.py#L8-L76
class _EnvironmentVariable:
    """
    Represents an environment variable.
    """

    def __init__(self, name, type_, default):
        self.name = name
        self.type = type_
        self.default = default

    @property
    def defined(self):
        return self.name in os.environ

    def get_raw(self):
        return os.getenv(self.name)

    def set(self, value):
        os.environ[self.name] = str(value)

    def unset(self):
        os.environ.pop(self.name, None)

    def get(self):
        """
        Reads the value of the environment variable if it exists and converts it to the desired
        type. Otherwise, returns the default value.
        """
        if (val := self.get_raw()) is not None:
            try:
                return self.type(val)
            except Exception as e:
                raise ValueError(f"Failed to convert {val!r} to {self.type} for {self.name}: {e}")
        return self.default

    def __str__(self):
        return f"{self.name} (default: {self.default}, type: {self.type.__name__})"

    def __repr__(self):
        return repr(self.name)

    def __format__(self, format_spec: str) -> str:
        return self.name.__format__(format_spec)


class _BooleanEnvironmentVariable(_EnvironmentVariable):
    """
    Represents a boolean environment variable.
    """

    def __init__(self, name, default):
        # `default not in [True, False, None]` doesn't work because `1 in [True]`
        # (or `0 in [False]`) returns True.
        if not (default is True or default is False or default is None):
            raise ValueError(f"{name} default value must be one of [True, False, None]")
        super().__init__(name, bool, default)

    def get(self):
        if not self.defined:
            return self.default

        val = os.getenv(self.name)
        lowercased = val.lower()
        if lowercased not in ["true", "false", "1", "0"]:
            raise ValueError(
                f"{self.name} value must be one of ['true', 'false', '1', '0'] (case-insensitive), " f"but got {val}"
            )
        return lowercased in ["true", "1"]


# Whether to enable rate limiting for the assessment.
# If set to ``False``, the rate limiter will be disabled for all assessments.
RAG_EVAL_ENABLE_RATE_LIMIT_FOR_ASSESSMENT = _BooleanEnvironmentVariable(
    "RAG_EVAL_ENABLE_RATE_LIMIT_FOR_ASSESSMENT", True
)

# Rate limit quota for the assessment.
RAG_EVAL_RATE_LIMIT_QUOTA = _EnvironmentVariable("RAG_EVAL_RATE_LIMIT_QUOTA", float, 8.0)

# Rate limit time_window for the assessment. Unit: seconds.
RAG_EVAL_RATE_LIMIT_TIME_WINDOW_IN_SECONDS = _EnvironmentVariable(
    "RAG_EVAL_RATE_LIMIT_TIME_WINDOW_IN_SECONDS", float, 1.0
)

# Maximum number of workers to run the eval job.
RAG_EVAL_MAX_WORKERS = _EnvironmentVariable("RAG_EVAL_MAX_WORKERS", int, 10)

# Client name for the eval session.
RAG_EVAL_EVAL_SESSION_CLIENT_NAME = _EnvironmentVariable(
    "RAG_EVAL_EVAL_SESSION_CLIENT_NAME", str, "databricks-agents-sdk"
)

AGENT_EVAL_ENABLE_MULTI_TURN_EVALUATION = _BooleanEnvironmentVariable("AGENT_EVAL_ENABLE_MULTI_TURN_EVALUATION", True)

AGENT_EVAL_TRACE_SERVER_ENABLED = _BooleanEnvironmentVariable("AGENT_EVAL_TRACE_SERVER_ENABLED", False)

# Flag to gate whether to log to MLflow during evaluation (e.g., monitoring should disable this)
AGENT_EVAL_LOG_TRACES_TO_MLFLOW_ENABLED = _BooleanEnvironmentVariable("AGENT_EVAL_LOG_TRACES_TO_MLFLOW_ENABLED", True)

# Flag to gate whether to log artifacts to MLflow during evaluation (e.g., eval results json table)
AGENT_EVAL_LOG_ARTIFACTS_TO_MLFLOW_ENABLED = _BooleanEnvironmentVariable(
    "AGENT_EVAL_LOG_ARTIFACTS_TO_MLFLOW_ENABLED", False
)

# ================ Retry Configurations for LLM Judges ================

# Maximum number of retries.
RAG_EVAL_LLM_JUDGE_MAX_JUDGE_ERROR_RETRIES = _EnvironmentVariable("RAG_EVAL_LLM_JUDGE_MAX_JUDGE_ERROR_RETRIES", int, 5)

# Backoff factor in seconds.
RAG_EVAL_LLM_JUDGE_BACKOFF_FACTOR = _EnvironmentVariable("RAG_EVAL_LLM_JUDGE_BACKOFF_FACTOR", float, 5)

# Jitter in seconds to add to the backoff factor.
RAG_EVAL_LLM_JUDGE_BACKOFF_JITTER = _EnvironmentVariable("RAG_EVAL_LLM_JUDGE_BACKOFF_JITTER", float, 20)

RAG_EVAL_LLM_JUDGE_WAIT_EXP_BASE = _EnvironmentVariable("RAG_EVAL_LLM_JUDGE_WAIT_MULTIPLIER", float, 1.3)

RAG_EVAL_LLM_JUDGE_MAX_WAIT = _EnvironmentVariable("RAG_EVAL_LLM_JUDGE_MAX_WAIT", float, 240)

RAG_EVAL_LLM_JUDGE_MIN_WAIT = _EnvironmentVariable("RAG_EVAL_LLM_JUDGE_MIN_WAIT", float, 30)

# ================ Retry Configurations for Other Judge Service Endpoints ================

# Maximum number of retries.
AGENT_EVAL_JUDGE_SERVICE_ERROR_RETRIES = _EnvironmentVariable("AGENT_EVAL_JUDGE_SERVICE_ERROR_RETRIES", int, 10)

# Backoff factor in seconds.
AGENT_EVAL_JUDGE_SERVICE_BACKOFF_FACTOR = _EnvironmentVariable("AGENT_EVAL_JUDGE_SERVICE_BACKOFF_FACTOR", float, 0)

# Jitter in seconds to add to the backoff factor.
AGENT_EVAL_JUDGE_SERVICE_JITTER = _EnvironmentVariable("AGENT_EVAL_JUDGE_SERVICE_JITTER", float, 5)

# ================ Harness Limits for User Inputs ================

# Maximum number of rows in the input eval dataset.
RAG_EVAL_MAX_INPUT_ROWS = _EnvironmentVariable("RAG_EVAL_MAX_INPUT_ROWS", int, 10000)

# Maximum number of guidelines that can be evaluated at a time.
AGENT_EVAL_MAX_NUM_GUIDELINES = _EnvironmentVariable("AGENT_EVAL_MAX_NUM_GUIDELINES", int, 20)

# ================ Configurations for Synthetic Generation ================

# Maximum number of retries when calling the synthetic generation APIs.
AGENT_EVAL_GENERATE_EVALS_MAX_RETRIES = _EnvironmentVariable("AGENT_EVAL_GENERATE_EVALS_MAX_RETRIES", int, 60)

# Backoff factor in seconds when calling the synthetic generation APIs.
# Set to 0 because max retries is a large number, and we don't want the backoff to be too long.
AGENT_EVAL_GENERATE_EVALS_BACKOFF_FACTOR = _EnvironmentVariable("AGENT_EVAL_GENERATE_EVALS_BACKOFF_FACTOR", float, 0)

# Jitter in seconds to add to the backoff factor when calling the synthetic generation APIs.
# Set to 30 seconds because backend has a per-minute limit.
AGENT_EVAL_GENERATE_EVALS_BACKOFF_JITTER = _EnvironmentVariable("AGENT_EVAL_GENERATE_EVALS_BACKOFF_JITTER", float, 30)

# Maximum number of questions to request at a single time from a document.
AGENT_EVAL_GENERATE_EVALS_MAX_NUM_EVALS_PER_CHUNK = _EnvironmentVariable(
    "AGENT_EVAL_GENERATE_EVALS_MAX_NUM_EVALS_PER_CHUNK", int, 5
)

# Maximum number of tokens from which to request question generations.
AGENT_EVAL_GENERATE_EVALS_MAX_NUM_TOKENS_PER_CHUNK = _EnvironmentVariable(
    "AGENT_EVAL_GENERATE_EVALS_MAX_NUM_TOKENS_PER_CHUNK", int, 8000
)

# Rate limit config for the question generation API.
AGENT_EVAL_GENERATE_EVALS_QUESTION_GENERATION_RATE_LIMIT_QUOTA = _EnvironmentVariable(
    "AGENT_EVAL_GENERATE_EVALS_QUESTION_GENERATION_RATE_LIMIT_QUOTA", float, 1.0
)
AGENT_EVAL_GENERATE_EVALS_QUESTION_GENERATION_RATE_LIMIT_TIME_WINDOW_IN_SECONDS = _EnvironmentVariable(
    "AGENT_EVAL_GENERATE_EVALS_QUESTION_GENERATION_RATE_LIMIT_TIME_WINDOW_IN_SECONDS",
    float,
    1.0,
)

# Rate limit config for the answer generation API.
AGENT_EVAL_GENERATE_EVALS_ANSWER_GENERATION_RATE_LIMIT_QUOTA = _EnvironmentVariable(
    "AGENT_EVAL_GENERATE_EVALS_ANSWER_GENERATION_RATE_LIMIT_QUOTA", float, 1.0
)
AGENT_EVAL_GENERATE_EVALS_ANSWER_GENERATION_RATE_LIMIT_TIME_WINDOW_IN_SECONDS = _EnvironmentVariable(
    "AGENT_EVAL_GENERATE_EVALS_ANSWER_GENERATION_RATE_LIMIT_TIME_WINDOW_IN_SECONDS",
    float,
    1.0,
)
