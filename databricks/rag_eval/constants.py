"""
File containing all the constants needed for the agent utils.
"""

# Common param names for the eval_fn in MLflow
MLFLOW_EVAL_FN_INPUTS = "inputs"
MLFLOW_EVAL_FN_PREDICTIONS = "predictions"
MLFLOW_EVAL_FN_CONTEXT = "context"
MLFLOW_EVAL_FN_TARGETS = "targets"

# Metrics
GROUND_TRUTH_RETRIEVAL_METRIC_NAMES = ["recall"]

# Configs
EVALUATOR_CONFIG_EXAMPLES_KEY_NAME = "examples_df"

DEFAULT_CONTEXT_CONCATENATION_DELIMITER = "\n"

CHUNK_CONTENT_IS_EMPTY_RATIONALE = "Chunk content is empty"
