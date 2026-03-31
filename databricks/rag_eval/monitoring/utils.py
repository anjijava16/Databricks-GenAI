import hashlib
import re
import time

import mlflow
from mlflow import entities as mlflow_entities
from mlflow.tracking import fluent

from databricks.agents.utils.mlflow_utils import get_workspace_url, get_workspace_id
from databricks.rag_eval.utils import uc_utils


def _get_experiment_id(experiment_id: str | None) -> str:
    """Get experiment ID, using active experiment if None provided.

    Args:
        experiment_id: The experiment ID or None to use active experiment

    Returns:
        The experiment ID to use

    Raises:
        ValueError: If experiment_id is None and no active experiment exists
    """
    if experiment_id is None:
        experiment_id = get_active_mlflow_experiment_id()
        if experiment_id is None:
            raise ValueError(
                "No active MLflow experiment found. Please provide an experiment_id or "
                "run this code within an active MLflow experiment."
            )
    return experiment_id


def get_monitoring_page_url(experiment_id: str) -> str:
    """Get the monitoring page URL.

    Args:
        experiment_id (str): id of the experiment

    Returns:
        str: the monitoring page URL
    """
    base_url = f"{get_workspace_url()}/ml/experiments/{experiment_id}?compareRunsMode=TRACES"
    workspace_id = get_workspace_id()
    if workspace_id is not None:
        return f"{base_url}&o={workspace_id}"
    return base_url


def get_active_mlflow_experiment_id() -> str | None:
    """Get the active MLflow experiment ID.

    If there is no active experiment, or the active experiment
    is the default experiment, return None.
    """
    try:
        experiment_id = fluent._get_experiment_id()
    except Exception:
        return None

    if experiment_id == mlflow.tracking.default_experiment.DEFAULT_EXPERIMENT_ID:
        return None

    return experiment_id


def get_databricks_mlflow_experiment(
    *,
    experiment_id: str | None = None,
    experiment_name: str | None = None,
) -> mlflow_entities.Experiment:
    if experiment_id:
        return mlflow.get_experiment(experiment_id=experiment_id)

    if experiment_name:
        exp = mlflow.get_experiment_by_name(name=experiment_name)
        if exp is None:
            raise ValueError(
                f"Experiment with name '{experiment_name}' does not exist. "
                "Please create the experiment before using it."
            )

    experiment_id = get_active_mlflow_experiment_id()
    if experiment_id is None:
        raise ValueError("Please provide an experiment_name or run this code within an active experiment.")

    return mlflow.get_experiment(experiment_id=str(experiment_id))


def simple_hex_hash(s: str) -> str:
    """
    Generate a hex hash string of length 6 for the given input string.

    Args:
        s (str): The input string.

    Returns:
        str: A 6-character long hex hash.
    """
    # Create an MD5 hash object with the encoded input string
    hash_object = hashlib.md5(s.encode("utf-8"))

    # Get the full hex digest and then take the first 6 characters
    return hash_object.hexdigest()[:6]


def create_checkpoint_table_entity_name(experiment_name: str) -> str:
    """Create a checkpoint table entity name based on the experiment name and current time.

    Example: "test_experiment_name" -> "ckpt_test_experiment_name_a1b2c3"

    Args:
        experiment_name (str): Name of the experiment.

    Returns:
        str: The generated checkpoint table entity name.
    """

    hash = simple_hex_hash(f"{experiment_name}_{time.time()}")
    prefix = "ckpt"
    # Subtract 8 from limit to provide extra room for `monitor_` prefix when building endpoint name.
    # This allows the endpoint name to also avoid conflicts thanks to the hash.
    max_name_chars = uc_utils.MAX_UC_ENTITY_NAME_LEN - len(hash) - len(prefix) - 2 - 8
    processed_experiment_name = re.sub(r"[./ ]", "_", experiment_name)

    return f"{prefix}_{processed_experiment_name[:max_name_chars]}_{hash}"
