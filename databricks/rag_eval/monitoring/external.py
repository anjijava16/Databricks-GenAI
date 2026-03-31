from databricks.agents.utils.uc import _sanitize_model_name
from databricks.rag_eval.utils import uc_utils


def build_endpoint_name(monitoring_table_name: str) -> str:
    """Builds the name of the serving endpoint associated with a given monitoring table.

    Args:
        monitoring_table_name (str): The name of the monitoring table.

    Returns:
        str: The name of the serving endpoint.
    """
    prefix = "monitor_"
    truncated_monitoring_table_name = monitoring_table_name[: uc_utils.MAX_UC_ENTITY_NAME_LEN - len(prefix)]
    sanitized_truncated_model_name = _sanitize_model_name(truncated_monitoring_table_name)
    return f"{prefix}{sanitized_truncated_model_name}"
