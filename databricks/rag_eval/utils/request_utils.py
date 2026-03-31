import os
from typing import Dict

DATABRICKS_TRAFFIC_ID_HEADER = "x-databricks-traffic-id"
DATABRICKS_TRAFFIC_ID_ENV_VAR = "X_DATABRICKS_TRAFFIC_ID"


def add_traffic_id_header(headers: Dict[str, str]) -> Dict[str, str]:
    """
    Adds the traffic ID header to the given headers if the traffic id environment variable is set.
    """
    traffic_id = os.getenv(DATABRICKS_TRAFFIC_ID_ENV_VAR)
    if traffic_id is not None and traffic_id.strip() != "":
        headers[DATABRICKS_TRAFFIC_ID_HEADER] = traffic_id
    return headers
