import requests

from databricks.sdk.service.serving import ServingEndpointDetailed

_MONITOR_EXPERIMENT_ID_TAG = "MONITOR_EXPERIMENT_ID"


def _get_endpoint_experiment_id_opt(serving_endpoint: ServingEndpointDetailed):
    # Try to get the experiment ID to use from tracing from the monitor
    # attached to the serving endpoint, if the serving endpoint already exists and
    # monitoring is enabled
    # First check for a legacy monitor
    try:
        if monitor := _get_monitor_opt_for_endpoint(agent_endpoint_name=serving_endpoint.name):
            return monitor.experiment_id
    except Exception:
        # If monitoring is disabled etc or fetching the current active
        # monitor fails for some reason, continue
        pass
    # Then check for a new external monitor, via the MONITOR_EXPERIMENT_ID tag,
    # and return its value if present
    tags_dict = {tag.key: tag for tag in serving_endpoint.tags} if serving_endpoint.tags else {}
    if external_monitor_exp_id := tags_dict.get(_MONITOR_EXPERIMENT_ID_TAG):
        return external_monitor_exp_id.value


def _get_monitor_opt_for_endpoint(agent_endpoint_name: str):
    """
    Try to get the monitor for the given agent endpoint name
    """
    from databricks.rag_eval.monitoring.api import (
        _get_monitor_internal,
    )

    try:
        return _get_monitor_internal(endpoint_name=agent_endpoint_name)
    except requests.HTTPError as e:
        # If _get_monitor_internal failed with a 404, the monitor does not exist, so return None
        # Otherwise, rethrow the exception
        if e.response.status_code != 404:
            raise
