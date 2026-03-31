"""Functions for managing trace archival for agent monitoring."""

from databricks.rag_eval import context


def _get_managed_evals_client():
    """Get the managed evaluations client."""
    return context.get_context().build_managed_evals_client()


@context.eval_context
def enable_trace_archival(
    *,
    experiment_id: str,
    table_fullname: str,
) -> str:
    """
    Enable trace archival for an experiment.

    This function enables archiving traces from the specified experiment to a Unity Catalog table.
    Archival allows for long-term storage and analysis of agent traces.

    Args:
        experiment_id: The MLflow experiment ID to archive traces from
        table_fullname: The fully qualified name of the Unity Catalog table
                        (e.g., "catalog.schema.table") where traces will be archived

    Returns:
        The job ID of the created archive job

    Raises:
        requests.HTTPError: If the experiment/monitor is not found (404) or if archival
                           is already configured (409)

    Example:
        >>> job_id = enable_trace_archival(
        ...     experiment_id="123456",
        ...     table_fullname="main.default.agent_traces"
        ... )
    """
    client = _get_managed_evals_client()
    job_id = client.start_trace_archival(experiment_id, table_fullname)

    print(f"Successfully enabled trace archival for experiment '{experiment_id}'.")
    print(f"Archive job ID: {job_id}")
    print(f"Traces will be archived to table: {table_fullname}")

    return job_id


@context.eval_context
def disable_trace_archival(
    *,
    experiment_id: str,
) -> None:
    """
    Disable trace archival for an experiment.

    This function disables the archival job for the specified experiment. The archived data
    in the Unity Catalog table is preserved and not deleted.

    Args:
        experiment_id: The MLflow experiment ID to stop archiving traces for

    Raises:
        requests.HTTPError: If the experiment/monitor is not found or has no archival
                           configured (404)

    Example:
        >>> disable_trace_archival(experiment_id="123456")
    """
    client = _get_managed_evals_client()
    client.stop_trace_archival(experiment_id)

    print(f"Successfully disabled trace archival for experiment '{experiment_id}'.")
    print("Note: Archived data has been preserved and not deleted.")
