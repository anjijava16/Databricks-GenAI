from databricks.rag_eval import context


def _get_managed_evals_client():
    return context.get_context().build_managed_evals_client()


def record_judge_model_usage_success(
    *,
    request_id: str | None,
    experiment_id: str,
    job_id: str | None,
    job_run_id: str | None,
    workspace_id: str,
    model_provider: str,
    endpoint_name: str,
    num_prompt_tokens: int | None,
    num_completion_tokens: int | None,
) -> None:
    client = _get_managed_evals_client()
    success_metadata = {}

    if num_prompt_tokens is not None:
        success_metadata["num_prompt_tokens"] = num_prompt_tokens

    if num_completion_tokens is not None:
        success_metadata["num_completion_tokens"] = num_completion_tokens

    client.record_custom_judge_model_event(
        request_id=request_id,
        experiment_id=experiment_id,
        job_id=job_id,
        job_run_id=job_run_id,
        workspace_id=workspace_id,
        model_provider=model_provider,
        endpoint_name=endpoint_name,
        success_metadata=success_metadata,
    )


def record_judge_model_usage_failure(
    *,
    experiment_id: str,
    job_id: str | None,
    job_run_id: str | None,
    workspace_id: str,
    model_provider: str,
    endpoint_name: str,
    error_code: str,
    error_message: str,
) -> None:
    client = _get_managed_evals_client()
    failure_metadata = {
        "error_code": error_code,
        "error_message": error_message,
    }

    client.record_custom_judge_model_event(
        request_id=None,
        experiment_id=experiment_id,
        job_id=job_id,
        job_run_id=job_run_id,
        workspace_id=workspace_id,
        model_provider=model_provider,
        endpoint_name=endpoint_name,
        failure_metadata=failure_metadata,
    )
