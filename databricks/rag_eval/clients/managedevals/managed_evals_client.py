import json
import logging
import warnings
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    Union,
)

import pandas as pd
import requests
from mlflow.data import spark_dataset_source
from mlflow.genai.scheduled_scorers import ScorerScheduleConfig
from mlflow.genai.scorers.base import Scorer
from urllib3.util import retry

from databricks import version
from databricks.agents.utils.mlflow_utils import get_workspace_url, add_workspace_id_to_url
from databricks.rag_eval import entities, env_vars, rest_entities, schemas, session
from databricks.rag_eval.clients import databricks_api_client
from databricks.rag_eval.clients.databricks_api_client import raise_for_status
from databricks.rag_eval.clients.managedevals import dataset_utils
from databricks.rag_eval.monitoring import assessments
from databricks.rag_eval.utils import (
    NO_CHANGE,
    callable_utils,
    collection_utils,
    input_output_utils,
    request_utils,
)
from databricks.rag_eval.utils.scheduled_scorer_errors import (
    AtLeastOneUndeserializableScorerError,
    UndeserializableScorerError,
)

SESSION_ID_HEADER = "managed-evals-session-id"
CLIENT_VERSION_HEADER = "managed-evals-client-version"
SYNTHETIC_GENERATION_NUM_DOCS_HEADER = "managed-evals-synthetic-generation-num-docs"
SYNTHETIC_GENERATION_NUM_EVALS_HEADER = "managed-evals-synthetic-generation-num-evals"
USE_NOTEBOOK_CLUSTER_ID = False
# When using batch endpoints, limit batches to this size, in bytes
# Technically 1MB = 1048576 bytes, but we leave 48kB for overhead of HTTP headers/other json fluff.
_BATCH_SIZE_LIMIT = 1_000_000
# When using batch endpoints, limit batches to this number of rows.
# The service has a hard limit at 2K nodes updated per request; sometimes 1 row is more than 1 node.
_BATCH_QUANTITY_LIMIT = 100
# Default page size when doing paginated requests.
_DEFAULT_PAGE_SIZE = 500

TagType = TypedDict("TagType", {"tag_name": str, "tag_id": str})

_logger = logging.getLogger(__name__)


def get_default_retry_config():
    return retry.Retry(
        total=env_vars.AGENT_EVAL_GENERATE_EVALS_MAX_RETRIES.get(),
        backoff_factor=env_vars.AGENT_EVAL_GENERATE_EVALS_BACKOFF_FACTOR.get(),
        status_forcelist=[429, 500, 502, 503, 504],
        backoff_jitter=env_vars.AGENT_EVAL_GENERATE_EVALS_BACKOFF_JITTER.get(),
        allowed_methods=frozenset(["GET", "POST"]),  # by default, it doesn't retry on POST
    )


def get_batch_edit_retry_config():
    return retry.Retry(
        total=3,
        backoff_factor=10,  # Retry after 0, 10, 20, 40... seconds.
        status_forcelist=[429],  # Adding lots of evals in a row can result in rate limiting errors
        allowed_methods=["POST"],  # POST not retried by default
    )


def _get_default_headers() -> Dict[str, str]:
    """
    Constructs the default request headers.
    """
    headers = {
        CLIENT_VERSION_HEADER: version.VERSION,
    }

    return request_utils.add_traffic_id_header(headers)


def _get_synthesis_headers() -> Dict[str, str]:
    """
    Constructs the request headers for synthetic generation.
    """
    eval_session = session.current_session()
    if eval_session is None:
        return {}
    return request_utils.add_traffic_id_header(
        {
            CLIENT_VERSION_HEADER: version.VERSION,
            SESSION_ID_HEADER: eval_session.session_id,
            SYNTHETIC_GENERATION_NUM_DOCS_HEADER: str(eval_session.synthetic_generation_num_docs),
            SYNTHETIC_GENERATION_NUM_EVALS_HEADER: str(eval_session.synthetic_generation_num_evals),
        }
    )


def _get_judge_and_custom_metrics(
    metrics: Optional[list[assessments.AssessmentConfig]],
) -> (list[assessments.AssessmentConfig], list[assessments.CustomMetric]):
    if metrics is None:
        return None, None
    judge_metrics = [metric for metric in metrics if not isinstance(metric, assessments.CustomMetric)]
    custom_metrics = [metric for metric in metrics if isinstance(metric, assessments.CustomMetric)]
    return judge_metrics, custom_metrics


class ManagedEvalsClient(databricks_api_client.DatabricksAPIClient):
    """
    Client to interact with the managed-evals service.
    """

    def __init__(self):
        super().__init__(version="2.0")

    # override from DatabricksAPIClient
    def get_default_request_session(self, *args, **kwargs):
        session = super().get_default_request_session(*args, **kwargs)
        if USE_NOTEBOOK_CLUSTER_ID:
            from pyspark.sql import SparkSession

            spark = SparkSession.builder.getOrCreate()
            cluster_id = spark.conf.get("spark.databricks.clusterUsageTags.clusterId")
            session.params = {"compute_cluster_id": cluster_id}
        return session

    def gracefully_batch_post(
        self,
        url: str,
        all_items: Sequence[Any],
        request_body_create: Callable[[Iterable[Any]], Any],
        response_body_read: Callable[[Any], Iterable[Any]],
    ):
        with self.get_default_request_session(
            headers=_get_default_headers(),
            retry_config=get_batch_edit_retry_config(),
        ) as session:
            return_values = []
            for batch in collection_utils.safe_batch(
                all_items,
                batch_byte_limit=_BATCH_SIZE_LIMIT,
                batch_quantity_limit=_BATCH_QUANTITY_LIMIT,
            ):
                request_body = request_body_create(batch)

                response = session.post(url=url, json=request_body)
                try:
                    raise_for_status(response)
                except requests.HTTPError as e:
                    _logger.error(
                        f"Created {len(return_values)}/{len(all_items)} items before encountering an error.\n"
                        f"Returning successfully created items; please take care to avoid double-creating objects.\n{e}"
                    )
                    return return_values
                return_values.extend(response_body_read(response))
            return return_values

    def generate_questions(
        self,
        *,
        doc: entities.Document,
        num_questions: int,
        agent_description: Optional[str],
        question_guidelines: Optional[str],
        experiment_id: Optional[str] = None,
    ) -> List[entities.SyntheticQuestion]:
        """
        Generate synthetic questions for the given document.
        """
        request_json = {
            "doc_content": doc.content,
            "num_questions": num_questions,
            "agent_description": agent_description,
            "question_guidelines": question_guidelines,
        }
        if experiment_id is not None:
            request_json["experiment_id"] = experiment_id
        with self.get_default_request_session(
            get_default_retry_config(),
            headers=_get_synthesis_headers(),
        ) as session:
            resp = session.post(
                url=self.get_method_url("/managed-evals/generate-questions"),
                json=request_json,
            )

        raise_for_status(resp)

        response_json = resp.json()
        if "questions_with_context" not in response_json or "error" in response_json:
            raise ValueError(f"Invalid response: {response_json}")
        return [
            entities.SyntheticQuestion(
                question=question_with_context["question"],
                source_doc_uri=doc.doc_uri,
                source_context=question_with_context["context"],
            )
            for question_with_context in response_json["questions_with_context"]
        ]

    def generate_answer(
        self,
        *,
        question: entities.SyntheticQuestion,
        answer_types: Collection[entities.SyntheticAnswerType],
        experiment_id: Optional[str] = None,
    ) -> entities.SyntheticAnswer:
        """
        Generate synthetic answer for the given question.
        """
        request_json = {
            "question": question.question,
            "context": question.source_context,
            "answer_types": [str(answer_type) for answer_type in answer_types],
        }
        if experiment_id is not None:
            request_json["experiment_id"] = experiment_id

        with self.get_default_request_session(
            get_default_retry_config(),
            headers=_get_synthesis_headers(),
        ) as session:
            resp = session.post(
                url=self.get_method_url("/managed-evals/generate-answer"),
                json=request_json,
            )

        raise_for_status(resp)

        response_json = resp.json()
        return entities.SyntheticAnswer(
            question=question,
            synthetic_ground_truth=response_json.get("synthetic_ground_truth"),
            synthetic_grading_notes=response_json.get("synthetic_grading_notes"),
            synthetic_minimal_facts=response_json.get("synthetic_minimal_facts"),
        )

    def create_managed_evals_instance(
        self,
        *,
        instance_id: str,
        agent_name: Optional[str] = None,
        agent_serving_endpoint: Optional[str] = None,
        experiment_ids: Optional[Iterable[str]] = None,
    ) -> entities.EvalsInstance:
        """
        Creates a new Managed Evals instance.

        Args:
            instance_id: Managed Evals instance ID.
            agent_name: (optional) The name of the agent.
            agent_serving_endpoint: (optional) The name of the model serving endpoint that serves the agent.
            experiment_ids: (optional) The experiment IDs to associate with the instance.

        Returns:
            The created EvalsInstance.
        """
        evals_instance = entities.EvalsInstance(
            agent_name=agent_name,
            agent_serving_endpoint=agent_serving_endpoint,
            experiment_ids=experiment_ids if experiment_ids is not None else [],
        )
        request_body = {"instance": evals_instance.to_json()}
        with self.get_default_request_session(headers=_get_default_headers()) as session:
            response = session.post(
                url=self.get_method_url(f"/managed-evals/instances/{instance_id}"),
                json=request_body,
            )
        raise_for_status(response)
        return entities.EvalsInstance.from_json(response.json())

    def delete_managed_evals_instance(self, instance_id: str) -> None:
        """
        Deletes a Managed Evals instance.

        Args:
            instance_id: Managed Evals instance ID.
        """
        with self.get_default_request_session(headers=_get_default_headers()) as session:
            response = session.delete(
                url=self.get_method_url(f"/managed-evals/instances/{instance_id}"),
            )
        raise_for_status(response)

    def update_managed_evals_instance(
        self,
        *,
        instance_id: str,
        agent_name: Optional[str] = NO_CHANGE,
        agent_serving_endpoint: Optional[str] = NO_CHANGE,
        experiment_ids: List[str] = NO_CHANGE,
    ) -> entities.EvalsInstance:
        """
        Updates a Managed Evals instance.

        Args:
            instance_id: Managed Evals instance ID.
            agent_name: (optional) The name of the agent.
            agent_serving_endpoint: (optional) The name of the model serving endpoint that serves the agent.
            experiment_ids: (optional) The experiment IDs to associate with the instance.

        Returns:
            The updated EvalsInstance.
        """
        evals_instance = entities.EvalsInstance(
            agent_name=agent_name,
            agent_serving_endpoint=agent_serving_endpoint,
            experiment_ids=experiment_ids,
        )
        request_body = {
            "instance": evals_instance.to_json(),
            "update_mask": evals_instance.get_update_mask(),
        }

        with self.get_default_request_session(headers=_get_default_headers()) as session:
            response = session.patch(
                url=self.get_method_url(f"/managed-evals/instances/{instance_id}"),
                json=request_body,
            )
        raise_for_status(response)
        return entities.EvalsInstance.from_json(response.json())

    def get_managed_evals_instance(self, instance_id: str) -> entities.EvalsInstance:
        """
        Gets a Managed Evals instance.

        Args:
            instance_id: Managed Evals instance ID.

        Returns:
            The EvalsInstance.
        """
        with self.get_default_request_session(headers=_get_default_headers()) as session:
            response = session.get(
                url=self.get_method_url(f"/managed-evals/instances/{instance_id}/configuration"),
            )
        raise_for_status(response)
        return entities.EvalsInstance.from_json(response.json())

    def sync_evals_to_uc(self, instance_id: str):
        """
        Syncs evals from the evals table to a user-visible UC table.

        Args:
            instance_id: Managed Evals instance ID.
        """
        with self.get_default_request_session(headers=_get_default_headers()) as session:
            response = session.post(
                url=self.get_method_url(f"/managed-evals/instances/{instance_id}/evals:sync"),
            )
        raise_for_status(response)

    def add_evals(
        self,
        *,
        instance_id: str,
        evals: List[Dict],
    ) -> List[str]:
        """
        Add evals to the evals table.

        Args:
            instance_id: The name of the evals table.
            evals: The evals to add to the evals table.

        Returns:
            The eval IDs of the created evals.
        """
        evals = [
            {
                "request_id": e.get(schemas.REQUEST_ID_COL),
                "source_type": e.get(schemas.SOURCE_TYPE_COL),
                "source_id": e.get(schemas.SOURCE_ID_COL),
                "json_serialized_request": json.dumps(
                    input_output_utils.to_chat_completion_request(e.get(schemas.REQUEST_COL)),
                ),
                "expected_response": e.get(schemas.EXPECTED_RESPONSE_COL),
                "expected_facts": [{"fact": fact} for fact in e.get(schemas.EXPECTED_FACTS_COL, [])],
                "expected_retrieved_context": e.get(schemas.EXPECTED_RETRIEVED_CONTEXT_COL),
                "tag_ids": e.get("tag_ids", []),
                "review_status": e.get("review_status"),
            }
            for e in evals
        ]
        return self.gracefully_batch_post(
            url=self.get_method_url(f"/managed-evals/instances/{instance_id}/evals:batchCreate"),
            all_items=evals,
            request_body_create=lambda batch: {"evals": batch},
            response_body_read=lambda response: response.json().get("eval_ids", []),
        )

    def delete_evals(
        self,
        instance_id: str,
        *,
        eval_ids: List[str],
    ):
        """
        Delete evals from the evals table.
        """
        # Delete in a loop - this is inefficient but we don't have a batch delete endpoint yet.
        with self.get_default_request_session(headers=_get_default_headers()) as session:
            for eval in eval_ids:
                response = session.delete(
                    url=self.get_method_url(f"/managed-evals/instances/{instance_id}/evals/{eval}"),
                )
                raise_for_status(response)

    def list_tags(
        self,
        instance_id: str,
    ) -> List[TagType]:
        """
        List all tags in the evals table.

        Args:
            instance_id: The name of the evals table.

        Returns:
            A list of tags.
        """
        tags = []
        with self.get_default_request_session(headers=_get_default_headers()) as session:
            next_page_token = None
            for _ in range(100):
                response = session.get(
                    url=self.get_method_url(
                        f"/managed-evals/instances/{instance_id}/tags"
                        + (("?next_page_token=" + next_page_token) if next_page_token else "")
                    )
                )
                raise_for_status(response)
                json_response = response.json()
                tags.extend(json_response.get("tags", []))
                if not (next_page_token := json_response.get("next_page_token")):
                    break
            else:
                warnings.warn("Giving up fetching tags after 100 pages of tags; potential internal error.")
        return tags

    def batch_create_tags(
        self,
        *,
        instance_id: str,
        tag_names: Collection[str],
    ) -> List[str]:
        """
        Call the batchCreate endpoint to create tags.

        Args:
            instance_id: The name of the evals table.
            tag_names: The tag names to create.

        Returns:
            The tag IDs of the created tags.
        """
        tag_bodies = [{"tag_name": tag} for tag in tag_names]
        return self.gracefully_batch_post(
            url=self.get_method_url(f"/managed-evals/instances/{instance_id}/tags:batchCreate"),
            all_items=tag_bodies,
            request_body_create=lambda batch: {"tags": batch},
            response_body_read=lambda response: response.json().get("tag_ids", []),
        )

    def batch_create_eval_tags(
        self,
        instance_id: str,
        *,
        eval_tags: List[entities.EvalTag],
    ):
        """
        Batch tag evals.

        Args:
            instance_id: The name of the evals table.
            eval_tags: A list of eval-tags; each item of the list is one tag on an eval.
        """
        eval_tag_bodies = [et.to_json() for et in eval_tags]
        return self.gracefully_batch_post(
            url=self.get_method_url(f"/managed-evals/instances/{instance_id}/eval_tags:batchCreate"),
            all_items=eval_tag_bodies,
            request_body_create=lambda batch: {"eval_tags": batch},
            response_body_read=lambda response: response.json().get("eval_tags", []),
        )

    def batch_delete_eval_tags(
        self,
        instance_id: str,
        *,
        eval_tags: List[entities.EvalTag],
    ):
        """
        Batch untag evals.

        Args:
            instance_id: The name of the evals table.
            eval_tags: A list of eval-tags; each item of the list is one tag on an eval.
        """
        eval_tag_bodies = [et.to_json() for et in eval_tags]
        return self.gracefully_batch_post(
            url=self.get_method_url(f"/managed-evals/instances/{instance_id}/eval_tags:batchDelete"),
            all_items=eval_tag_bodies,
            request_body_create=lambda batch: {"eval_tags": batch},
            response_body_read=lambda response: response.json().get("eval_tags", []),
        )

    def update_eval_permissions(
        self,
        instance_id: str,
        *,
        add_emails: Optional[List[str]] = None,
        remove_emails: Optional[List[str]] = None,
    ):
        """Add or remove user permissions to edit an eval instance.

        Args:
            instance_id: The name of the evals table.
            add_emails: The emails to add to the permissions list.
            remove_emails: The emails to remove from the permissions list.
        """
        request_body = {"permission_change": {}}
        if add_emails:
            request_body["permission_change"]["add"] = [
                {
                    "user_email": email,
                    "permissions": ["WRITE"],
                }
                for email in add_emails
            ]
        if remove_emails:
            request_body["permission_change"]["remove"] = [
                {
                    "user_email": email,
                    "permissions": ["WRITE"],
                }
                for email in remove_emails
            ]

        with self.get_default_request_session(headers=_get_default_headers()) as session:
            response = session.post(
                url=self.get_method_url(f"/managed-evals/instances/{instance_id}/permissions"),
                json=request_body,
            )
        raise_for_status(response)

    def create_monitor(
        self,
        *,
        endpoint_name: str,
        assessments_config: assessments.AssessmentsSuiteConfig,
        experiment_id: str,
        monitoring_table: str | None = None,
    ) -> entities.Monitor | entities.ExternalMonitor:
        pause_status: Optional[str] = None
        if assessments_config.paused is not None:
            pause_status = (
                entities.SchedulePauseStatus.PAUSED
                if assessments_config.paused
                else entities.SchedulePauseStatus.UNPAUSED
            ).value

        schedule_config = rest_entities.ScheduleConfig(pause_status=pause_status)

        # Convert guidelines to NamedGuidelines format
        named_guidelines = None
        guidelines_judge = assessments_config.get_guidelines_judge()
        if guidelines_judge:
            entries = []
            for key, guidelines in guidelines_judge.guidelines.items():
                entries.append(rest_entities.NamedGuidelineEntry(key=key, guidelines=guidelines))
            named_guidelines = rest_entities.NamedGuidelines(entries=entries)

        metrics, custom_metrics = _get_judge_and_custom_metrics(assessments_config.assessments)

        monitor_rest = rest_entities.Monitor(
            experiment_id=experiment_id,
            evaluation_config=rest_entities.EvaluationConfig(
                metrics=(
                    [
                        rest_entities.AssessmentConfig(name=metric.name, sample_rate=metric.sample_rate)
                        for metric in metrics
                    ]
                    if metrics is not None
                    else None
                ),
                no_metrics=(metrics is not None) and len(metrics) == 0,
                named_guidelines=named_guidelines,
                custom_metrics=(
                    [
                        rest_entities.CustomMetricConfig(
                            name=custom_metric.metric_fn.name,
                            function_body=callable_utils.extract_function_body(custom_metric.metric_fn.eval_fn)[0],
                            sample_rate=custom_metric.sample_rate,
                        )
                        for custom_metric in custom_metrics
                    ]
                    if custom_metrics is not None
                    else None
                ),
            ),
            sampling=rest_entities.SamplingConfig(sampling_rate=assessments_config.sample),
            schedule=schedule_config,
            is_agent_external=bool(monitoring_table),
            evaluated_traces_table=monitoring_table,
        )
        request_body = {"monitor_config": monitor_rest.to_dict()}

        with self.get_default_request_session(headers=_get_default_headers()) as session:
            response = session.post(
                url=self.get_method_url(f"/managed-evals/monitors/{endpoint_name}"),
                json=request_body,
            )
        raise_for_status(response)
        monitor_info_rest = rest_entities.MonitorInfo.from_dict(response.json(), infer_missing=True)

        if monitor_info_rest.is_external:
            monitor = monitor_info_rest.to_external_monitor()
        else:
            monitor = monitor_info_rest.to_monitor()

        return monitor

    def update_monitor(
        self,
        *,
        endpoint_name: str,
        assessments_config: assessments.AssessmentsSuiteConfig,
    ) -> entities.Monitor:
        # Unless there are updates to the evaluation config, do not pass an EvaluationConfig to the service.
        evaluation_config: Optional[rest_entities.EvaluationConfig] = None

        # Update metrics if provided
        metrics = assessments_config.assessments
        if metrics is not None:
            judge_metrics, custom_metrics = _get_judge_and_custom_metrics(metrics)
            evaluation_config = rest_entities.EvaluationConfig(
                metrics=[
                    rest_entities.AssessmentConfig(name=metric.name, sample_rate=metric.sample_rate)
                    for metric in judge_metrics
                ],
                no_metrics=len(judge_metrics) == 0,
                custom_metrics=[
                    rest_entities.CustomMetricConfig(
                        name=custom_metric.metric_fn.name,
                        function_body=callable_utils.extract_function_body(custom_metric.metric_fn.eval_fn)[0],
                        sample_rate=custom_metric.sample_rate,
                    )
                    for custom_metric in custom_metrics
                ],
            )

            # Update guidelines if provided
            guidelines_judge = assessments_config.get_guidelines_judge()
            if guidelines_judge:
                entries = []
                for key, guidelines in guidelines_judge.guidelines.items():
                    entries.append(rest_entities.NamedGuidelineEntry(key=key, guidelines=guidelines))
                evaluation_config.named_guidelines = rest_entities.NamedGuidelines(entries=entries)

        sampling_config: Optional[rest_entities.SamplingConfig] = None
        if assessments_config.sample:
            sampling_config = rest_entities.SamplingConfig(sampling_rate=assessments_config.sample)

        schedule_config: Optional[rest_entities.ScheduleConfig] = None
        if assessments_config.paused is not None:
            pause_status = (
                entities.SchedulePauseStatus.PAUSED
                if assessments_config.paused
                else entities.SchedulePauseStatus.UNPAUSED
            )
            schedule_config = rest_entities.ScheduleConfig(
                pause_status=pause_status,
            )

        monitor_rest = rest_entities.Monitor(
            evaluation_config=evaluation_config,
            sampling=sampling_config,
            schedule=schedule_config,
        )

        request_body = {
            "monitor": monitor_rest.to_dict(),
            "update_mask": monitor_rest.get_update_mask(),
        }
        with self.get_default_request_session(headers=_get_default_headers()) as session:
            response = session.patch(
                url=self.get_method_url(f"/managed-evals/monitors/{endpoint_name}"),
                json=request_body,
            )
        raise_for_status(response)
        monitor_info_rest = rest_entities.MonitorInfo.from_dict(response.json(), infer_missing=True)

        if monitor_info_rest.is_external:
            monitor = monitor_info_rest.to_external_monitor()
        else:
            monitor = monitor_info_rest.to_monitor()

        monitoring_page_url = f"{get_workspace_url()}/ml/experiments/{monitor.experiment_id}/evaluation-monitoring?endpointName={endpoint_name}"
        user_message = f"""Updated monitor for endpoint "{endpoint_name}".

View traces: {monitoring_page_url}"""
        print(user_message)

        return monitor

    def list_monitors(self, *, experiment_id: str) -> list[entities.Monitor | entities.ExternalMonitor]:
        """List all monitors for a given experiment.

        Args:
            experiment_id (str): The ID of the experiment that the monitors are associated with.

        Returns:
            list[entities.Monitor]: A list of monitors associated with the given experiment.
        """
        with self.get_default_request_session(headers=_get_default_headers()) as session:
            response = session.post(
                url=self.get_method_url("/managed-evals/monitors"),
                json={"experiment_id": experiment_id},
            )
        raise_for_status(response)

        monitor_infos_raw = response.json().get("monitor_infos", [])
        monitor_infos_raw = map(
            lambda monitor_info: rest_entities.MonitorInfo.from_dict(monitor_info, infer_missing=True),
            monitor_infos_raw,
        )
        return list(
            map(
                lambda monitor_info: (
                    monitor_info.to_external_monitor() if monitor_info.is_external else monitor_info.to_monitor()
                ),
                monitor_infos_raw,
            )
        )

    def get_monitor(
        self,
        *,
        endpoint_name: str | None = None,
        monitoring_table: str | None = None,
    ) -> entities.Monitor | entities.ExternalMonitor:
        """Call the get monitor endpoint to get information on a monitor.

        Args:
            endpoint_name (str | None, optional): The name of the endpoint. Defaults to None.
            monitoring_table (str | None, optional): The fullname of the monitoring table. Defaults to None.

        Raises:
            ValueError: When both or neither of 'endpoint_name' and 'monitoring_table' are provided.

        Returns:
            Monitor | ExternalMonitor: The monitor object. If the server notes that the monitor is
                for an external agent, returns an ExternalMonitor. Otherwise, returns a Monitor.
        """
        has_endpoint = endpoint_name is not None
        has_table = monitoring_table is not None
        if not (has_endpoint ^ has_table):
            raise ValueError("Exactly one of 'endpoint_name' and 'monitoring_table' must be provided.")

        base_url = "monitors"
        urlpath = f"{base_url}/{endpoint_name}"
        if monitoring_table is not None:
            urlpath = f"{base_url}/table_name/{monitoring_table}/"

        request_body = {}
        with self.get_default_request_session(headers=_get_default_headers()) as session:
            response = session.get(
                url=self.get_method_url(f"/managed-evals/{urlpath}"),
                json=request_body,
            )
        raise_for_status(response)

        monitor_info_rest = rest_entities.MonitorInfo.from_dict(response.json(), infer_missing=True)

        if monitor_info_rest.is_external:
            return monitor_info_rest.to_external_monitor()
        return monitor_info_rest.to_monitor()

    def get_monitor_settings(
        self,
        *,
        endpoint_name: str,
    ) -> entities.MonitorSettings:
        """Call the GetMonitorSettings endpoint to get the monitor settings.

        Args:
            endpoint_name (str): The name of the endpoint.

        Returns:
            entities.MonitorSettings: The monitor settings.
        """
        with self.get_default_request_session(headers=_get_default_headers()) as session:
            response = session.get(
                url=self.get_method_url(f"/managed-evals/monitors/{endpoint_name}/settings"),
            )
        raise_for_status(response)
        return entities.MonitorSettings.from_dict(response.json())

    def delete_monitor(self, endpoint_name: str) -> None:
        request_body = {}
        with self.get_default_request_session(headers=_get_default_headers()) as session:
            response = session.delete(
                url=self.get_method_url(f"/managed-evals/monitors/{endpoint_name}"),
                json=request_body,
            )
        raise_for_status(response)

    def monitoring_usage_events(
        self,
        *,
        endpoint_name: str,
        job_id: str,
        run_id: str,
        run_ended: bool,
        num_traces: Optional[int],
        num_traces_evaluated: Optional[int],
        is_agent_external: Optional[bool],
        error_message: Optional[str] = None,
        stdout: Optional[str] = None,
        stderr: Optional[str] = None,
        additional_headers: Optional[Dict[str, str]] = None,
        source: Optional[str] = None,
    ) -> None:
        """
        :param endpoint_name: Name of endpoint associated with monitor.
        :param job_id: ID of job.
        :param run_id: ID of job run.
        :param num_traces: Number of traces in the run.
        :param num_traces_evaluated: Number of traces evaluated in the run.
        :param is_agent_external: Whether the agent is external.
        :param run_ended: Whether this usage event is triggered by the completion of a job run, either success or failure.
        :param error_message: Error message associated with failed run. May be empty.
        :param stdout: Standard output logs from the job execution.
        :param stderr: Standard error logs from the job execution.
        :param additional_headers: Additional headers to be passed when sending the request. May be empty.
        :param source: Source of the monitoring event. E.g., "LEGACY_MONITORING_JOB" or "TRACE_PROCESSOR".
        """

        job_start = None if run_ended else {}
        job_completion = (
            rest_entities.JobCompletionEvent(
                success=error_message is None,
                error_message=error_message,
                stdout=stdout,
                stderr=stderr,
            )
            if run_ended
            else None
        )

        monitoring_event = rest_entities.MonitoringEvent(
            job_start=job_start,
            job_completion=job_completion,
            source=source,
        )

        request_body = {
            "job_id": job_id,
            "run_id": run_id,
            "events": [monitoring_event.to_dict()],
        }

        metrics = []
        if num_traces is not None:
            num_traces_metric = rest_entities.MonitoringMetric(num_traces=num_traces)
            metrics.append(num_traces_metric.to_dict())

        if num_traces_evaluated is not None:
            num_traces_evaluated_metric = rest_entities.MonitoringMetric(num_traces_evaluated=num_traces_evaluated)
            metrics.append(num_traces_evaluated_metric.to_dict())

        if len(metrics) > 0:
            request_body["metrics"] = metrics

        if is_agent_external is not None:
            request_body["is_agent_external"] = is_agent_external

        default_headers = _get_default_headers()
        headers = {**default_headers, **(additional_headers or {})}

        with self.get_default_request_session(headers=headers) as session:
            response = session.post(
                url=self.get_method_url(f"/managed-evals/monitors/{endpoint_name}/usage-logging"),
                json=request_body,
            )
        raise_for_status(response)

    ##### Review App REST APIs #####
    def create_review_app(self, review_app: entities.ReviewApp) -> entities.ReviewApp:
        review_app_rest = rest_entities.ReviewApp.from_review_app(review_app)
        request_body = review_app_rest.to_dict()
        with self.get_default_request_session(headers=_get_default_headers()) as session:
            response = session.post(
                url=self.get_method_url("/managed-evals/review-apps"),
                json=request_body,
            )
        raise_for_status(response)
        review_app_rest: rest_entities.ReviewApp = rest_entities.ReviewApp.from_dict(response.json())
        return review_app_rest.to_review_app()

    def _paginate_review_apps(
        self, filter: str, page_token: Optional[str]
    ) -> Tuple[list[entities.ReviewApp], Optional[str]]:
        url = self.get_method_url(f"/managed-evals/review-apps?filter={filter}")
        if page_token:
            url += f"&page_token={page_token}"

        with self.get_default_request_session(headers=_get_default_headers()) as session:
            response = session.get(url=url)
        raise_for_status(response)
        response = response.json()
        next_page_token = response.get("next_page_token")
        review_apps: list[entities.ReviewApp] = []
        for app_dict in response.get("review_apps", []):
            review_apps.append(rest_entities.ReviewApp.from_dict(app_dict).to_review_app())
        return review_apps, next_page_token

    def list_review_apps(self, filter: str) -> list[entities.ReviewApp]:
        # Url encode the filter string
        filter = requests.utils.quote(filter)
        next_page_token = None
        all_review_apps: list[entities.ReviewApp] = []
        while True:
            review_apps, next_page_token = self._paginate_review_apps(filter, next_page_token)
            all_review_apps.extend(review_apps)
            if not next_page_token:
                break

        return all_review_apps

    def update_review_app(self, review_app: entities.ReviewApp, update_mask: str) -> entities.ReviewApp:
        review_app_rest = rest_entities.ReviewApp.from_review_app(review_app)
        request_body = review_app_rest.to_dict()
        with self.get_default_request_session(headers=_get_default_headers()) as session:
            response = session.patch(
                url=self.get_method_url(
                    f"/managed-evals/review-apps/{review_app.review_app_id}?update_mask={update_mask}"
                ),
                json=request_body,
            )
        raise_for_status(response)
        review_app_rest: rest_entities.ReviewApp = rest_entities.ReviewApp.from_dict(response.json())
        return review_app_rest.to_review_app()

    def create_labeling_session(
        self,
        review_app: entities.ReviewApp,
        name: str,
        *,
        # Must be workspace users for now due to ACL
        assigned_users: list[str],
        # agent names must already be added to the backend.
        agent: Optional[str],
        # the schema names, must be already added to backend.
        label_schemas: list[Union[str, entities.LabelSchema]],
        enable_multi_turn_chat: bool = False,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> entities.LabelingSession:
        # Validate label schemas.
        label_schema_names: list[str] = []
        for s in label_schemas:
            if isinstance(s, entities.LabelSchema):
                label_schema_names.append(s.name)
            elif isinstance(s, str):
                label_schema_names.append(s)
            else:
                raise ValueError(f"Invalid type for label_schemas: {type(s)}. Must be str or LabelSchema.")

        labeling_session_rest = rest_entities.LabelingSession(
            labeling_session_id=None,  # Not yet created.
            mlflow_run_id=None,  # Not yet created.
            name=name,
            assigned_users=assigned_users,
            agent=rest_entities.AgentRef(agent_name=agent) if agent else None,
            labeling_schemas=[rest_entities.LabelingSchemaRef(name=name) for name in label_schema_names],
            additional_configs=rest_entities.AdditionalConfigs(
                disable_multi_turn_chat=not enable_multi_turn_chat,
                custom_inputs_json=json.dumps(custom_inputs) if custom_inputs else None,
            ),
        )
        request_body = labeling_session_rest.to_dict()
        with self.get_default_request_session(headers=_get_default_headers()) as session:
            response = session.post(
                url=self.get_method_url(f"/managed-evals/review-apps/{review_app.review_app_id}/labeling-sessions"),
                json=request_body,
            )
        raise_for_status(response)
        labeling_session_rest: rest_entities.LabelingSession = rest_entities.LabelingSession.from_dict(response.json())
        session_url = f"{review_app.url}/tasks/labeling/{labeling_session_rest.labeling_session_id}"
        session_url = add_workspace_id_to_url(session_url)
        return labeling_session_rest.to_labeling_session(
            review_app.review_app_id, review_app.experiment_id, session_url
        )

    def _paginate_labeling_sessions(
        self, review_app: entities.ReviewApp, page_token: Optional[str]
    ) -> Tuple[list[entities.LabelingSession], Optional[str]]:
        url = self.get_method_url(
            f"/managed-evals/review-apps/{review_app.review_app_id}/labeling-sessions?page_size={_DEFAULT_PAGE_SIZE}"
        )
        if page_token:
            url += f"&page_token={page_token}"

        with self.get_default_request_session(headers=_get_default_headers()) as session:
            response = session.get(url=url)
        raise_for_status(response)
        response = response.json()
        next_page_token = response.get("next_page_token")
        sessions: list[entities.LabelingSession] = []
        for session_dict in response.get("labeling_sessions", []):
            rest_session: rest_entities.LabelingSession = rest_entities.LabelingSession.from_dict(session_dict)
            session_url = f"{review_app.url}/tasks/labeling/{rest_session.labeling_session_id}"
            session_url = add_workspace_id_to_url(session_url)
            sessions.append(
                rest_session.to_labeling_session(review_app.review_app_id, review_app.experiment_id, session_url)
            )
        return sessions, next_page_token

    def list_labeling_sessions(self, review_app: entities.ReviewApp) -> list[entities.LabelingSession]:
        next_page_token = None
        all_sessions: list[entities.LabelingSession] = []
        while True:
            sessions, next_page_token = self._paginate_labeling_sessions(review_app, next_page_token)
            all_sessions.extend(sessions)
            if not next_page_token:
                break

        return all_sessions

    def _paginate_labeling_items(
        self, review_app_id: str, labeling_session_id: str, page_token: Optional[str]
    ) -> Tuple[list[rest_entities.Item], Optional[str]]:
        url = self.get_method_url(
            f"/managed-evals/review-apps/{review_app_id}/labeling-sessions/{labeling_session_id}/items?page_size={_DEFAULT_PAGE_SIZE}"
        )
        if page_token:
            url += f"&page_token={page_token}"

        with self.get_default_request_session(headers=_get_default_headers()) as session:
            response = session.get(url=url)
        raise_for_status(response)
        response = response.json()
        next_page_token = response.get("next_page_token")
        items = [rest_entities.Item.from_dict(item) for item in response.get("items", [])]
        return items, next_page_token

    def list_items_in_labeling_session(self, labeling_session: entities.LabelingSession) -> list[rest_entities.Item]:
        next_page_token = None
        all_items: list[rest_entities.Item] = []
        while True:
            items, next_page_token = self._paginate_labeling_items(
                labeling_session.review_app_id,
                labeling_session.labeling_session_id,
                next_page_token,
            )
            all_items.extend(items)
            if not next_page_token:
                break

        return all_items

    def update_labeling_session(
        self,
        labeling_session: entities.LabelingSession,
        update_mask: str,
    ) -> entities.LabelingSession:
        labeling_session_rest = rest_entities.LabelingSession(
            labeling_session_id=labeling_session.labeling_session_id,
            mlflow_run_id=labeling_session.mlflow_run_id,
            name=labeling_session.name,
            assigned_users=labeling_session.assigned_users,
            agent=(rest_entities.AgentRef(agent_name=labeling_session.agent) if labeling_session.agent else None),
            labeling_schemas=[
                rest_entities.LabelingSchemaRef(name=schema_name) for schema_name in labeling_session.label_schemas
            ],
            additional_configs=rest_entities.AdditionalConfigs(
                disable_multi_turn_chat=not labeling_session.enable_multi_turn_chat
            ),
        )
        request_body = labeling_session_rest.to_dict()
        with self.get_default_request_session(headers=_get_default_headers()) as session:
            response = session.patch(
                url=self.get_method_url(
                    f"/managed-evals/review-apps/{labeling_session.review_app_id}/labeling-sessions/{labeling_session.labeling_session_id}?update_mask={update_mask}"
                ),
                json=request_body,
            )
        raise_for_status(response)
        labeling_session_rest: rest_entities.LabelingSession = rest_entities.LabelingSession.from_dict(response.json())
        return labeling_session_rest.to_labeling_session(
            labeling_session.review_app_id,
            labeling_session.experiment_id,
            labeling_session.url,
        )

    def delete_labeling_session(self, review_app_id: str, labeling_session_id: str) -> None:
        with self.get_default_request_session(headers=_get_default_headers()) as session:
            response = session.delete(
                url=self.get_method_url(
                    f"/managed-evals/review-apps/{review_app_id}/labeling-sessions/{labeling_session_id}"
                ),
            )
        raise_for_status(response)

    def batch_create_items_in_labeling_session(
        self,
        labeling_session: entities.LabelingSession,
        trace_ids: Optional[list[str]] = None,
        dataset_id: Optional[str] = None,
        dataset_record_ids: Optional[list[str]] = None,
    ) -> None:
        items = []
        review_app_id = labeling_session.review_app_id
        labeling_session_id = labeling_session.labeling_session_id
        if trace_ids:
            items.extend([{"source": {"trace_id": trace_id}} for trace_id in trace_ids])

        assert (dataset_id is None and dataset_record_ids is None) or (
            dataset_id and dataset_record_ids
        ), "If dataset_id is provided, dataset_record_ids must also be provided."

        if dataset_id:
            items.extend(
                [
                    {
                        "source": {
                            "dataset_record": {
                                "dataset_id": dataset_id,
                                "dataset_record_id": record_id,
                            }
                        }
                    }
                    for record_id in dataset_record_ids
                ]
            )
        request_body = {"items": items}
        with self.get_default_request_session(headers=_get_default_headers()) as session:
            response = session.post(
                url=self.get_method_url(
                    f"/managed-evals/review-apps/{review_app_id}/labeling-sessions/{labeling_session_id}/items:batchCreate"
                ),
                json=request_body,
            )
        raise_for_status(response)

    def run_metric_backfill(
        self,
        *,
        experiment_id: str,
        scorer_sample_rates: dict[str, float],
        start_timestamp_ms: int,
        end_timestamp_ms: int,
        trace_ids: Optional[list[str]] = None,
    ) -> dict[str, str]:
        """
        Run a metric backfill for a given experiment, endpoint, and time range.
        Args:
            experiment_id: The ID of the experiment to backfill.
            scorer_sample_rates: A dictionary of scorer names to sample rates.
            start_timestamp_ms: The start timestamp in milliseconds.
            end_timestamp_ms: The end timestamp in milliseconds.
            trace_ids: A list of trace IDs to backfill.
        Returns:
            A dictionary containing the job_id of the created backfill job.
        """

        monitors = self.list_monitors(experiment_id=experiment_id)
        if not monitors:
            raise ValueError(f"No monitors found for experiment_id '{experiment_id}'")

        first_monitor = monitors[0]
        if hasattr(first_monitor, "endpoint_name"):
            endpoint_name = first_monitor.endpoint_name
        elif hasattr(first_monitor, "_legacy_ingestion_endpoint_name"):
            endpoint_name = first_monitor._legacy_ingestion_endpoint_name
        else:
            raise ValueError(f"Unable to determine endpoint name from monitor: {first_monitor}")

        request_body = {
            "experiment_id": experiment_id,
            "endpoint_name": endpoint_name,
            "scorer_sample_rates": [{"name": name, "sample_rate": rate} for name, rate in scorer_sample_rates.items()],
            "start_timestamp_ms": start_timestamp_ms,
            "end_timestamp_ms": end_timestamp_ms,
            "trace_ids": trace_ids,
        }

        with self.get_default_request_session(headers=_get_default_headers()) as session:
            response = session.post(
                url=self.get_method_url(f"/managed-evals/scheduled-scorers/{experiment_id}/backfill"),
                json=request_body,
            )
        raise_for_status(response)
        return response.json()

    def create_scheduled_scorers(
        self,
        *,
        experiment_id: str,
        scheduled_scorers: list[ScorerScheduleConfig],
    ) -> list[ScorerScheduleConfig]:
        """
        Create scheduled scorers for an experiment.

        Args:
            experiment_id: The ID of the experiment.
            scheduled_scorers: List of ScorerScheduleConfig objects.

        Returns:
            The created scheduled scorers as ScorerScheduleConfig objects.
        """
        # Convert ScorerScheduleConfig objects to REST entities
        scorer_configs = []
        for config in scheduled_scorers:
            scorer_configs.append(self._scorer_schedule_config_to_rest_entity(config))

        scheduled_scorers_entity = rest_entities.ScheduledScorers(scorers=scorer_configs)

        request_body = {
            "experiment_id": experiment_id,
            "scheduled_scorers": scheduled_scorers_entity.to_dict(),
        }

        with self.get_default_request_session(headers=_get_default_headers()) as session:
            response = session.post(
                url=self.get_method_url(f"/managed-evals/scheduled-scorers/{experiment_id}"),
                json=request_body,
            )
        raise_for_status(response)
        result = rest_entities.ScheduledScorersInfo.from_dict(response.json())

        return [self._rest_entity_to_scorer_config(scorer) for scorer in result.scheduled_scorers.scorers or []]

    def get_scheduled_scorers(
        self,
        *,
        experiment_id: str,
    ) -> list[ScorerScheduleConfig]:
        """
        Get scheduled scorers for an experiment.

        Args:
            experiment_id: The ID of the experiment.

        Returns:
            The scheduled scorers as ScorerScheduleConfig objects.
        """
        with self.get_default_request_session(headers=_get_default_headers()) as session:
            response = session.get(
                url=self.get_method_url(f"/managed-evals/scheduled-scorers/{experiment_id}"),
            )
        raise_for_status(response)
        result = rest_entities.ScheduledScorersInfo.from_dict(response.json())

        successful_configs = []
        errors = []

        for scorer in result.scheduled_scorers.scorers or []:
            try:
                config = self._rest_entity_to_scorer_config(scorer)
                successful_configs.append(config)
            except UndeserializableScorerError as e:
                errors.append(e)

        if errors:
            raise AtLeastOneUndeserializableScorerError(errors)

        return successful_configs

    def update_scheduled_scorers(
        self,
        *,
        experiment_id: str,
        scheduled_scorers: list[ScorerScheduleConfig],
        update_mask: str,
    ) -> list[ScorerScheduleConfig]:
        """
        Update scheduled scorers for an experiment.

        Args:
            experiment_id: The ID of the experiment.
            scheduled_scorers: List of ScorerScheduleConfig objects.
            update_mask: The update mask for the fields to update.

        Returns:
            The updated scheduled scorers as ScorerScheduleConfig objects.
        """
        scorer_configs = []
        for config in scheduled_scorers:
            scorer_configs.append(self._scorer_schedule_config_to_rest_entity(config))

        scheduled_scorers_entity = rest_entities.ScheduledScorers(scorers=scorer_configs)

        request_body = {
            "experiment_id": experiment_id,
            "scheduled_scorers": scheduled_scorers_entity.to_dict(),
            "update_mask": update_mask,
        }

        with self.get_default_request_session(headers=_get_default_headers()) as session:
            response = session.patch(
                url=self.get_method_url(f"/managed-evals/scheduled-scorers/{experiment_id}"),
                json=request_body,
            )
        raise_for_status(response)
        result = rest_entities.ScheduledScorersInfo.from_dict(response.json())

        return [self._rest_entity_to_scorer_config(scorer) for scorer in result.scheduled_scorers.scorers or []]

    def delete_scheduled_scorers(
        self,
        *,
        experiment_id: str,
    ) -> None:
        """
        Delete scheduled scorers for an experiment.

        Args:
            experiment_id: The ID of the experiment.
        """
        with self.get_default_request_session(headers=_get_default_headers()) as session:
            response = session.delete(
                url=self.get_method_url(f"/managed-evals/scheduled-scorers/{experiment_id}"),
            )
        raise_for_status(response)

    def delete_scheduled_scorer(
        self,
        *,
        experiment_id: str,
        scheduled_scorer_name: str,
    ) -> None:
        """
        Delete a specific scheduled scorer from an experiment.

        Args:
            experiment_id: The ID of the experiment.
            scheduled_scorer_name: The name of the scheduled scorer to delete.

        Raises:
            ValueError: If no scheduled scorer with the specified name is found.
        """
        # Get current schedulers using raw REST entities
        with self.get_default_request_session(headers=_get_default_headers()) as session:
            response = session.get(
                url=self.get_method_url(f"/managed-evals/scheduled-scorers/{experiment_id}"),
            )
        raise_for_status(response)
        result = rest_entities.ScheduledScorersInfo.from_dict(response.json())

        # Filter out the scorer to delete using raw REST entities
        existing_scorers = result.scheduled_scorers.scorers or []
        updated_scorers = [scorer for scorer in existing_scorers if scorer.name != scheduled_scorer_name]

        if len(updated_scorers) == len(existing_scorers):
            raise ValueError(f"No registered scorer found with name '{scheduled_scorer_name}'")

        # Update with the filtered list
        scheduled_scorers_entity = rest_entities.ScheduledScorers(scorers=updated_scorers)

        request_body = {
            "experiment_id": experiment_id,
            "scheduled_scorers": scheduled_scorers_entity.to_dict(),
            "update_mask": "scheduled_scorers.scorers",
        }

        with self.get_default_request_session(headers=_get_default_headers()) as session:
            response = session.patch(
                url=self.get_method_url(f"/managed-evals/scheduled-scorers/{experiment_id}"),
                json=request_body,
            )
        raise_for_status(response)

    def cleanup_metric_backfill(
        self,
        *,
        experiment_id: str,
        job_id: str,
    ) -> None:
        """
        Cleanup metric backfill resources for a given experiment.

        Args:
            experiment_id: The ID of the experiment.
            job_id: The ID of the backfill job to cleanup.
        """
        with self.get_default_request_session(headers=_get_default_headers()) as session:
            response = session.delete(
                url=self.get_method_url(f"/managed-evals/experiments/{experiment_id}/backfill/{job_id}"),
            )
        raise_for_status(response)

    def _scorer_schedule_config_to_rest_entity(self, scorer_config) -> rest_entities.ScorerConfig:
        """Convert a ScorerScheduleConfig to a REST API ScorerConfig."""
        import json

        from mlflow.genai.scorers.builtin_scorers import BuiltInScorer

        sample_rate = scorer_config.sample_rate

        if not isinstance(sample_rate, (int, float)):
            raise TypeError(
                f"Invalid sample_rate for scorer '{scorer_config.scheduled_scorer_name}'. Expected a valid float, got invalid value '{sample_rate}'."
            )

        serialized_scorer = json.dumps(scorer_config.scorer.model_dump())

        if isinstance(scorer_config.scorer, BuiltInScorer):
            builtin_scorer = rest_entities.BuiltinScorer(name=scorer_config.scorer.name)
            return rest_entities.ScorerConfig(
                name=scorer_config.scheduled_scorer_name,
                serialized_scorer=serialized_scorer,
                builtin=builtin_scorer,
                sample_rate=sample_rate,
                filter_string=scorer_config.filter_string,
            )
        else:
            custom_scorer = rest_entities.CustomScorer()
            return rest_entities.ScorerConfig(
                name=scorer_config.scheduled_scorer_name,
                serialized_scorer=serialized_scorer,
                custom=custom_scorer,
                sample_rate=sample_rate,
                filter_string=scorer_config.filter_string,
            )

    def _rest_entity_to_scorer_config(self, rest_scorer_config: rest_entities.ScorerConfig) -> ScorerScheduleConfig:
        """Convert a REST API ScorerConfig to a ScorerScheduleConfig."""
        import json

        try:
            serialized_data = json.loads(rest_scorer_config.serialized_scorer or "{}")
        except json.JSONDecodeError:
            serialized_data = {}

        try:
            scorer = Scorer.model_validate(serialized_data)
        except Exception as e:
            raise UndeserializableScorerError(scorer_name=rest_scorer_config.name, error=e)

        return ScorerScheduleConfig(
            scorer=scorer,
            scheduled_scorer_name=rest_scorer_config.name,
            sample_rate=(rest_scorer_config.sample_rate if rest_scorer_config.sample_rate is not None else 1.0),
            filter_string=rest_scorer_config.filter_string,
        )

    def _parse_dataset_source(self, dataset: entities.Dataset) -> entities.Dataset:
        """Parse the dataset source field from JSON to SparkDatasetSource if present."""
        dataset.source = spark_dataset_source.SparkDatasetSource.from_json(dataset.source) if dataset.source else None
        return dataset

    def create_dataset(self, uc_table_name: str, experiment_ids: list[str]) -> entities.Dataset:
        url = self.get_method_url("/managed-evals/datasets")
        if experiment_ids:
            url += f"?experiment_ids={','.join(experiment_ids)}"
        dataset_rest = rest_entities.Dataset(
            name=uc_table_name,
            source_type="databricks-uc-table",
        )
        request_body = dataset_rest.to_dict()
        with self.get_default_request_session(headers=_get_default_headers()) as session:
            response = session.post(
                url=url,
                json=request_body,
            )
        raise_for_status(response)
        # The REST and public python entities are the same for datasets.
        dataset = entities.Dataset.from_dict(response.json())
        return self._parse_dataset_source(dataset)

    def sync_dataset_to_uc(self, dataset_id: str, uc_table_name: str) -> None:
        dataset_rows = [
            entities.DatasetRow.from_rest_dataset_record(record) for record in self.list_dataset_records(dataset_id)
        ]
        df = pd.DataFrame.from_records([row.to_dict() for row in dataset_rows])
        dataset_utils.sync_dataset_to_uc(uc_table_name, df)

    def get_dataset(self, dataset_id: str) -> entities.Dataset:
        with self.get_default_request_session(headers=_get_default_headers()) as session:
            response = session.get(url=self.get_method_url(f"/managed-evals/datasets/{dataset_id}"))
        raise_for_status(response)
        dataset = entities.Dataset.from_dict(response.json())
        return self._parse_dataset_source(dataset)

    def delete_dataset(self, dataset_id: str) -> None:
        with self.get_default_request_session(headers=_get_default_headers()) as session:
            response = session.delete(url=self.get_method_url(f"/managed-evals/datasets/{dataset_id}"))
        raise_for_status(response)

    def update_dataset(self, dataset: entities.Dataset, update_mask: str) -> entities.Dataset:
        # The REST and public python entities are the same.
        request_body = dataset.to_dict()
        with self.get_default_request_session(headers=_get_default_headers()) as session:
            response = session.patch(
                url=self.get_method_url(f"/managed-evals/datasets/{dataset.dataset_id}?update_mask={update_mask}"),
                json=request_body,
            )
        raise_for_status(response)
        # The REST and public python entities are the same.
        dataset = entities.Dataset.from_dict(response.json())
        return self._parse_dataset_source(dataset)

    def _paginate_dataset_records(
        self, dataset_id: str, page_token: Optional[str]
    ) -> Tuple[list[rest_entities.DatasetRecord], Optional[str]]:
        url = self.get_method_url(f"/managed-evals/datasets/{dataset_id}/records?page_size={_DEFAULT_PAGE_SIZE}")
        if page_token:
            url += f"&page_token={page_token}"

        with self.get_default_request_session(headers=_get_default_headers()) as session:
            response = session.get(url=url)
        raise_for_status(response)
        response = response.json()
        next_page_token = response.get("next_page_token")
        records = [rest_entities.DatasetRecord.from_dict(record) for record in response.get("dataset_records", [])]
        return records, next_page_token

    def list_dataset_records(self, dataset_id: str) -> list[rest_entities.DatasetRecord]:
        next_page_token = None
        all_records: list[rest_entities.DatasetRecord] = []
        while True:
            records, next_page_token = self._paginate_dataset_records(dataset_id, next_page_token)
            all_records.extend(records)
            if not next_page_token:
                break

        return all_records

    def batch_create_dataset_records(
        self,
        uc_table_name: str,
        dataset_id: str,
        records: list[rest_entities.DatasetRecord],
    ) -> None:
        request_items = [{"dataset_id": dataset_id, "dataset_record": r.to_dict()} for r in records]

        def request_body_create(batch_requests):
            return {"requests": batch_requests}

        def response_body_read(response):
            return []

        self.gracefully_batch_post(
            url=self.get_method_url(f"/managed-evals/datasets/{dataset_id}/records:batchCreate"),
            all_items=request_items,
            request_body_create=request_body_create,
            response_body_read=response_body_read,
        )

    def upsert_dataset_record_expectations(
        self,
        uc_table_name: str,
        dataset_id: str,
        dataset_record_id: str,
        expectations: dict[str, rest_entities.ExpectationValue],
    ) -> None:
        request_body = {"expectations": {key: value.to_dict() for key, value in expectations.items()}}
        with self.get_default_request_session(headers=_get_default_headers()) as session:
            response = session.post(
                url=self.get_method_url(
                    f"/managed-evals/datasets/{dataset_id}/records/{dataset_record_id}/expectations"
                ),
                json=request_body,
            )
        raise_for_status(response)

    def update_dataset_record(
        self,
        dataset_id: str,
        dataset_record: rest_entities.DatasetRecord,
        update_mask: str,
    ) -> rest_entities.DatasetRecord:
        request_body = dataset_record.to_dict()
        with self.get_default_request_session(headers=_get_default_headers()) as session:
            response = session.patch(
                url=self.get_method_url(
                    f"/managed-evals/datasets/{dataset_id}/records/{dataset_record.dataset_record_id}?update_mask={update_mask}"
                ),
                json=request_body,
            )
        raise_for_status(response)
        return rest_entities.DatasetRecord.from_dict(response.json())

    def start_trace_archival(
        self,
        experiment_id: str,
        table_fullname: str,
    ) -> str:
        """
        Start trace archival for an experiment.

        Args:
            experiment_id: The MLflow experiment ID
            table_fullname: The full name of the Unity Catalog table for archiving traces

        Returns:
            The job ID of the created archive job

        Raises:
            requests.HTTPError: If the experiment/monitor is not found (404) or if archival is already configured (409)
        """
        request_body = {
            "experiment_id": experiment_id,
            "archive_table_fullname": table_fullname,
        }

        with self.get_default_request_session(headers=_get_default_headers()) as session:
            response = session.post(
                url=self.get_method_url(f"/managed-evals/experiments/{experiment_id}/trace-archival"),
                json=request_body,
            )

        try:
            raise_for_status(response)
        except requests.HTTPError as e:
            if response.status_code == 404:
                raise requests.HTTPError(f"Experiment '{experiment_id}' not found.") from e
            elif response.status_code == 409:
                raise requests.HTTPError(
                    f"Trace archival is already configured for experiment '{experiment_id}'. "
                    f"Please stop existing archival first using mlflow.tracing.archival.disable_databricks_trace_archival()."
                ) from e
            else:
                raise

        return response.json().get("job_id")

    def stop_trace_archival(
        self,
        experiment_id: str,
    ) -> None:
        """
        Stop trace archival for an experiment.

        Args:
            experiment_id: The MLflow experiment ID

        Raises:
            requests.HTTPError: If the experiment/monitor is not found or has no archival configured (404)
        """
        request_body = {
            "experiment_id": experiment_id,
        }

        with self.get_default_request_session(headers=_get_default_headers()) as session:
            response = session.delete(
                url=self.get_method_url(f"/managed-evals/experiments/{experiment_id}/trace-archival"),
                json=request_body,
            )

        try:
            raise_for_status(response)
        except requests.HTTPError as e:
            if response.status_code == 404:
                raise requests.HTTPError(
                    f"Cannot stop trace archival for experiment '{experiment_id}'. "
                    f"Either the experiment does not exist or trace archival is not currently configured."
                ) from e
            else:
                raise

    def record_archive_job_event(
        self,
        *,
        experiment_id: str,
        job_id: str,
        run_id: str,
        workspace_id: str,
        metadata: dict[str, Any],
    ) -> None:
        """
        Record an archive job event to the managed-evals server.

        Args:
            experiment_id: The MLflow experiment ID
            job_id: The Databricks job ID
            run_id: The Databricks run ID
            workspace_id: The Databricks workspace ID
            metadata: The event metadata (job_success or job_fail)
        """
        request_body = {
            "experiment_id": experiment_id,
            "job_id": job_id,
            "run_id": run_id,
            "workspace_id": workspace_id,
            **metadata,
        }

        with self.get_default_request_session(headers=_get_default_headers()) as session:
            response = session.post(
                url=self.get_method_url(f"/managed-evals/archive-job/{experiment_id}/record-event"),
                json=request_body,
            )
        raise_for_status(response)

    def record_custom_judge_model_event(
        self,
        *,
        request_id: str | None,
        experiment_id: str,
        job_id: str | None,
        job_run_id: str | None,
        workspace_id: str,
        model_provider: str,
        endpoint_name: str,
        success_metadata: dict | None = None,
        failure_metadata: dict | None = None,
    ) -> None:
        if success_metadata and failure_metadata:
            raise ValueError("Cannot specify both success_metadata and failure_metadata")
        if not success_metadata and not failure_metadata:
            raise ValueError("Must specify either success_metadata or failure_metadata")

        request_body = {
            "experiment_id": experiment_id,
            "workspace_id": workspace_id,
            "model_provider": model_provider,
            "endpoint_name": endpoint_name,
        }

        if request_id:
            request_body["request_id"] = request_id

        if job_id:
            request_body["job_id"] = job_id

        if job_run_id:
            request_body["run_id"] = job_run_id

        if success_metadata:
            request_body["success_metadata"] = success_metadata

        if failure_metadata:
            request_body["failure_metadata"] = failure_metadata

        with self.get_default_request_session(headers=_get_default_headers()) as session:
            response = session.post(
                url=self.get_method_url("/managed-evals/custom-judge-model/record-event"),
                json=request_body,
            )
        raise_for_status(response)
