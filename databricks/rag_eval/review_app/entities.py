import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Literal, Optional, Union

import mlflow
import pandas as pd
from dataclasses_json import dataclass_json

from databricks.rag_eval import schemas
from databricks.rag_eval.review_app.rest_entities import (
    InputCategorical,
    InputCategoricalList,
    InputNumeric,
    InputText,
    InputTextList,
)
from databricks.rag_eval.utils import serialization_utils

from .utils import (
    add_users_to_dataset,
    add_users_to_experiment,
    add_users_to_serving_endpoint,
    assessments_to_expectations_dict,
    batch_link_traces_to_run,
    extract_user_email_from_trace,
    log_trace_to_experiment,
)

if TYPE_CHECKING:
    from databricks.rag_eval.clients.managedevals.managed_evals_client import (
        ManagedEvalsClient,
    )
from databricks.sdk import WorkspaceClient


def _get_client() -> "ManagedEvalsClient":
    from databricks.rag_eval import context
    from databricks.rag_eval.clients.managedevals.managed_evals_client import (
        ManagedEvalsClient,  # noqa: F401
    )

    @context.eval_context
    def getter():
        return context.get_context().build_managed_evals_client()

    return getter()


@dataclass_json
@dataclass(frozen=True)
class LabelSchema:
    """A label schema for collecting input from stakeholders."""

    name: str  # Must be unique across the review app.
    type: Literal["feedback", "expectation"]

    # Title shown in the SME UI as the title of the task.
    # e.g., "Does the response contain sensitive information?"
    title: str

    ### Only one of categorical, categorical_list, text, text_list, numeric needs to be defined. ###
    input: Union[InputCategorical, InputCategoricalList, InputText, InputTextList, InputNumeric]

    # Instruction shown in the SME UI to help SMEs understand the task.
    # e.g., "Please review the response and check if it contains any sensitive information."
    instruction: Optional[str] = None
    enable_comment: bool = False


@dataclass_json
@dataclass
class Agent:
    """The agent configuration, used for generating responses in the review app."""

    agent_name: str
    model_serving_endpoint: str


@dataclass_json
@dataclass
class LabelingSession:
    """A session for labeling items in the review app."""

    name: str
    assigned_users: list[str]
    agent: Optional[str]
    label_schemas: list[str]

    ## Auto-created.
    labeling_session_id: str
    mlflow_run_id: str
    review_app_id: str
    experiment_id: str
    url: str

    enable_multi_turn_chat: bool
    custom_inputs: Optional[dict[str, Any]] = None

    def add_dataset(self, dataset_name: str, record_ids: Optional[list[str]] = None) -> "LabelingSession":
        """Add a dataset to the labeling session.

        Args:
            dataset_name: The name of the dataset.
            record_ids: Optional. The individiual record ids to be added to the session. If not
                provided, all records in the dataset will be added.
        """

        assert isinstance(dataset_name, str), "`dataset_name` must be a string"
        if record_ids:
            assert all(isinstance(record_id, str) for record_id in record_ids), "`record_ids` must be a list of strings"

        if self.assigned_users:
            add_users_to_dataset(self.assigned_users, dataset_name)

        # Get all the records in the dataset.
        w = WorkspaceClient()
        dataset_id = w.tables.get(dataset_name).table_id
        records = _get_client().list_dataset_records(dataset_id=dataset_id)
        dataset_records = {r.dataset_record_id: r for r in records}

        # Get the existing record ids in the session.
        items = _get_client().list_items_in_labeling_session(self)
        existing_record_ids = set(
            [
                item.source.dataset_record.dataset_record_id
                for item in items
                if item.source and item.source.dataset_record and item.source.dataset_record.dataset_record_id
            ]
        )

        # Add the records that are not already in the session.
        record_ids_to_add = set(record_ids or dataset_records.keys()) - existing_record_ids
        if record_ids_to_add:
            if self.agent:
                # Add the record ids to the session.
                _get_client().batch_create_items_in_labeling_session(
                    self, dataset_id=dataset_id, dataset_record_ids=record_ids_to_add
                )
            else:
                traces: list[mlflow.entities.Trace] = []
                # Get the associated traces to add to the session.
                for record_id in record_ids_to_add:
                    record = dataset_records[record_id]
                    if not record.source or not record.source.trace or not record.source.trace.trace_id:
                        raise ValueError(
                            f"Record {record_id} has no trace and this session has no agent. "
                            "Either provide records that have associated traces or "
                            "add an agent to this session to generate responses"
                        )
                    trace = mlflow.get_trace(record.source.trace.trace_id)
                    traces.append(trace)

                self.add_traces(traces)

        return self

    def add_traces(
        self,
        traces: Union[Iterable[mlflow.entities.Trace], Iterable[str], pd.DataFrame],
    ) -> "LabelingSession":
        """Add traces to the labeling session.

        Args:
            traces: Can be either:
                a) a pandas DataFrame with a 'trace' column. The 'trace' column should contain
                either `mlflow.entities.Trace` objects or their json string representations.
                b) an iterable of `mlflow.entities.Trace` objects.
                c) an iterable of json string representations of `mlflow.entities.Trace` objects.
        """

        if isinstance(traces, pd.DataFrame):
            if "trace" not in traces.columns:
                raise ValueError("traces must have a 'trace' column like the result of mlflow.search_traces()")
            # Convert the pd.Series to a list of dicts.
            traces = traces["trace"].to_list()

        # If the traces are not already deserialized, deserialize them.
        traces = [serialization_utils.deserialize_trace(t) if isinstance(t, str) else t for t in traces]

        # Extract the trace ids from the traces.
        trace_ids: list[str] = []
        for trace in traces:
            if not trace:
                raise ValueError(
                    "trace can not be None. " "Must be `mlflow.entities.Trace` or its json string representation."
                )

            # Log the trace to this experiment (without linking to run yet).
            trace = log_trace_to_experiment(trace, self.experiment_id)

            trace_ids.append(trace.info.trace_id)

        # Batch link all traces to the run (if run_id is provided).
        if self.mlflow_run_id and trace_ids:
            batch_link_traces_to_run(trace_ids, self.mlflow_run_id)

        _get_client().batch_create_items_in_labeling_session(self, trace_ids=trace_ids)
        return self

    # TODO: Bring back session.insert(...) method after customer feedback.
    # def insert(
    #     self,
    #     items: Union[list[Dict], pd.DataFrame, "pyspark.sql.DataFrame"],  # noqa: F821
    # ) -> "LabelingSession":
    #     """Insert items into the labeling session.

    #     Args:
    #         items: A list of dictionaries, a pandas DataFrame, or a Spark DataFrame.
    #     """
    #     df: pd.DataFrame

    #     if isinstance(items, list) and all(isinstance(item, dict) for item in items):
    #         df = pd.DataFrame.from_records(items)
    #     else:
    #         df = spark_utils.normalize_spark_df(items)

    #     eval_items = EvaluationDataframe(df).eval_items

    #     # Materialize a trace for each row before inserting into the session
    #     trace_ids = []
    #     for i, eval_item in enumerate(eval_items):
    #         if not eval_item.raw_response and not self.agent:
    #             raise ValueError(
    #                 f"items[{i}] has no response and this session has no agent assigned. "
    #                 "An agent is required for this session to generate responses"
    #             )

    #         # TODO: store these under trace.info.assessments once we have the mlflow python APIs.
    #         (
    #             eval_item.expected_facts,
    #             eval_item.ground_truth_answer,
    #             eval_item.ground_truth_retrieval_context,
    #             eval_item.guidelines,
    #         )  # noqa: F841

    #         # TODO: Figure out what to do with context: make a span of type RETRIEVER?
    #         (eval_item.retrieval_context,)  # noqa: F841

    #         # Make a baseline trace with the request and any expectations.
    #         trace = mlflow.entities.Trace(
    #             info=mlflow.entities.TraceInfo(
    #                 request_id="0",
    #                 experiment_id=self.experiment_id,
    #                 timestamp_ms=int(time.time() * 1000),
    #                 execution_time_ms=0,
    #                 status=mlflow.entities.trace_status.TraceStatus.OK,
    #             ),
    #             data=mlflow.entities.TraceData(
    #                 spans=[],
    #                 request=json.dumps(eval_item.raw_request),
    #                 response=json.dumps(eval_item.raw_response),
    #             ),
    #         )
    #         # If no response is provided, the SME UI should call the model serving endpoint to
    #         # generate a response, and store the resulting trace under the run. Therefore if no
    #         # response is provided, we do not associate the baseline trace with the same run to
    #         # avoid duplicating traces.
    #         run_id = self.review_app_id if eval_item.raw_response else None

    #         log_trace_to_experiment(trace, self.experiment_id, run_id)
    #         trace_ids.append(trace.info.trace_id)

    #     _get_client().batch_create_items_in_labeling_session(self, trace_ids=trace_ids)
    #     return self

    def sync_expectations(self, to_dataset: str) -> None:
        """Sync the expectations from the labeling session to a dataset."""

        mlflow.tracking.set_tracking_uri("databricks")
        items = _get_client().list_items_in_labeling_session(self)
        records_to_insert: list[dict] = []

        for item in items:
            for chat_round in item.chat_rounds:
                if not chat_round.trace_id:
                    continue
                trace = mlflow.get_trace(chat_round.trace_id)
                if not trace:
                    print(f"Trace {chat_round.trace_id} not found in the databricks server.")
                    continue

                record = {
                    schemas.INPUTS_COL: json.loads(trace.data.request),
                    schemas.TRACE_COL: trace,
                }

                user_email = extract_user_email_from_trace(trace)
                if user_email:
                    record[schemas.SOURCE_ID_COL] = user_email
                    record[schemas.SOURCE_TYPE_COL] = "HUMAN"

                expectations = assessments_to_expectations_dict(trace.info.assessments)
                if expectations:
                    record[schemas.EXPECTATIONS_COL] = expectations
                records_to_insert.append(record)

        # The dataset will take care of deduping records with the same inputs.
        from databricks.agents import datasets

        datasets.get_dataset(to_dataset).merge_records(records_to_insert)

    def set_assigned_users(self, assigned_users: list[str]) -> "LabelingSession":
        """Set the assigned users for the labeling session."""

        add_users_to_experiment(assigned_users, self.experiment_id)

        client = _get_client()
        if self.agent:
            review_app = client.list_review_apps(filter=f"experiment_id={self.experiment_id}")[0]
            agent_config = next(a for a in review_app.agents if a.agent_name == self.agent)
            if agent_config.model_serving_endpoint:
                add_users_to_serving_endpoint(assigned_users, agent_config.model_serving_endpoint)

        self.assigned_users = assigned_users
        if self.assigned_users:
            print(
                "Make sure that the new assigned users have SELECT privilege "
                "to any previously added datasets in this session."
            )
        return client.update_labeling_session(self, update_mask="assigned_users")


@dataclass_json
@dataclass
class ReviewApp:
    """A review app is used to collect feedback from stakeholders for a given experiment.

    Attributes:
        review_app_id: The ID of the review app.
        experiment_id: The ID of the experiment.
        url: The URL of the review app for stakeholders to provide feedback.
        agents: The agents to be used to generate responses.
        label_schemas: The label schemas to be used in the review app.
    """

    review_app_id: str
    experiment_id: str
    url: str
    agents: list[Agent] = field(default_factory=list)
    label_schemas: list[LabelSchema] = field(default_factory=list)

    def add_agent(self, *, agent_name: str, model_serving_endpoint: str, overwrite: bool = False) -> "ReviewApp":
        """Add an agent to the review app to be used to generate responses."""
        assert isinstance(agent_name, str), "`agent_name` must be a string"

        prev_agent = next((x for x in self.agents if x.agent_name == agent_name), None)
        if prev_agent:
            if overwrite:
                logging.warn(
                    f"The agent with name '{agent_name}' already exists and will be overwritten. "
                    "This impacts any labeling sessions using this agent."
                )
                self.remove_agent(agent_name)
            elif prev_agent.model_serving_endpoint != model_serving_endpoint:
                raise ValueError(
                    f"Agent {agent_name} already exists with a different model serving endpoint. "
                    "Please use overwrite=True to update the model serving endpoint."
                )
            else:
                return self

        self.agents.append(Agent(agent_name=agent_name, model_serving_endpoint=model_serving_endpoint))
        return _get_client().update_review_app(self, update_mask="agents")

    def remove_agent(self, agent_name: str) -> "ReviewApp":
        """Remove an agent from the review app."""
        agent_to_remove = next((x for x in self.agents if x.agent_name == agent_name), None)
        if not agent_to_remove:
            raise ValueError(f"Agent {agent_name} not found")

        self.agents.remove(agent_to_remove)
        return _get_client().update_review_app(self, update_mask="agents")

    def create_label_schema(
        self,
        name: str,
        *,
        type: Literal["feedback", "expectation"],
        title: str,
        input: Union[
            InputCategorical,
            InputCategoricalList,
            InputText,
            InputTextList,
            InputNumeric,
        ],
        instruction: Optional[str] = None,
        enable_comment: bool = False,
        overwrite: bool = False,
    ) -> LabelSchema:
        """Create a new label schema for the review app.

        A label schema defines the type of input that stakeholders will provide when labeling items
        in the review app.

        Args:
            name: The name of the label schema. Must be unique across the review app.
            type: The type of the label schema. Either "feedback" or "expectation".
            title: The title of the label schema shown to stakeholders.
            input: The input type of the label schema.
            instruction: Optional. The instruction shown to stakeholders.
            enable_comment: Optional. Whether to enable comments for the label schema.
            overwrite: Optional. Whether to overwrite the existing label schema with the same name.
        """
        assert type in [
            "feedback",
            "expectation",
        ], "type must be 'feedback' or 'expectation'"

        # If overwrite is true, check if the label schema already exists and remove it.
        if overwrite and any(x.name == name for x in self.label_schemas):
            logging.warn(
                f"Label schema with name '{name}' already exists and will be overwritten. "
                "This impacts any labeling sessions using this schema."
            )
            self.delete_label_schema(name)

        label_schema = LabelSchema(
            name=name,
            type=type,
            title=title,
            input=input,
            instruction=instruction,
            enable_comment=enable_comment,
        )
        self.label_schemas.append(label_schema)
        _get_client().update_review_app(self, update_mask="labeling_schemas")
        return label_schema

    def delete_label_schema(self, label_schema_name: str) -> "ReviewApp":
        """Delete a label schema from the review app."""
        schema_to_remove = next(
            (label_schema for label_schema in self.label_schemas if label_schema.name == label_schema_name),
            None,
        )
        if not schema_to_remove:
            raise ValueError(f"Label schema {label_schema_name} not found.")
        self.label_schemas.remove(schema_to_remove)
        return _get_client().update_review_app(self, update_mask="labeling_schemas")

    def create_labeling_session(
        self,
        name: str,
        *,
        # Must be workspace users for now due to ACL
        assigned_users: list[str] = [],
        # agent names must already be added to the backend.
        agent: Optional[str] = None,
        # the schema names, must be already added to backend.
        label_schemas: list[str] = [],
        enable_multi_turn_chat: bool = False,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> LabelingSession:
        """Create a new labeling session in the review app.

        Args:
            name: The name of the labeling session.
            assigned_users: The users that will be assigned to label items in the session.
            agent: The agent to be used to generate responses for the items in the session.
            label_schemas: The label schemas to be used in the session.
            enable_multi_turn_chat: Whether to enable multi-turn chat labeling for the session.
            custom_inputs: Optional. Custom inputs to be used in the session.
        """
        valid_schema_names = set(s.name for s in self.label_schemas)
        for schema_name in label_schemas:
            if schema_name not in valid_schema_names:
                raise ValueError(
                    f"Label schema {schema_name} not found. " "Please create it first via `create_label_schema`."
                )

        # Give `assigned_users` access to the experiment and the model serving endpoint
        # (if agent is set).
        if assigned_users:
            add_users_to_experiment(assigned_users, self.experiment_id)

        if agent:
            agent_config = next((a for a in self.agents if a.agent_name == agent), None)
            if not agent_config:
                raise ValueError(
                    f"Agent {agent} not found in the review app. Please add the agent "
                    "first using `add_agent` method."
                )

            if agent_config.model_serving_endpoint and assigned_users:
                add_users_to_serving_endpoint(assigned_users, agent_config.model_serving_endpoint)

        return _get_client().create_labeling_session(
            self,
            name,
            assigned_users=assigned_users,
            agent=agent,
            label_schemas=label_schemas,
            enable_multi_turn_chat=enable_multi_turn_chat,
            custom_inputs=custom_inputs,
        )

    def get_labeling_sessions(self) -> list[LabelingSession]:
        """Get all labeling sessions in the review app."""
        return _get_client().list_labeling_sessions(self)

    def delete_labeling_session(self, labeling_session: LabelingSession) -> "ReviewApp":
        """Delete a labeling session from the review app."""
        return _get_client().delete_labeling_session(self.review_app_id, labeling_session.labeling_session_id)
