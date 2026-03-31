"""REST API entities for the review app."""

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Union

from dataclasses_json import dataclass_json

from databricks.rag_eval.utils.enum_utils import StrEnum

if TYPE_CHECKING:
    from databricks.rag_eval.review_app import entities


class InputType:
    pass


@dataclass_json
@dataclass
class InputCategorical(InputType):  # Dropdown
    """A single-select dropdown for collecting assessments from stakeholders."""

    options: list[str]


@dataclass_json
@dataclass
class InputCategoricalList(InputType):  # Multi-select
    """A multi-select dropdown for collecting assessments from stakeholders."""

    options: list[str]


@dataclass_json
@dataclass
class InputTextList(InputType):  # List of free-form text
    """Like `Text`, but allows multiple entries."""

    max_length_each: Optional[int] = None
    max_count: Optional[int] = None


@dataclass_json
@dataclass
class InputText(InputType):  # Single free-form text.
    """A free-form text box for collecting assessments from stakeholders."""

    max_length: Optional[int] = None


@dataclass_json
@dataclass
class InputNumeric(InputType):  # Numeric input.
    """A numeric input for collecting assessments from stakeholders."""

    min_value: Optional[float] = None
    max_value: Optional[float] = None


class LabelingSchemaType(StrEnum):
    TYPE_UNSPECIFIED = "TYPE_UNSPECIFIED"
    FEEDBACK = "FEEDBACK"
    EXPECTATION = "EXPECTATION"


@dataclass_json
@dataclass
class LabelingSchema:
    name: Optional[str]
    type: LabelingSchemaType
    title: Optional[str]
    instruction: Optional[str]
    enable_comment: Optional[bool]

    categorical: Optional[InputCategorical] = None
    categorical_list: Optional[InputCategoricalList] = None
    text: Optional[InputText] = None
    text_list: Optional[InputTextList] = None
    numeric: Optional[InputNumeric] = None


@dataclass_json
@dataclass
class ModelServingEndpoint:
    endpoint_name: Optional[str]
    served_entity_name: Optional[str] = None


@dataclass_json
@dataclass
class Agent:
    agent_name: Optional[str]
    model_serving_endpoint: Optional[ModelServingEndpoint]


def _extract_input_type_from_rest_label_schema(
    label_schema: LabelingSchema,
) -> Union[
    InputCategorical,
    InputCategoricalList,
    InputText,
    InputTextList,
    InputNumeric,
]:
    if label_schema.categorical:
        return InputCategorical(
            options=label_schema.categorical.options,
        )
    elif label_schema.categorical_list:
        return InputCategoricalList(
            options=label_schema.categorical_list.options,
        )
    elif label_schema.text:
        return InputText(
            max_length=label_schema.text.max_length,
        )
    elif label_schema.text_list:
        return InputTextList(
            max_length_each=label_schema.text_list.max_length_each,
            max_count=label_schema.text_list.max_count,
        )
    elif label_schema.numeric:
        return InputNumeric(
            min_value=label_schema.numeric.min_value,
            max_value=label_schema.numeric.max_value,
        )


@dataclass_json
@dataclass
class ReviewApp:
    review_app_id: Optional[str]
    experiment_id: Optional[str]
    agents: list[Agent] = field(default_factory=list)
    labeling_schemas: list[LabelingSchema] = field(default_factory=list)

    @classmethod
    def from_review_app(cls, review_app: "entities.ReviewApp") -> "ReviewApp":
        rest_agents = [
            Agent(
                agent_name=agent.agent_name,
                model_serving_endpoint=ModelServingEndpoint(endpoint_name=agent.model_serving_endpoint),
            )
            for agent in review_app.agents
        ]
        rest_schemas = [
            LabelingSchema(
                name=s.name,
                type=(LabelingSchemaType.FEEDBACK if s.type == "feedback" else LabelingSchemaType.EXPECTATION),
                title=s.title,
                instruction=s.instruction,
                enable_comment=s.enable_comment,
                categorical=(s.input if type(s.input).__name__ == "InputCategorical" else None),
                categorical_list=(s.input if type(s.input).__name__ == "InputCategoricalList" else None),
                text=s.input if type(s.input).__name__ == "InputText" else None,
                text_list=(s.input if type(s.input).__name__ == "InputTextList" else None),
                numeric=s.input if type(s.input).__name__ == "InputNumeric" else None,
            )
            for s in review_app.label_schemas
        ]
        return ReviewApp(
            review_app_id=review_app.review_app_id,
            experiment_id=review_app.experiment_id,
            agents=rest_agents,
            labeling_schemas=rest_schemas,
        )

    def to_review_app(self) -> "entities.ReviewApp":
        """Converts the REST API response to a Python Review App object."""
        from databricks.agents.utils.mlflow_utils import get_workspace_url
        from databricks.rag_eval.review_app import entities

        agents = [
            entities.Agent(
                agent_name=agent.agent_name,
                model_serving_endpoint=agent.model_serving_endpoint.endpoint_name,
            )
            for agent in self.agents
        ]
        schemas = [
            entities.LabelSchema(
                name=schema.name,
                type=("feedback" if schema.type == LabelingSchemaType.FEEDBACK else "expectation"),
                title=schema.title,
                instruction=schema.instruction,
                enable_comment=schema.enable_comment,
                input=_extract_input_type_from_rest_label_schema(schema),
            )
            for schema in self.labeling_schemas
        ]
        return entities.ReviewApp(
            review_app_id=self.review_app_id,
            experiment_id=self.experiment_id,
            agents=agents,
            label_schemas=schemas,
            # TODO: Propagete the url from the REST entity once the backend returns it.
            url=f"{get_workspace_url()}/ml/review-v2/{self.review_app_id}",
        )


@dataclass_json
@dataclass
class AgentRef:
    agent_name: Optional[str]


@dataclass_json
@dataclass
class LabelingSchemaRef:
    name: Optional[str]


@dataclass_json
@dataclass
class AdditionalConfigs:
    disable_multi_turn_chat: Optional[bool] = None
    custom_inputs_json: Optional[str] = None


@dataclass_json
@dataclass
class LabelingSession:
    labeling_session_id: Optional[str]
    mlflow_run_id: Optional[str]

    name: Optional[str]
    assigned_users: list[str] = field(default_factory=list)
    agent: Optional[AgentRef] = None
    labeling_schemas: list[LabelingSchemaRef] = field(default_factory=list)
    additional_configs: Optional[AdditionalConfigs] = None

    def to_labeling_session(
        self, review_app_id: str, experiment_id: str, session_url: str
    ) -> "entities.LabelingSession":
        from databricks.rag_eval.review_app import entities

        enable_multi_turn_chat = False
        custom_inputs = None

        if self.additional_configs:
            if not self.additional_configs.disable_multi_turn_chat:
                enable_multi_turn_chat = True

            if self.additional_configs.custom_inputs_json:
                custom_inputs = json.loads(self.additional_configs.custom_inputs_json)

        return entities.LabelingSession(
            name=self.name,
            assigned_users=self.assigned_users,
            agent=self.agent.agent_name if self.agent else None,
            label_schemas=[schema_ref.name for schema_ref in self.labeling_schemas],
            labeling_session_id=self.labeling_session_id,
            mlflow_run_id=self.mlflow_run_id,
            review_app_id=review_app_id,
            experiment_id=experiment_id,
            url=session_url,
            enable_multi_turn_chat=enable_multi_turn_chat,
            custom_inputs=custom_inputs,
        )


@dataclass_json
@dataclass
class DatasetRecordRef:
    dataset_id: Optional[str] = None
    dataset_record_id: Optional[str] = None


@dataclass_json
@dataclass
class Source:
    dataset_record: Optional[DatasetRecordRef] = None
    trace_id: Optional[str] = None


class State(StrEnum):
    STATE_UNSPECIFIED = "STATE_UNSPECIFIED"
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    SKIPPED = "SKIPPED"


@dataclass_json
@dataclass
class ChatRound:
    trace_id: Optional[str] = None
    dataset_record: Optional[DatasetRecordRef] = None


@dataclass_json
@dataclass
class Item:
    # Auto-generated.
    item_id: Optional[str] = None
    create_time: Optional[str] = None
    created_by: Optional[str] = None
    last_update_time: Optional[str] = None
    last_updated_by: Optional[str] = None

    source: Optional[Source] = None
    state: Optional[str] = State.PENDING

    chat_rounds: list[ChatRound] = field(default_factory=list)
