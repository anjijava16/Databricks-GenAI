# ruff: noqa: F811
import dataclasses
import json
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union

import mlflow
import pandas as pd
from dataclasses_json import dataclass_json
from mlflow.data import dataset_source
from typing_extensions import Self

from databricks.rag_eval import schemas
from databricks.rag_eval.clients.managedevals import dataset_utils
from databricks.rag_eval.evaluation.datasets import EvaluationDataframe
from databricks.rag_eval.utils import (
    NO_CHANGE,
    error_utils,
    spark_utils,
    token_counting_utils,
)
from databricks.rag_eval.utils.collection_utils import deep_getattr, deep_setattr
from databricks.rag_eval.utils.enum_utils import StrEnum

from . import rest_entities

_logger = logging.getLogger(__name__)

_FIELD_PATH = "FIELD_PATH"
_FIELD_IS_UPDATABLE = "FIELD_IS_UPDATABLE"
_FIELD_FROM_DICT = "FIELD_FROM_DICT"
_FIELD_TO_DICT = "FIELD_TO_DICT"


if TYPE_CHECKING:
    from databricks.rag_eval.clients.managedevals.managed_evals_client import (
        ManagedEvalsClient,
    )


def _get_client() -> "ManagedEvalsClient":
    from databricks.rag_eval import context
    from databricks.rag_eval.clients.managedevals.managed_evals_client import (
        ManagedEvalsClient,  # noqa: F401
    )

    @context.eval_context
    def getter():
        return context.get_context().build_managed_evals_client()

    return getter()


def _get_json_field_path(field: dataclasses.Field) -> Sequence[str]:
    """Get the JSON field path from the field metadata."""
    return field.metadata.get(_FIELD_PATH).split(".")


def _has_json_field_path(field: dataclasses.Field) -> bool:
    """Check if the field has a JSON field path."""
    return _FIELD_PATH in field.metadata


# The REST entity for `Dataset` and `Source` is the same as the public API entity.
Source = rest_entities.Source
Human = rest_entities.Human
Document = rest_entities.Document
Trace = rest_entities.Trace


# TODO: Expose this as a public API as `mlflow.entities.DatasetRow`.
@dataclass_json
@dataclasses.dataclass
class DatasetRow:
    dataset_record_id: str
    inputs: dict[str, str]
    expectations: Optional[dict[str, str]] = None
    source: Optional[Source] = None
    tags: Optional[dict[str, str]] = None

    # Auto-generated fields.
    create_time: Optional[str] = None
    last_update_time: Optional[str] = None
    created_by: Optional[str] = None
    last_updated_by: Optional[str] = None

    @classmethod
    def from_rest_dataset_record(cls, dataset_record: rest_entities.DatasetRecord) -> "DatasetRow":
        return cls(
            dataset_record_id=dataset_record.dataset_record_id,
            inputs={input.key: input.value for input in dataset_record.inputs},
            expectations={key: exp.value for key, exp in (dataset_record.expectations or {}).items()},
            tags=dataset_record.tags,
            source=dataset_record.source,
            create_time=dataset_record.create_time,
            last_update_time=dataset_record.last_update_time,
            created_by=dataset_record.created_by,
            last_updated_by=dataset_record.last_updated_by,
        )


@dataclass_json
@dataclasses.dataclass
class Dataset(mlflow.data.Dataset):
    """A dataset for storing evaluation records (inputs and expectations)."""

    dataset_id: str
    """The unique identifier of the dataset."""

    digest: Optional[str] = None
    """String digest (hash) of the dataset provided by the caller that uniquely identifies"""

    name: Optional[str] = None
    """The UC table name of the dataset."""

    schema: Optional[str] = None
    """The schema of the dataset. E.g., MLflow ColSpec JSON for a dataframe, MLflow TensorSpec JSON
    for an ndarray, or another schema format."""

    profile: Optional[str] = None
    """The profile of the dataset, summary statistics."""

    source: Optional[dataset_source.DatasetSource] = None
    """Source information for the dataset."""

    source_type: Optional[str] = None
    """The type of the dataset source, e.g. "databricks-uc-table", "DBFS", "S3", ..."""

    create_time: Optional[str] = None
    """The time the dataset was created."""

    created_by: Optional[str] = None
    """The user who created the dataset."""

    last_update_time: Optional[str] = None
    """The time the dataset was last updated."""

    last_updated_by: Optional[str] = None
    """The user who last updated the dataset."""

    def set_profile(self, profile: str) -> "Dataset":
        """Set the profile of the dataset."""
        self.profile = profile
        return _get_client().update_dataset(self, update_mask="profile")

    def insert(
        self,
        records: Union[list[Dict], pd.DataFrame, "pyspark.sql.DataFrame"],  # noqa: F821
    ) -> "Dataset":
        _logger.warning(
            "DeprecationWarning: `Dataset.insert` is deprecated and will be removed in a future version. "
            "Use `Dataset.merge_records` instead."
        )
        return self.merge_records(records)

    def merge_records(
        self,
        records: Union[list[Dict], pd.DataFrame, "pyspark.sql.DataFrame"],  # noqa: F821
    ) -> "Dataset":
        """Merge records into the dataset. Records that share the same inputs will be merged into a
        single record with the merged expectations and tags.

        Args:
            records: A list of dicts, a pandas DataFrame, or a Spark DataFrame. For the input schema
            see https://docs.databricks.com/en/generative-ai/agent-evaluation/evaluation-schema.html
        """

        df: pd.DataFrame

        if isinstance(records, list) and all(isinstance(r, dict) for r in records):
            df = pd.DataFrame.from_records(records)
        else:
            df = spark_utils.normalize_spark_df(records)

        _validate_new_records_schema(df)

        # Normalize old and new eval schema by converting the dataframe to eval items.
        # items first.
        input_eval_items = EvaluationDataframe(df).eval_items if not df.empty else []

        # Dedup input records against each other.
        input_records_map: dict[str, rest_entities.DatasetRecord] = {}
        for input_eval_item in input_eval_items:
            input_record = dataset_utils.eval_item_to_rest_dataset_record(input_eval_item)
            inputs_key = _convert_record_inputs_to_json_str(input_record)
            if inputs_key in input_records_map:
                # Merge expectations.
                existing_expectations = input_records_map[inputs_key].expectations or {}
                new_expectations = input_record.expectations or {}
                merged_expectations = {
                    **existing_expectations,
                    **new_expectations,
                }
                input_records_map[inputs_key].expectations = merged_expectations
            else:
                input_records_map[inputs_key] = input_record

        # Dedup input records against existing records.
        existing_records = _get_client().list_dataset_records(self.dataset_id)

        if existing_records and input_records_map:
            _validate_new_schema_against_existing(list(input_records_map.values()), existing_records)

        existing_record_map: dict[str, rest_entities.DatasetRecord] = {
            _convert_record_inputs_to_json_str(r): r for r in existing_records
        }
        new_records_to_create: list[rest_entities.DatasetRecord] = []
        records_to_update: list[rest_entities.DatasetRecord] = []
        for inputs_key, input_record in input_records_map.items():
            existing_record = existing_record_map.get(inputs_key)
            if existing_record:
                # Merge expectations.
                input_record.expectations = {
                    **(existing_record.expectations or {}),
                    **(input_record.expectations or {}),
                }
                # Merge tags.
                input_record.tags = {
                    **(existing_record.tags or {}),
                    **(input_record.tags or {}),
                }
                input_record.dataset_record_id = existing_record.dataset_record_id
                records_to_update.append(input_record)
            else:
                new_records_to_create.append(input_record)

        if new_records_to_create:
            _get_client().batch_create_dataset_records(self.name, self.dataset_id, new_records_to_create)

        # TODO: Remove the loop when the backend supports batch update.
        # TODO: Call this in parallel with a thread pool.
        for record in records_to_update:
            _get_client().update_dataset_record(self.dataset_id, record, "expectations,tags")

        if new_records_to_create or records_to_update:
            _get_client().sync_dataset_to_uc(self.dataset_id, self.name)

        return self

    def to_df(self) -> pd.DataFrame:
        """Convert the dataset to a pandas DataFrame."""
        dataset_rows = [
            DatasetRow.from_rest_dataset_record(record)
            for record in _get_client().list_dataset_records(self.dataset_id)
        ]
        return pd.DataFrame.from_records([row.to_dict() for row in dataset_rows])


# noinspection PyTypeChecker,PyArgumentList
@dataclasses.dataclass
class JsonSerializable:
    def to_json(self) -> dict:
        """Convert the object to a JSON serializable dictionary."""
        json = {}
        for field in dataclasses.fields(self):
            if _has_json_field_path(field):
                value = getattr(self, field.name)
                if value is not None and value is not NO_CHANGE:
                    if _FIELD_TO_DICT in field.metadata:
                        to_dict_fn = field.metadata[_FIELD_TO_DICT]
                        value = to_dict_fn(value)
                    deep_setattr(json, _get_json_field_path(field), value)
        return json

    @classmethod
    def from_json(cls, json: dict) -> Self:
        """Create an instance from a JSON dictionary."""
        values = {}
        for field in dataclasses.fields(cls):
            if _has_json_field_path(field):
                raw_value = deep_getattr(json, _get_json_field_path(field))
                if raw_value is not None:
                    if _FIELD_FROM_DICT in field.metadata:
                        from_dict_fn = field.metadata[_FIELD_FROM_DICT]
                        value = from_dict_fn(raw_value)
                    else:
                        value = raw_value
                    values[field.name] = value
        return cls(**values)

    def get_update_mask(self) -> str:
        """Get the update mask for the fields that have changed."""
        return ",".join(
            field.metadata.get(_FIELD_PATH)
            for field in dataclasses.fields(self)
            if getattr(self, field.name) is not NO_CHANGE and field.metadata.get(_FIELD_IS_UPDATABLE, True)
        )


@dataclasses.dataclass
class UIPageConfig:
    name: str
    display_name: str
    path: str


@dataclasses.dataclass
class EvalsInstance(JsonSerializable):
    instance_id: Optional[str] = dataclasses.field(
        default=None, metadata={_FIELD_PATH: "instance_id", _FIELD_IS_UPDATABLE: False}
    )
    version: Optional[str] = dataclasses.field(
        default=None, metadata={_FIELD_PATH: "version", _FIELD_IS_UPDATABLE: False}
    )
    agent_name: Optional[str] = dataclasses.field(default=None, metadata={_FIELD_PATH: "agent_config.agent_name"})
    agent_serving_endpoint: Optional[str] = dataclasses.field(
        default=None,
        metadata={_FIELD_PATH: "agent_config.model_serving_config.model_serving_endpoint_name"},
    )
    ui_page_configs: Optional[List[UIPageConfig]] = dataclasses.field(
        default=None,
        metadata={
            _FIELD_PATH: "ui_page_configs",
            _FIELD_IS_UPDATABLE: False,
            _FIELD_FROM_DICT: lambda value: [UIPageConfig(**page) for page in value],
            _FIELD_TO_DICT: lambda value: [dataclasses.asdict(page) for page in value],
        },
    )
    experiment_ids: List[str] = dataclasses.field(
        default_factory=list,
        metadata={
            _FIELD_PATH: "experiment_ids",
        },
    )


@dataclasses.dataclass
class EvalTag(JsonSerializable):
    """A tag on an eval row."""

    tag_id: str = dataclasses.field(metadata={_FIELD_PATH: "tag_id"})
    eval_id: str = dataclasses.field(metadata={_FIELD_PATH: "eval_id"})


@dataclasses.dataclass
class Document:
    """A document that holds the source data for an agent application."""

    content: str
    """The raw content of the document."""

    doc_uri: str
    """The URI of the document."""

    num_tokens: Optional[int] = None
    """The number of tokens in the document."""

    def __post_init__(self):
        if not self.content or not isinstance(self.content, str):
            raise error_utils.ValidationError(
                f"'content' of a document must be a non-empty string. Got: {self.content}"
            )

        if not self.doc_uri or not isinstance(self.doc_uri, str):
            raise error_utils.ValidationError(
                f"'doc_uri' of a document must be a non-empty string. Got: {self.doc_uri}"
            )

        if self.num_tokens is None:
            self.num_tokens = token_counting_utils.count_tokens(self.content)


@dataclasses.dataclass
class SyntheticQuestion:
    """A synthetic question generated by the synthetic API that can be used for evaluation."""

    question: str
    """The raw question text."""

    source_doc_uri: str
    """The URI of the document from which the question was generated."""

    source_context: str
    """
    The context from which the question was generated. 
    Could be a chunk of text from the source document or the whole document content.
    """


@dataclasses.dataclass
class SyntheticAnswer:
    """A synthetic answer generated by the synthetic API that can be used for evaluation."""

    question: SyntheticQuestion
    """The synthetic question to which the answer corresponds."""

    synthetic_ground_truth: Optional[str] = None
    """The synthetic ground truth answer for the question."""

    synthetic_grading_notes: Optional[str] = None
    """The synthetic grading notes to help judge the correctness of the question."""

    synthetic_minimal_facts: Optional[List[str]] = None
    """The synthetic minimum expected facts required to answer the question."""


class SyntheticAnswerType(StrEnum):
    GROUND_TRUTH = "GROUND_TRUTH"
    GRADING_NOTES = "GRADING_NOTES"
    MINIMAL_FACTS = "MINIMAL_FACTS"


class SchemaType(StrEnum):
    INPUTS = "inputs"
    REQUEST = "request"
    MULTITURN = "multiturn"


def _deep_convert_int_to_float(data):
    """google.protobuf.Value implicitly converts int to float, so we need to do the same."""
    if isinstance(data, dict):
        return {k: _deep_convert_int_to_float(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple, set)):
        return type(data)(_deep_convert_int_to_float(x) for x in data)
    elif isinstance(data, int):
        return float(data)
    return data


def _convert_record_inputs_to_json_str(
    record: rest_entities.DatasetRecord,
) -> str:
    res: dict[str, Any] = {}
    for input_pair in record.inputs:
        res[input_pair.key] = input_pair.value
    protobuf_dict = _deep_convert_int_to_float(res)
    return json.dumps(protobuf_dict, sort_keys=True)


def _validate_new_records_schema(df: pd.DataFrame) -> None:
    """
    Validate schema of new records before transformation.

    Checks:
    1. When using multiturn columns, only known schema columns are allowed
    2. Can't mix multiturn fields with custom fields within the 'inputs' column
    3. Across records: all records must consistently use multiturn or custom

    Args:
        df: The DataFrame with new records to validate.

    Raises:
        ValueError: If schemas are inconsistent.
    """
    if df.empty:
        return

    df_cols = set(df.columns)
    multiturn_cols_present = df_cols & schemas.MULTITURN_IDENTIFIER_COLS

    if multiturn_cols_present:
        known_col_names = {col.name for col in EvaluationDataframe.COLS} | {schemas.EXPECTATIONS_COL}
        unknown_cols = df_cols - schemas.MULTITURN_INPUT_COLS_SET - known_col_names
        if unknown_cols:
            raise ValueError(
                f"When using multiturn input fields ({schemas.MULTITURN_COLS_STR}), "
                f"only multiturn fields are allowed as input fields. "
                f"Found unknown columns: {unknown_cols}. "
                f"Either remove these columns or put them in the '{schemas.CONTEXT_COL}' field."
            )

    _validate_inputs_column(df)


def _validate_inputs_column(df: pd.DataFrame) -> None:
    """
    Validate the 'inputs' column for consistent schema across all records.

    Checks:
    1. If using multiturn fields, can't mix with custom fields in the same record
    2. All records must consistently use either multiturn or custom fields

    Args:
        df: The DataFrame with an 'inputs' column to validate.

    Raises:
        ValueError: If schemas are inconsistent within or across records.
    """
    multiturn_rows, custom_rows = [], []
    for idx, row in df.iterrows():
        inputs_value = row[schemas.INPUTS_COL]
        if pd.isna(inputs_value) or not isinstance(inputs_value, dict) or not inputs_value:
            continue
        input_keys = set(inputs_value.keys())
        has_multiturn = bool(input_keys & schemas.MULTITURN_IDENTIFIER_COLS)
        if has_multiturn:
            non_multiturn_keys = input_keys - schemas.MULTITURN_INPUT_COLS_SET
            if non_multiturn_keys:
                raise ValueError(
                    f"When using multiturn input fields ({schemas.MULTITURN_COLS_STR}) "
                    f"in the '{schemas.INPUTS_COL}' column, only multiturn fields are allowed. "
                    f"Found invalid fields in row {idx}: {non_multiturn_keys}. "
                    f"You can put them in the {schemas.CONTEXT_COL} field instead."
                )
            multiturn_rows.append(idx)
        else:
            custom_rows.append(idx)
    if multiturn_rows and custom_rows:
        raise ValueError(
            f"Inconsistent input schemas: Some records use multiturn fields "
            f"({schemas.MULTITURN_COLS_STR}) while others use custom fields. "
            f"Row {multiturn_rows[0]} uses multiturn fields, row {custom_rows[0]} uses custom fields. "
            f"All records must use the same schema."
        )


def _validate_new_schema_against_existing(
    new_records: list[rest_entities.DatasetRecord],
    existing_records: list[rest_entities.DatasetRecord],
) -> None:
    """
    Validate that new records' schema matches existing records' schema.

    Checks:
    1. 'request' vs 'inputs' schema
    2. Multiturn vs custom input fields

    Args:
        new_records: The new dataset records to validate.
        existing_records: The existing dataset records to validate against.

    Raises:
        ValueError: If schemas don't match.
    """
    if not existing_records or not new_records:
        return

    new_schema = _get_schema_from_records(new_records)
    existing_schema = _get_schema_from_records(existing_records)

    if existing_schema == new_schema:
        return

    if existing_schema == SchemaType.REQUEST or new_schema == SchemaType.REQUEST:
        raise ValueError(
            f"Schema mismatch: Dataset uses '{existing_schema}' schema, "
            f"but new records use '{new_schema}' schema. "
            f"Only one schema type is allowed per dataset."
        )

    if existing_schema == SchemaType.MULTITURN:
        raise ValueError(
            f"Schema mismatch: Dataset uses multiturn input fields ({schemas.MULTITURN_COLS_STR}), "
            f"but new records have none. Include at least one multiturn field."
        )
    else:
        raise ValueError(
            f"Schema mismatch: Dataset does not use multiturn fields, "
            f"but new records use multiturn fields ({schemas.MULTITURN_COLS_STR})."
        )


def _get_schema_from_records(records: list[rest_entities.DatasetRecord]) -> SchemaType:
    """
    Get the schema type from existing dataset records.
    Only checks the first record since validations have been done at this point.

    Args:
        records: The dataset records.

    Returns:
        The schema type of the records.
    """
    if not records:
        return SchemaType.INPUTS

    first_record = records[0]
    if len(first_record.inputs) == 1 and first_record.inputs[0].key == schemas.REQUEST_COL:
        return SchemaType.REQUEST

    input_keys = {input.key for input in first_record.inputs}
    if input_keys & schemas.MULTITURN_IDENTIFIER_COLS:
        return SchemaType.MULTITURN

    return SchemaType.INPUTS
