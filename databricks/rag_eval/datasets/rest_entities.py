from dataclasses import dataclass
from typing import Any, Optional

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class Human:
    user_name: str


@dataclass_json
@dataclass
class Document:
    doc_uri: str
    content: Optional[str] = None


@dataclass_json
@dataclass
class Trace:
    trace_id: str


@dataclass_json
@dataclass
class Source:
    # One of the following fields must be set.
    human: Optional[Human] = None
    document: Optional[Document] = None
    trace: Optional[Trace] = None


@dataclass_json
@dataclass
class Input:
    key: str
    value: Any  # The backend uses protobuf.Value to support any json object.


@dataclass_json
@dataclass
class ExpectationValue:
    value: Any  # The backend uses protobuf.Value to support any json object.


@dataclass_json
@dataclass
class Dataset:
    dataset_id: Optional[str] = None
    digest: Optional[str] = None
    name: Optional[str] = None
    schema: Optional[str] = None
    profile: Optional[str] = None
    source: Optional[str] = None
    source_type: Optional[str] = None
    create_time: Optional[str] = None
    created_by: Optional[str] = None
    last_update_time: Optional[str] = None
    last_updated_by: Optional[str] = None


@dataclass_json
@dataclass
class DatasetRecord:
    inputs: list[Input]

    # Auto-generated fields.
    dataset_record_id: Optional[str] = None
    create_time: Optional[str] = None
    created_by: Optional[str] = None
    last_update_time: Optional[str] = None
    last_updated_by: Optional[str] = None

    # User-provided fields.
    source: Optional[Source] = None
    expectations: Optional[dict[str, ExpectationValue]] = None
    tags: Optional[dict[str, str]] = None

    def to_row_dict(self) -> dict:
        return {
            "dataset_record_id": self.dataset_record_id,
            "create_time": self.create_time,
            "created_by": self.created_by,
            "last_update_time": self.last_update_time,
            "last_updated_by": self.last_updated_by,
            "source": self.source,
            "inputs": {input.key: input.value for input in self.inputs},
            "expectations": {key: expectation.value for key, expectation in self.expectations.items()},
            "tags": self.tags,
        }
