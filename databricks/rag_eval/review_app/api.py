"""Public API for the Review App."""

from typing import TYPE_CHECKING, Optional

import mlflow
from mlflow.tracking import fluent

from .entities import InputText, InputTextList, LabelSchema, ReviewApp
from .label_schemas import (
    EXPECTED_FACTS,
    EXPECTED_RESPONSE,
    GUIDELINES,
    LabelSchemaType,
)

if TYPE_CHECKING:
    from databricks.rag_eval.clients.managedevals.managed_evals_client import (
        ManagedEvalsClient,
    )

_FACTS_LEN_LIMIT = 1000


def _get_client() -> "ManagedEvalsClient":
    from databricks.rag_eval import context
    from databricks.rag_eval.clients.managedevals.managed_evals_client import (
        ManagedEvalsClient,  # noqa: F401
    )

    @context.eval_context
    def getter():
        return context.get_context().build_managed_evals_client()

    return getter()


def _create_review_app(client: "ManagedEvalsClient", experiment_id: str) -> ReviewApp:
    review_app = ReviewApp(
        review_app_id=None,  # Not defined at create time.
        experiment_id=experiment_id,
        url=None,  # Not defined at create time.
        # Always add the builtin schemas.
        label_schemas=[
            LabelSchema(
                name=EXPECTED_FACTS,
                type=LabelSchemaType.EXPECTATION,
                title="Expected facts",
                input=InputTextList(
                    max_length_each=_FACTS_LEN_LIMIT,
                ),
                instruction="Please provide a list of facts that you expect to see in a correct response.",
            ),
            LabelSchema(
                name=GUIDELINES,
                type=LabelSchemaType.EXPECTATION,
                title="Guidelines",
                input=InputTextList(max_length_each=_FACTS_LEN_LIMIT),
                instruction="Please provide guidelines that the model's output is expected to adhere to.",
            ),
            LabelSchema(
                name=EXPECTED_RESPONSE,
                type=LabelSchemaType.EXPECTATION,
                title="Expected response",
                input=InputText(),
                instruction="Please provide a correct agent response.",
            ),
        ],
    )
    return client.create_review_app(review_app)


def get_review_app(experiment_id: Optional[str] = None) -> ReviewApp:
    """Gets or creates (if it doesn't exist) the review app for the given experiment ID.

    Args:
        experiment_id: Optional. The experiment ID for which to get the review app. If not provided,
            the experiment ID is inferred from the current active environment.
    """
    if not experiment_id:
        # Infer the experiment ID from the current environment.
        experiment_id = fluent._get_experiment_id()
        if experiment_id is None or experiment_id == mlflow.tracking.default_experiment.DEFAULT_EXPERIMENT_ID:
            raise ValueError("Please provide an experiment_id or run this code within an active experiment.")
        # Make an actual experiment in case the experiment_id is the default notebook id.
        mlflow.set_experiment(experiment_id=experiment_id)

    client = _get_client()
    review_apps = client.list_review_apps(filter=f"experiment_id={experiment_id}")
    if not review_apps:
        # No review app exists for the given experiment ID. Create a new one.
        return _create_review_app(client, experiment_id)

    elif len(review_apps) > 1:
        raise ValueError(f"Multiple review apps found for experiment ID {experiment_id}:\n{review_apps}")
    else:
        return review_apps[0]
