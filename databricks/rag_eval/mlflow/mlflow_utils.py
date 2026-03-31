"""Helper functions to convert RagEval entities to MLflow entities."""

import time
from typing import List, Union

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowException
from mlflow import entities as mlflow_entities
from mlflow.deployments import get_deployments_target
from mlflow.entities import metric as mlflow_metric
from mlflow.models import evaluation as mlflow_models_evaluation
from packaging.version import Version

from databricks.rag_eval import env_vars, schemas
from databricks.rag_eval.evaluation import datasets, entities
from databricks.rag_eval.utils import enum_utils, error_utils

_DEFAULT_MLFLOW_DEPLOYMENT_TARGET = "databricks"
_IS_MLFLOW_3 = Version(mlflow.__version__).major >= 3


class EvaluationErrorCode(enum_utils.StrEnum):
    MODEL_ERROR = "MODEL_ERROR"


def eval_result_to_mlflow_metrics(
    eval_result: entities.EvalResult,
) -> List[mlflow_metric.Metric]:
    """Get a list of MLflow Metric objects from an EvalResult object."""
    return [
        _construct_mlflow_metrics(
            key=k,
            value=v,
        )
        for k, v in eval_result.get_metrics_dict().items()
        # Do not log metrics with non-numeric-or-boolean values
        if isinstance(v, (int, float, bool))
    ]


def _construct_mlflow_metrics(key: str, value: Union[int, float, bool]) -> mlflow_metric.Metric:
    """
    Construct an MLflow Metric object from key and value.
    Timestamp is the current time and step is 0.
    """
    return mlflow_metric.Metric(
        key=key,
        value=value,
        timestamp=int(time.time() * 1000),
        step=0,
    )


def _get_mlflow_assessment_to_log_from_metric_results(
    metric_results: List[entities.MetricResult],
) -> List[mlflow_entities.Assessment]:
    """Get a list of MLflow Assessment objects from a list of MetricResult objects."""
    return [
        assessment
        for metric in metric_results
        if not metric.legacy_metric and (assessment := metric.to_mlflow_assessment())
    ]


def _cast_to_pandas_dataframe(data: Union[pd.DataFrame, np.ndarray], flatten: bool = True) -> pd.DataFrame:
    """
    Cast data to a pandas DataFrame. If already a pandas DataFrame, passes the data through.
    :param data: Data to cast to a pandas DataFrame
    :param flatten: Whether to flatten the data from 2d to 1d
    :return: A pandas DataFrame
    """
    if isinstance(data, pd.DataFrame):
        return data

    data = data.tolist()
    if flatten:
        data = [item for feature in data for item in feature]
    try:
        return pd.DataFrame(data)
    except Exception as e:
        raise error_utils.ValidationError(
            f"Data must be a DataFrame or a list of dictionaries. Got: {type(data[0])}"
        ) from e


def _validate_mlflow_dataset(ds: mlflow_models_evaluation.EvaluationDataset):
    """Validates an MLflow evaluation dataset."""
    features_df = _cast_to_pandas_dataframe(ds.features_data)

    # Validate max number of rows in the eval dataset
    if len(features_df) > env_vars.RAG_EVAL_MAX_INPUT_ROWS.get():
        raise error_utils.ValidationError(
            f"The number of rows in the dataset exceeds the maximum: {env_vars.RAG_EVAL_MAX_INPUT_ROWS.get()}. "
            f"Got {len(features_df)} rows." + error_utils.CONTACT_FOR_LIMIT_ERROR_SUFFIX
        )
    if ds.predictions_data is not None:
        # Predictions data is one-dimensional so it does not need to be flattened
        predictions_df = _cast_to_pandas_dataframe(ds.predictions_data, flatten=False)
        assert features_df.shape[0] == predictions_df.shape[0], (
            f"Features data and predictions must have the same number of rows. "
            f"Features: {features_df.shape[0]}, Predictions: {predictions_df.shape[0]}"
        )


def mlflow_dataset_to_evaluation_dataset(
    ds: mlflow_models_evaluation.EvaluationDataset,
) -> datasets.EvaluationDataframe:
    """Creates an instance of the class from an MLflow evaluation dataset and model predictions."""
    _validate_mlflow_dataset(ds)
    df = _cast_to_pandas_dataframe(ds.features_data).copy()
    if ds.predictions_data is not None:
        # Predictions data is one-dimensional so it does not need to be flattened
        df[schemas.RESPONSE_COL] = _cast_to_pandas_dataframe(ds.predictions_data, flatten=False)
    return datasets.EvaluationDataframe(df)


def resolve_deployments_target() -> str:
    """
    Resolve the current deployment target for MLflow deployments.

    If the deployment target is not set, use the default deployment target.

    We need this because user might set the deployment target explicitly using `set_deployments_target` to use
    endpoints from another workspace. We want to respect that config.
    """
    try:
        return get_deployments_target()
    except MlflowException:
        return _DEFAULT_MLFLOW_DEPLOYMENT_TARGET
