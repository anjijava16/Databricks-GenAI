"""Databricks Agent Datasets Python SDK.

WARNING: This API is deprecated. Please use the new Datasets API in `MLflow 3 <https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/build-eval-dataset>`_.
"""

from databricks.rag_eval.datasets.api import create_dataset, delete_dataset, get_dataset
from databricks.rag_eval.datasets.entities import Dataset

__all__ = [
    "create_dataset",
    "Dataset",
    "delete_dataset",
    "get_dataset",
]
