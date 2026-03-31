from abc import ABC, abstractmethod
from typing import Dict, Any


class Reranker(ABC):
    """Abstract base class for all rerankers."""


class DatabricksReranker(Reranker):
    def __init__(self, columns_to_rerank: list[str]):
        """
        Initialize a DatabricksReranker config object.

        Args:
            columns_to_rerank: A list of column names to use for reranking the results.
        """
        self.columns_to_rerank = columns_to_rerank
