import logging
from typing import Callable, Dict, List, Union

import numpy as np

_logger = logging.getLogger(__name__)

_AGGREGATION_TO_AGGREGATE_FUNCTION = {
    "min": np.min,
    "max": np.max,
    "mean": np.mean,
    "median": np.median,
    "variance": np.var,
    "p90": lambda x: np.percentile(x, 90) if x else None,
}


def get_aggregate_results(scores: List[float], aggregations: List[Union[str, Callable]]) -> Dict[str, float]:
    """Compute aggregate statistics for a list of scores based on specified aggregations.

    Args:
        scores: List of numeric scores to aggregate
        aggregations: List of aggregation types to compute (e.g. ["min", "max", "mean"])

    Returns:
        Dictionary mapping aggregation names to computed values
    """
    scores_for_aggregation = [score for score in scores if score is not None]
    if not scores_for_aggregation:
        return {}

    results = {}
    for aggregation in aggregations:
        if isinstance(aggregation, str):
            if aggregation not in _AGGREGATION_TO_AGGREGATE_FUNCTION:
                raise ValueError(f"Invalid aggregation: {aggregation}")
            results[aggregation] = _AGGREGATION_TO_AGGREGATE_FUNCTION[aggregation](scores_for_aggregation)
        else:
            try:
                results[aggregation.__name__] = aggregation(scores_for_aggregation)
            except Exception as e:
                _logger.error(f"Error computing aggregation {aggregation} due to: {e}")
                continue

    return results
