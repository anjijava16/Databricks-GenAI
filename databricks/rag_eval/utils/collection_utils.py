"""Utilities for manipulating collections."""

import json
from typing import Any, Dict, Iterator, List, Mapping, Sequence, TypeVar, Union

import numpy as np


def omit_keys(d: Mapping[str, Any], keys: List[str]) -> Dict[str, Any]:
    """
    Omit keys from a dictionary.
    :param d:
    :param keys:
    :return: A new dictionary with the keys removed.
    """
    return {k: v for k, v in d.items() if k not in keys}


def position_map_to_list(d: Mapping[int, Any], default: Any = None) -> List[Any]:
    """
    Convert a position map to a list ordered by position. Missing positions are filled with the default value.
    Position starts from 0.

    e.g. {0: 'a', 1: 'b', 3: 'c'} -> ['a', 'b', default, 'c']

    :param d: A position map.
    :param default: The default value to fill missing positions.
    :return: A list of values in the map.
    """
    length = max(d.keys(), default=-1) + 1
    return [d.get(i, default) for i in range(length)]


def deep_update(original, updates):
    """
    Recursively update a dictionary with another dictionary.

    Example:
        deep_update({a: {b: 1}}, {a: {c: 2}})
        # Result: {a: {b: 1, c: 2}}

    :param original: The original dictionary to update.
    :param updates: The dictionary with updates.
    :return: A new dictionary with the updates applied.
    """
    # Make a copy of the original dictionary to avoid modifying it in place.
    result = original.copy()

    for key, value in updates.items():
        if isinstance(value, Mapping) and key in result:
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value

    return result


def drop_none_values(d: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Drop None values from a dictionary.

    :param d: The dictionary.
    :return: A new dictionary with the None values removed.
    """
    return {k: v for k, v in d.items() if v is not None}


def deep_setattr(d: dict, key_path: Sequence[str], value: Any) -> None:
    """
    Set a value in a nested dictionary using a key path.

    Args:
        d: The dictionary to update.
        key_path: A list of keys to traverse the nested dictionary.
        value: The value to set.

    Example:
        deep_setattr({}, ['a', 'b', 'c'], 1)
        # Result: {'a': {'b': {'c': 1}}}
    """
    for key in key_path[:-1]:
        if key not in d:
            d[key] = {}
        d = d[key]
    d[key_path[-1]] = value


def deep_getattr(d: dict, key_path: Sequence[str]) -> Any:
    """
    Get a value from a nested dictionary using a key path.

    Args:
        d: The dictionary to get the value from.
        key_path: A list of keys to traverse the nested dictionary.

    Returns:
        The value at the key path or None if the key path does not exist.

    Example:
        deep_getattr({'a': {'b': {'c': 1}}}, ['a', 'b', 'c'])
        # Result: 1
    """
    for key in key_path:
        if key in d:
            d = d[key]
        else:
            return None
    return d


def convert_ndarray_to_list(data):
    """
    Recursively converts all numpy.ndarray objects in a dictionary (or any nested structure)
    to Python lists.

    Args:
        data: The input data (dictionary, list, or any nested structure).

    Returns:
        A new data structure with numpy.ndarray objects converted to Python lists.
    """
    if isinstance(data, dict):
        return {key: convert_ndarray_to_list(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_ndarray_to_list(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(convert_ndarray_to_list(item) for item in data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    else:
        return data


def safe_batch(items: List[Any], batch_byte_limit: int, batch_quantity_limit: int) -> Iterator[List[Any]]:
    """Split up items into batches, with each batch meeting size limit when JSON-serialized."""
    batch = []
    batch_size = 0
    for item in items:
        item_size = len(json.dumps(item))
        if item_size > batch_byte_limit:
            raise ValueError(
                f"Item {repr(item)[:200]}... ({item_size} bytes) exceeds Databricks HTTP request size limit ({batch_byte_limit} bytes)."
            )
        if batch_size + item_size > batch_byte_limit or len(batch) >= batch_quantity_limit:
            yield batch
            batch = []
            batch_size = 0
        batch.append(item)
        batch_size += item_size
    if batch:
        yield batch


T = TypeVar("T")


def to_list(value: Union[T, List[T]]) -> List[T]:
    """Convert a value to a list if it is not already a list."""
    return value if isinstance(value, list) else [value]
