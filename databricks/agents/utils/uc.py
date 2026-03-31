from typing import Tuple

_INVALID_ENDPOINT_NAME_SUFFIX_CHARS = "_-"


def _check_model_name(model_name: str):
    """
    Check if the model name is in the correct UC format.

    :param model_name: The model name to check.
    """
    if len(model_name.split(".")) != 3:
        raise ValueError("Model name must be in the format catalog.schema.model_name")


def _get_catalog_and_schema(model_name: str) -> Tuple[str, str]:
    """
    Get the catalog and schema from the model name.

    :param model_name: The model name to extract the catalog and schema from.
    """
    _check_model_name(model_name)
    parts = model_name.split(".")
    return parts[0], parts[1]


def _sanitize_model_name(model_name: str) -> str:
    """
    Sanitize the UC model name for use in endpoint and served entity names`.

    :param model_name: The full uc model name to remove dots from.
    """
    return model_name.replace(".", "-").rstrip(_INVALID_ENDPOINT_NAME_SUFFIX_CHARS)


def _escape_uc_name(uc_name: str) -> str:
    """
    Escape each of catalog, schema, and UC entity name with backticks.
    """
    _check_model_name(uc_name)
    parts = uc_name.split(".")
    return ".".join([f"`{part}`" for part in parts])
