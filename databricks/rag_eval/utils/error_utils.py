"""Utilities for errors."""

CONTACT_FOR_LIMIT_ERROR_SUFFIX = (
    "If you need to change this limit, please reach out to agent-evaluation-public@databricks.com"
)


class ValidationError(Exception):
    """Error class for all user-facing validation errors."""

    pass
