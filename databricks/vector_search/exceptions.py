class VectorSearchException(Exception):
    """
    Base exception for all Vector Search SDK errors.

    Attributes:
        status_code: HTTP status code if applicable
        response_content: Raw response content from the API
    """

    def __init__(self, message, status_code=None, response_content=None):
        super().__init__(message)
        self.status_code = status_code
        self.response_content = response_content


class InvalidInputException(VectorSearchException):
    pass


class NotFound(VectorSearchException):
    pass


# Alias for compatibility with databricks SDK naming convention
ResourceDoesNotExist = NotFound


class BadRequest(VectorSearchException):
    pass


class PermissionDenied(VectorSearchException):
    pass


class ResourceConflict(VectorSearchException):
    pass


class TooManyRequests(VectorSearchException):
    pass
