"""Helper functions for counting tokens."""

import functools
import logging
import os
from typing import Optional

import tiktoken

_DEFAULT_ENCODING = "o200k_base"

_logger = logging.getLogger(__name__)


def count_tokens(text: Optional[str], encoding: Optional[str] = None) -> int:
    """
    Count the number of tokens in the input text.

    Allowing all special tokens so that it is resilient to any input text.

    :param text: Input text
    :param encoding: Optional encoding to use for tokenization; defaults to "o200k_base"
    :return: Number of tokens
    """
    if not text:
        return 0
    try:
        encoder = _cached_encoder(encoding)
        return len(encoder.encode(text, allowed_special="all"))
    except Exception as e:
        _logger.warning(f"Failed to count tokens for text: {text}. Error: {e}")
        return 0


@functools.lru_cache(maxsize=8)
def _cached_encoder(encoding: Optional[str] = None) -> tiktoken.Encoding:
    """
    Load the tiktoken encoder and cache it.
    """
    # ref: https://github.com/openai/tiktoken/issues/75
    os.environ["TIKTOKEN_CACHE_DIR"] = ""
    return tiktoken.get_encoding(encoding or _DEFAULT_ENCODING)
