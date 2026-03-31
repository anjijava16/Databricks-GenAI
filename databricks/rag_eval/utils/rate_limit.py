import abc
import dataclasses
import functools
import threading
import time
from contextlib import contextmanager


@dataclasses.dataclass(frozen=True)
class RateLimitConfig:
    """
    Rate limit configuration for the RAG evaluation harness.
    """

    quota: float
    """
    The number of tasks allowed to run in the time window.
    """

    time_window_in_seconds: float
    """
    The time window in seconds.
    """


class RateLimiter(abc.ABC):
    """
    Rate limiter to limit the rate of function calls or code block execution.

    Example usage as a decorator:
    ```
    limiter = RateLimiter(quota=4, time_window_in_seconds=1)

    @limiter
    def task(args):
        ...

    # Call `task` as needed, and the rate limiter will ensure that no more than 4 calls are made per second.
    ```

    Example usage as a context manager:
    ```
    limiter = RateLimiter(quota=4, time_window_in_seconds=1)

    with limiter:
      # operations that need to be rate limited

    # Rate limiter will ensure that no more than 4 executions of the code block are made per second.
    ```

    Example usage by wrapping a function:
    ```
    limiter = RateLimiter(quota=4, time_window_in_seconds=1)
    limited_task = limiter(task)

    # Call `limited_task` as needed, and the rate limiter will ensure that no more than 4 calls are made per second.
    ```
    """

    @abc.abstractmethod
    def __call__(self, fn):
        """Wrapper for the function to rate limit."""

    @abc.abstractmethod
    def __enter__(self):
        """Define the logic to be executed when entering the rate limit context."""

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Define the logic to be executed when exiting the rate limit context."""
        pass

    @classmethod
    def build(cls, **kwargs):
        """Factory method to build a rate limiter. Use TokenBucketRateLimiter by default."""
        return TokenBucketRateLimiter(**kwargs)

    @classmethod
    def no_op(cls):
        """Factory method to build a no-op rate limiter."""
        return NoOpRateLimiter()

    @classmethod
    def build_from_config(cls, config: RateLimitConfig):
        """Factory method to build a rate limiter from a RateLimitConfig."""
        return TokenBucketRateLimiter(quota=config.quota, time_window_in_seconds=config.time_window_in_seconds)


class TokenBucketRateLimiter(RateLimiter):
    """
    Rate limiter that uses a token bucket algorithm.

    :param quota: The maximum number of calls allowed per `time_window_in_seconds`.
    :param time_window_in_seconds: The number of seconds to replenish the tokens. Can be less than a second.
    """

    def __init__(self, quota: float, time_window_in_seconds: float):
        # Validate input
        if quota < 1:
            raise ValueError("quota must be >= 1.")
        if time_window_in_seconds <= 0:
            raise ValueError("time_window_in_seconds must be > 0.")

        self._capacity: float = quota
        self._tokens: float = 0
        self._replenish_seconds: float = time_window_in_seconds
        self._refill_per_second: float = quota / time_window_in_seconds
        self._last_update: float = time.monotonic()
        self._lock = threading.RLock()

        # Counter for the number of threads waiting, mainly used to expose states for testing
        self._waiting_thread_counter = 0
        # Lock for the waiting thread counter to make incr and decr atomic
        self._lock_for_waiting_thread_counter = threading.Lock()

    def __call__(self, fn):
        """Wrap a function with this rate limiter."""

        @functools.wraps(fn)
        def wrapper(*args, **kw):
            self._wait()
            return fn(*args, **kw)

        return wrapper

    def __enter__(self):
        self._wait()

    def _can_go(self) -> bool:
        """
        Check if a call can be made now.
        If so, decrement the token count and return True. Otherwise, return False.
        """
        with self._lock:
            if 1 <= self._refresh_and_get_token():
                self._tokens -= 1
                return True
            return False

    def _expected_wait(self) -> float:
        """Return the expected wait time in seconds before a call can be made."""
        with self._lock:
            tokens = self._refresh_and_get_token()
            if tokens >= 1:
                return 0
            expected_wait = (1 - tokens) / self._refill_per_second
            return expected_wait

    def _wait(self) -> None:
        """Wait until a call can be made."""
        with self._count_waiting_thread():
            while not self._can_go():
                time.sleep(self._expected_wait())

    def _refresh_and_get_token(self) -> float:
        """Refresh the token bucket and return the current number of tokens available."""
        with self._lock:
            now = time.monotonic()

            delta = self._refill_per_second * (now - self._last_update)
            self._tokens = min(self._capacity, self._tokens + delta)

            self._last_update = now

        return self._tokens

    @contextmanager
    def _count_waiting_thread(self):
        self._incr_waiting_thread_counter()
        try:
            yield
        finally:
            self._decr_waiting_thread_counter()

    def _incr_waiting_thread_counter(self):
        with self._lock_for_waiting_thread_counter:
            self._waiting_thread_counter += 1

    def _decr_waiting_thread_counter(self):
        with self._lock_for_waiting_thread_counter:
            self._waiting_thread_counter -= 1

    @property
    def num_waiting_threads(self):
        """Return the number of threads waiting for the rate limiter. Used for testing."""
        return self._waiting_thread_counter


class NoOpRateLimiter(RateLimiter):
    """A no-op rate limiter that does not rate limit calls."""

    _instance = None

    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super(NoOpRateLimiter, cls).__new__(cls)
        return cls._instance

    def __call__(self, fn):
        """No-op wrapper."""
        return fn

    def __enter__(self):
        """No-op context manager."""
        pass
