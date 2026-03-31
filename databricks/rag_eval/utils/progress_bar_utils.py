import threading

from tqdm.auto import tqdm


class ThreadSafeProgressManager:
    """
    A thread-safe progress manager that can be used to update a progress bar from multiple threads.
    """

    def __init__(self, total, **kwargs):
        self._total = total
        self._lock = threading.RLock()
        self._pbar = None
        self._kwargs = kwargs

    def __enter__(self):
        self._pbar = tqdm(total=self._total, **self._kwargs)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._pbar.close()

    def update(self, n=1):
        with self._lock:
            self._pbar.update(n)
