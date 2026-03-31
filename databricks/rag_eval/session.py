"""Session for Agent Evaluation."""

import dataclasses
import threading
from typing import List, Optional


@dataclasses.dataclass
class Session:
    """
    Agent evaluation session.

    This class serves as the per-run storage for the current evaluation run.
    """

    session_id: str
    """The session ID for the current evaluation."""
    session_batch_size: Optional[int] = None
    """The number of questions to evaluate in the current evaluation."""
    synthetic_generation_num_docs: Optional[int] = None
    """The number of documents used for this synthetic generation session."""
    synthetic_generation_num_evals: Optional[int] = None
    """The number of evaluations to generate in total in this synthetic generation session."""
    monitoring_wheel_version: Optional[str] = None
    """The internal version of the monitoring wheel used for this evaluation."""

    warnings: List[str] = dataclasses.field(default_factory=list)
    """The list of warning messages raised from the execution of the current evaluation."""

    def set_session_batch_size(self, batch_size: int) -> None:
        """Sets the batch size for the current evaluation."""
        self.session_batch_size = batch_size

    def set_synthetic_generation_num_docs(self, num_docs: int) -> None:
        """Sets the number of documents used for this synthetic generation session."""
        self.synthetic_generation_num_docs = num_docs

    def set_synthetic_generation_num_evals(self, num_evals: int) -> None:
        """Sets the number of evaluations to generate in total in this synthetic generation session."""
        self.synthetic_generation_num_evals = num_evals

    def set_monitoring_wheel_version(self, wheel_version: str) -> None:
        """Sets the internal version of the monitoring wheel used for this evaluation."""
        self.monitoring_wheel_version = wheel_version


# We use a thread-local storage to store the session for the current thread so that we allow multi-thread
# execution of the eval APIs. The session can be propogated between threads, for example:
#   from concurrent.futures import ThreadPoolExecutor
#   parent_session = current_session()
#   def worker_fn(x):
#       set_session(parent_session)  # Propagate session to worker thread
#       return do_work(x)
#   with ThreadPoolExecutor(max_workers=4) as executor:
#       results = list(executor.map(worker_fn, work_items))
_sessions = threading.local()
_SESSION_KEY = "rag-eval-session"


def init_session(session_id: str) -> None:
    """Initializes the session for the current thread."""
    session = Session(session_id)
    setattr(_sessions, _SESSION_KEY, session)


def current_session() -> Optional[Session]:
    """Gets the session for the current thread."""
    return getattr(_sessions, _SESSION_KEY, None)


def clear_session() -> None:
    """Clears the session for the current thread."""
    setattr(_sessions, _SESSION_KEY, None)


def set_session(session_obj: Optional[Session]) -> None:
    """Sets the session for the current thread."""
    setattr(_sessions, _SESSION_KEY, session_obj)
