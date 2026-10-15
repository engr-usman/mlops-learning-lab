import logging
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from uuid import uuid4


_run_id = ContextVar("logging_run_id", default="-")


@contextmanager
def run_logging_context(run_id=None):
    """Attach one execution ID to logs and restore the caller's context on exit."""
    execution_id = str(run_id) if run_id is not None else str(uuid4())
    token = _run_id.set(execution_id)
    try:
        yield execution_id
    finally:
        _run_id.reset(token)


class RunIdFilter(logging.Filter):
    def filter(self, record):
        record.run_id = _run_id.get()
        return True


def setup_logging(level=logging.INFO):
    """Configure console logging once at an application entry point.

    Existing handlers are preserved when embedded in another application.
    Pass logging.DEBUG to include diagnostic details.
    Logs outside a run_logging_context use run_id=-.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(RunIdFilter())
    logging.basicConfig(
        level=level,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "run_id=%(run_id)s | "
            "%(message)s"
        ),
        handlers=[handler],
    )
