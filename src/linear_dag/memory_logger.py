import logging
import os
import sys

from typing import IO, Optional

import psutil

DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - [Memory: %(memory_usage).2f MB] - %(message)s"
_MEMORY_HANDLER_ATTR = "_linear_dag_memory_handler"


def get_memory_usage_mb() -> float:
    """Return current process RSS memory usage in megabytes."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


class MemoryUsageFilter(logging.Filter):
    """Inject `memory_usage` onto log records when missing."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "memory_usage"):
            record.memory_usage = get_memory_usage_mb()
        return True


def ensure_memory_usage_filter(logger: logging.Logger) -> None:
    """Attach `MemoryUsageFilter` once to a logger."""
    if any(isinstance(existing_filter, MemoryUsageFilter) for existing_filter in logger.filters):
        return
    logger.addFilter(MemoryUsageFilter())


def remove_managed_memory_handlers(logger: logging.Logger) -> None:
    """Remove and close handlers created by `configure_memory_logger`."""
    for handler in list(logger.handlers):
        if getattr(handler, _MEMORY_HANDLER_ATTR, False):
            logger.removeHandler(handler)
            handler.close()


def configure_memory_logger(
    logger: logging.Logger,
    *,
    level: int = logging.INFO,
    stream: Optional[IO[str]] = None,
    log_file: Optional[str] = None,
    replace_managed_handlers: bool = False,
    propagate: bool = False,
    formatter: Optional[logging.Formatter] = None,
) -> list[logging.Handler]:
    """Configure a logger with memory-aware formatting and optional managed handlers."""
    logger.setLevel(level)
    logger.propagate = propagate
    ensure_memory_usage_filter(logger)
    if replace_managed_handlers:
        remove_managed_memory_handlers(logger)

    active_formatter = formatter or logging.Formatter(DEFAULT_LOG_FORMAT)
    created_handlers: list[logging.Handler] = []

    if stream is not None:
        stream_handler = logging.StreamHandler(stream)
        stream_handler.setFormatter(active_formatter)
        setattr(stream_handler, _MEMORY_HANDLER_ATTR, True)
        logger.addHandler(stream_handler)
        created_handlers.append(stream_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(active_formatter)
        setattr(file_handler, _MEMORY_HANDLER_ATTR, True)
        logger.addHandler(file_handler)
        created_handlers.append(file_handler)

    return created_handlers


class MemoryLogger:
    """Logger wrapper that injects current RSS memory usage into each log message.

    !!! Example

        ```python
        logger = MemoryLogger(__name__, log_file="run.log")
        logger.info("Starting pipeline stage")
        ```
    """

    def __init__(
        self,
        name: Optional[str] = None,
        log_file: Optional[str] = "memory_usage.log",
        logger: Optional[logging.Logger] = None,
    ):
        if logger is None and name is None:
            raise ValueError("Must provide either logger name or logger instance.")

        self.logger = logger if logger is not None else logging.getLogger(name)
        ensure_memory_usage_filter(self.logger)

        if not self.logger.handlers:
            configure_memory_logger(
                self.logger,
                level=logging.INFO,
                stream=sys.stderr,
                log_file=log_file,
            )

    def info(self, message: str) -> None:
        """Log an INFO message annotated with the current process memory usage.

        **Arguments:**

        - `message`: Message content to emit.

        **Returns:**

        - `None`.
        """

        self.logger.info(message)

    def warning(self, message: str) -> None:
        self.logger.warning(message)

    def error(self, message: str) -> None:
        self.logger.error(message)

    def debug(self, message: str) -> None:
        self.logger.debug(message)
