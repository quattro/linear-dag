import logging
import os

from typing import Optional

import psutil


class MemoryLogger:
    """Logger wrapper that injects current RSS memory usage into each log message.

    !!! Example

        ```python
        logger = MemoryLogger(__name__, log_file="run.log")
        logger.info("Starting pipeline stage")
        ```
    """

    def __init__(self, name: str, log_file: Optional[str] = "memory_usage.log"):
        # Configure logging
        self.logger = logging.getLogger(name)

        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)

            # Create formatters and handlers
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - [Memory: %(memory_usage).2f MB] - %(message)s"
            )

            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)

            # File handler
            if log_file:
                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)

            self.logger.addHandler(console_handler)

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # Convert bytes to MB

    def info(self, message: str) -> None:
        """Log an INFO message annotated with the current process memory usage.

        **Arguments:**

        - `message`: Message content to emit.

        **Returns:**

        - `None`.
        """

        self.logger.info(message, extra={"memory_usage": self._get_memory_usage()})
