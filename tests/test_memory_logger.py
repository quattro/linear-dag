import io
import logging

from linear_dag.memory_logger import configure_memory_logger, MemoryLogger, remove_managed_memory_handlers


def _reset_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    remove_managed_memory_handlers(logger)
    logger.filters.clear()
    return logger


def test_configure_memory_logger_adds_memory_usage_to_records():
    logger = _reset_logger("linear_dag.tests.memory.configure")
    stream = io.StringIO()
    handlers = configure_memory_logger(
        logger,
        stream=stream,
        log_file=None,
        replace_managed_handlers=True,
    )
    try:
        logger.info("hello memory")
        for handler in handlers:
            handler.flush()
        text = stream.getvalue()
        assert "hello memory" in text
        assert "Memory:" in text
    finally:
        remove_managed_memory_handlers(logger)
        logger.filters.clear()


def test_memory_logger_reuses_preconfigured_logger_without_duplicate_handlers():
    logger = _reset_logger("linear_dag.tests.memory.wrapper")
    stream = io.StringIO()
    configure_memory_logger(
        logger,
        stream=stream,
        log_file=None,
        replace_managed_handlers=True,
    )
    before = len(logger.handlers)

    wrapped = MemoryLogger(logger=logger)
    wrapped.info("wrapped message")

    after = len(logger.handlers)
    assert after == before
    assert "wrapped message" in stream.getvalue()

    remove_managed_memory_handlers(logger)
    logger.filters.clear()
