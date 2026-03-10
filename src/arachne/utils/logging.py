"""Logging utilities for arachne."""

import logging


def setup_named_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Set up a named logger with a standard format.

    Args:
        name: Logger name, typically `__name__` of the calling module.
        level: Logging level. Defaults to INFO.

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
