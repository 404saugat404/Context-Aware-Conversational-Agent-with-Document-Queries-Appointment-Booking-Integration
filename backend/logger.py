"""
Centralized logging configuration.

Usage in any module:
    from backend.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Something happened")
"""

import logging
import sys

from backend.config import LOG_LEVEL

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(
    level=LOG_LEVEL.upper(),
    format=_LOG_FORMAT,
    datefmt=_DATE_FORMAT,
    stream=sys.stdout,
)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger that inherits the root configuration."""
    return logging.getLogger(name)
