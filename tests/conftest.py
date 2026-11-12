"""Fixtures compartidas."""

import logging
from collections.abc import Iterator

import pytest

from src import logger


@pytest.fixture
def clean_root_logger() -> Iterator[None]:
    """Deja el logger raíz del paquete como estaba al terminar el test."""
    root = logging.getLogger(logger.ROOT_LOGGER_NAME)
    old_level, old_handlers = root.level, list(root.handlers)
    yield
    root.setLevel(old_level)
    root.handlers[:] = old_handlers
