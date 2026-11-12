"""Tests de src/logger.py."""

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from src import db, logger


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Ruta de una base temporal."""
    return tmp_path / "test.db"


@pytest.fixture
def log(db_path: Path) -> Iterator[logging.Logger]:
    """Logger de prueba con un SQLiteHandler sobre la base temporal."""
    test_logger = logging.getLogger("src.test_logger")
    test_logger.setLevel(logging.DEBUG)
    handler = logger.SQLiteHandler(db_path)
    test_logger.addHandler(handler)
    yield test_logger
    test_logger.removeHandler(handler)


def read_logs(db_path: Path) -> list:
    """Lee todos los eventos de logs."""
    with db.get_connection(db_path) as conn:
        return conn.execute("SELECT * FROM logs ORDER BY id").fetchall()


def test_writes_and_reads_events(log: logging.Logger, db_path: Path) -> None:
    """Un evento escrito se puede leer de la tabla logs."""
    log.info("hola %s", "mundo")
    log.error("algo falló")
    rows = read_logs(db_path)
    assert [r["level"] for r in rows] == ["INFO", "ERROR"]
    assert rows[0]["message"] == "hola mundo"
    assert rows[0]["module"] == "src.test_logger"
    assert rows[0]["ts"]


def test_run_id_taken_from_context(log: logging.Logger, db_path: Path) -> None:
    """El run_id del ContextVar queda en cada evento."""
    log.info("sin corrida")
    token = logger.current_run_id.set("abc12345")
    try:
        log.info("con corrida")
    finally:
        logger.current_run_id.reset(token)
    rows = read_logs(db_path)
    assert rows[0]["run_id"] is None
    assert rows[1]["run_id"] == "abc12345"


def test_handler_failure_does_not_raise(
    log: logging.Logger, db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Si falla la base, el logging no rompe al programa."""

    def broken(*args, **kwargs):
        raise RuntimeError("base caída")

    monkeypatch.setattr(db, "get_connection", broken)
    monkeypatch.setattr(logging, "raiseExceptions", False)
    log.info("no debe lanzar")


def test_setup_logging_is_idempotent(db_path: Path) -> None:
    """setup_logging no duplica el handler."""
    root = logging.getLogger(logger.ROOT_LOGGER_NAME)
    before = len(root.handlers)
    try:
        logger.setup_logging(db_path)
        logger.setup_logging(db_path)
        added = [h for h in root.handlers if isinstance(h, logger.SQLiteHandler)]
        assert len(root.handlers) == before + 1
        assert added
    finally:
        for h in list(root.handlers)[before:]:
            root.removeHandler(h)
