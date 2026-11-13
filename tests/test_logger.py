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

    failures: list[logging.LogRecord] = []
    handler = log.handlers[0]
    monkeypatch.setattr(db, "get_connection", broken)
    monkeypatch.setattr(handler, "handleError", failures.append)
    log.info("no debe lanzar")
    log.error("tampoco este")
    # El fallo se reportó al manejador de errores de logging, no se propagó.
    assert [r.getMessage() for r in failures] == ["no debe lanzar", "tampoco este"]


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


@pytest.fixture
def step_logging(db_path: Path) -> Iterator[None]:
    """Conecta el logger de este módulo (el que usa @log_step) a la base."""
    module_logger = logging.getLogger(__name__)
    old_level = module_logger.level
    handler = logger.SQLiteHandler(db_path)
    module_logger.setLevel(logging.INFO)
    module_logger.addHandler(handler)
    yield
    module_logger.removeHandler(handler)
    module_logger.setLevel(old_level)


@pytest.mark.usefixtures("step_logging")
def test_log_step_logs_start_and_end(db_path: Path) -> None:
    """Un paso exitoso registra inicio y fin y devuelve su resultado."""

    @logger.log_step
    def suma(a: int, b: int) -> int:
        """Suma dos números."""
        return a + b

    assert suma(2, 3) == 5
    rows = read_logs(db_path)
    assert [r["level"] for r in rows] == ["INFO", "INFO"]
    assert rows[0]["message"] == "Inicio: suma"
    assert rows[1]["message"].startswith("Fin: suma (")
    assert suma.__name__ == "suma"
    assert suma.__doc__ == "Suma dos números."


@pytest.mark.usefixtures("step_logging")
def test_log_step_logs_error_and_reraises(db_path: Path) -> None:
    """Un paso que falla registra ERROR y relanza la misma excepción."""

    @logger.log_step
    def falla() -> None:
        raise ValueError("datos malos")

    with pytest.raises(ValueError, match="datos malos"):
        falla()
    rows = read_logs(db_path)
    assert [r["level"] for r in rows] == ["INFO", "ERROR"]
    assert "ValueError: datos malos" in rows[1]["message"]


def test_muted_blocks_writes_only_inside_the_block(
    log: logging.Logger, db_path: Path
) -> None:
    """Dentro de muted() no se guarda nada; antes y después sí."""
    log.info("antes")
    with logger.muted():
        log.info("silenciado")
        log.error("también silenciado")
    log.info("después")
    assert [r["message"] for r in read_logs(db_path)] == ["antes", "después"]


def test_muted_is_restored_even_if_the_block_fails(
    log: logging.Logger, db_path: Path
) -> None:
    """Si el bloque lanza una excepción, el logger vuelve a guardar."""
    with pytest.raises(RuntimeError), logger.muted():
        raise RuntimeError("falla")
    log.info("sigue funcionando")
    assert [r["message"] for r in read_logs(db_path)] == ["sigue funcionando"]


def test_muted_nests_and_silences_log_step(
    log: logging.Logger, db_path: Path, step_logging: None
) -> None:
    """Un paso con @log_step dentro de muted() tampoco deja eventos."""

    @logger.log_step
    def paso() -> int:
        return 1

    with logger.muted():
        with logger.muted():
            paso()
        paso()
    assert read_logs(db_path) == []
    paso()
    assert len(read_logs(db_path)) == 2


def test_setup_logging_adds_its_handler_even_with_other_handlers_or_paths(
    tmp_path: Path,
) -> None:
    """Otros handlers no cuentan como el nuestro, y otra base agrega otro handler."""
    root = logging.getLogger(logger.ROOT_LOGGER_NAME)
    other = logging.StreamHandler()
    root.addHandler(other)
    first, second = tmp_path / "a.db", tmp_path / "b.db"
    logger.setup_logging(first)
    logger.setup_logging(second)
    logger.setup_logging(first)  # ya estaba: no se repite
    ours = [h for h in root.handlers if isinstance(h, logger.SQLiteHandler)]
    assert [h.db_path for h in ours] == [first, second]
    assert other in root.handlers  # el ajeno quedó como estaba
