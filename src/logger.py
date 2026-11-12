"""Logger propio: guarda los eventos en la tabla ``logs`` de ``structured.db``.

El ``run_id`` de la corrida actual se mantiene en un ``ContextVar``, así el
handler lo toma solo, sin importar si se ejecuta por línea de comandos o
desde Streamlit.
"""

import functools
import logging
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from src import db

ROOT_LOGGER_NAME = "src"

current_run_id: ContextVar[str | None] = ContextVar("current_run_id", default=None)
_muted: ContextVar[bool] = ContextVar("logging_muted", default=False)


@contextmanager
def muted() -> Iterator[None]:
    """Silencia el guardado en ``logs`` dentro del bloque.

    Sirve para lecturas de la interfaz que reutilizan funciones con
    ``@log_step``: mirar una página no debe agregar eventos al historial.
    """
    token = _muted.set(True)
    try:
        yield
    finally:
        _muted.reset(token)


class SQLiteHandler(logging.Handler):
    """Handler de ``logging`` que inserta cada evento en la tabla ``logs``."""

    def __init__(self, db_path: Path | None = None) -> None:
        """Crea el handler y asegura que el esquema exista.

        Args:
            db_path: ruta de la base; por defecto ``structured.db``.
        """
        super().__init__()
        self.db_path = db_path
        db.init_structured_db(db_path)

    def emit(self, record: logging.LogRecord) -> None:
        """Inserta un evento en ``logs`` (abre una conexión por evento)."""
        if _muted.get():
            return
        try:
            with db.get_connection(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO logs (ts, level, module, message, run_id) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        db.now_iso(),
                        record.levelname,
                        record.name,
                        record.getMessage(),
                        current_run_id.get(),
                    ),
                )
        except Exception:  # noqa: BLE001 - un fallo al loguear no debe romper el programa
            self.handleError(record)


def setup_logging(
    db_path: Path | None = None, level: int = logging.INFO
) -> logging.Logger:
    """Conecta el logger del paquete ``src`` con ``structured.db``.

    Es idempotente: si ya hay un ``SQLiteHandler`` para esa base, no agrega otro.

    Args:
        db_path: ruta de la base; por defecto ``structured.db``.
        level: nivel mínimo a registrar.

    Returns:
        El logger raíz del paquete.
    """
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    logger.setLevel(level)
    for handler in logger.handlers:
        if isinstance(handler, SQLiteHandler) and handler.db_path == db_path:
            return logger
    logger.addHandler(SQLiteHandler(db_path))
    return logger


def log_step[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    """Decora un paso del pipeline y registra su inicio, su fin y sus errores.

    Registra ``INFO`` al empezar y al terminar (con la duración) y ``ERROR`` si
    la función lanza una excepción, que se vuelve a lanzar sin modificarla. El
    evento queda a nombre del módulo de la función decorada.

    Args:
        func: función a decorar.

    Returns:
        La función envuelta, con el mismo nombre y docstring.
    """
    step_logger = logging.getLogger(func.__module__)

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        step_logger.info("Inicio: %s", func.__name__)
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
        except Exception as exc:
            elapsed = time.perf_counter() - start
            step_logger.error(
                "Error en %s tras %.2f s: %s: %s",
                func.__name__,
                elapsed,
                type(exc).__name__,
                exc,
            )
            raise
        elapsed = time.perf_counter() - start
        step_logger.info("Fin: %s (%.2f s)", func.__name__, elapsed)
        return result

    return wrapper
