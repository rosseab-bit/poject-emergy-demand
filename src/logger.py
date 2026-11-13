"""Logger propio: guarda los eventos en la colección ``logs`` de MongoDB.

Cada evento es un documento distinto, con ``run_id``, ``ts``, ``level``,
``module`` y ``message``. Los que ocurren dentro de un paso decorado con
``@log_step`` llevan además el nombre del ``step``, y los del propio decorador
(inicio, fin y error) traen ``event`` y, al terminar, ``duration_ms``. Así se
puede filtrar por corrida, por paso o por tipo de evento.

Los eventos se juntan en memoria y se envían de a lotes, porque cada ida y
vuelta a Atlas cuesta unos 30 ms. Un lote sale: al terminar un paso de primer
nivel, ante cualquier ``ERROR`` (la traza no se pierde justo cuando importa),
al llegar a ``BATCH_SIZE`` eventos, cuando el más viejo lleva ``MAX_AGE_SECONDS``
esperando y al cerrar el programa. Si el proceso muere de golpe, se pierden los
eventos del paso en curso que todavía no se enviaron.

El ``run_id`` de la corrida actual se mantiene en un ``ContextVar``, así el
handler lo toma solo, sin importar si se ejecuta por línea de comandos o
desde Streamlit.
"""

import functools
import logging
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime

from pymongo.collection import Collection

from src import mongo

ROOT_LOGGER_NAME = "src"

BATCH_SIZE = 100
MAX_AGE_SECONDS = 5.0
MAX_PENDING = 1000  # tope de eventos retenidos si MongoDB no responde

EVENT_START = "inicio"
EVENT_END = "fin"
EVENT_ERROR = "error"

current_run_id: ContextVar[str | None] = ContextVar("current_run_id", default=None)
_muted: ContextVar[bool] = ContextVar("logging_muted", default=False)
_steps: ContextVar[tuple[str, ...]] = ContextVar("logging_steps", default=())


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


@contextmanager
def run_context(run_id: str) -> Iterator[None]:
    """Deja activo un ``run_id`` dentro del bloque y envía lo pendiente al salir.

    Sirve para registrar un evento que ocurre fuera de una corrida pero le
    pertenece, como avisar dónde se guardó su reporte.
    """
    token = current_run_id.set(run_id)
    try:
        yield
    finally:
        current_run_id.reset(token)
        flush_logs()


def build_event(record: logging.LogRecord) -> dict:
    """Arma el documento de un evento a partir de un registro de ``logging``.

    Los campos ``step``, ``event`` y ``duration_ms`` solo están cuando aplican.

    Args:
        record: el registro; puede traer ``step``, ``event`` y ``duration_ms``
            como atributos extra (los pone ``@log_step``).

    Returns:
        El documento listo para insertar en ``logs``.
    """
    steps = _steps.get()
    optional = {
        "step": getattr(record, "step", None) or (steps[-1] if steps else None),
        "event": getattr(record, "event", None),
        "duration_ms": getattr(record, "duration_ms", None),
    }
    return {
        "run_id": current_run_id.get(),
        "ts": datetime.fromtimestamp(record.created, UTC),
        "level": record.levelname,
        "module": record.name,
        "message": record.getMessage(),
        **{key: value for key, value in optional.items() if value is not None},
    }


class MongoHandler(logging.Handler):
    """Handler de ``logging`` que guarda cada evento en ``logs``, en lotes."""

    def __init__(self, collection: Collection | None = None) -> None:
        """Crea el handler sin conectarse: la conexión se abre al primer envío.

        Args:
            collection: colección destino; por defecto la colección ``logs`` de
                la base compartida (``mongo.get_database``).
        """
        super().__init__()
        self._collection = collection
        self._pending: list[dict] = []
        self._oldest = 0.0

    @property
    def collection(self) -> Collection:
        """La colección donde se guardan los eventos."""
        if self._collection is not None:
            return self._collection
        return mongo.get_database()[mongo.LOGS]

    @property
    def pending(self) -> int:
        """Cantidad de eventos esperando para enviarse."""
        return len(self._pending)

    def emit(self, record: logging.LogRecord) -> None:
        """Anota un evento y envía el lote si corresponde."""
        if _muted.get():
            return
        try:
            self._pending.append(build_event(record))
            if not self._oldest:
                self._oldest = time.monotonic()
            if self._must_send(record):
                self._send()
        except Exception:  # noqa: BLE001 - un fallo al loguear no debe romper el programa
            self.handleError(record)

    def _must_send(self, record: logging.LogRecord) -> bool:
        """Indica si el lote debe salir ya: error, lote lleno o evento viejo."""
        return (
            record.levelno >= logging.ERROR
            or len(self._pending) >= BATCH_SIZE
            or time.monotonic() - self._oldest >= MAX_AGE_SECONDS
        )

    def _send(self) -> None:
        """Inserta los eventos pendientes; si falla, los conserva para reintentar."""
        if not self._pending:
            return
        batch, self._pending, self._oldest = self._pending, [], 0.0
        try:
            self.collection.insert_many(batch)
        except Exception as error:  # noqa: BLE001 - MongoDB caído o sin configurar
            self._pending = (batch + self._pending)[-MAX_PENDING:]
            self._oldest = time.monotonic()
            sys.stderr.write(
                f"No se pudieron guardar {len(batch)} eventos de log en MongoDB "
                f"({type(error).__name__}); se reintentará.\n"
            )

    def flush(self) -> None:
        """Envía ya los eventos pendientes."""
        self.acquire()
        try:
            self._send()
        finally:
            self.release()

    def close(self) -> None:
        """Envía lo pendiente y cierra el handler."""
        self.flush()
        super().close()


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Conecta el logger del paquete ``src`` con MongoDB.

    Es idempotente: si ya hay un ``MongoHandler``, no agrega otro.

    Args:
        level: nivel mínimo a registrar.

    Returns:
        El logger raíz del paquete.
    """
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    logger.setLevel(level)
    if not any(isinstance(handler, MongoHandler) for handler in logger.handlers):
        logger.addHandler(MongoHandler())
    return logger


def flush_logs() -> None:
    """Envía a MongoDB los eventos pendientes de todos los handlers del paquete.

    Se llama al cerrar una corrida, una predicción o un reporte, para que nada
    quede esperando en memoria.
    """
    for handler in logging.getLogger(ROOT_LOGGER_NAME).handlers:
        if isinstance(handler, MongoHandler):
            handler.flush()


def log_step[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    """Decora un paso del pipeline y registra su inicio, su fin y sus errores.

    Registra ``INFO`` al empezar y al terminar (con la duración) y ``ERROR`` si
    la función lanza una excepción, que se vuelve a lanzar sin modificarla. El
    evento queda a nombre del módulo de la función decorada, con el ``step``, el
    ``event`` (``inicio``, ``fin`` o ``error``) y la ``duration_ms``. Los eventos
    que ocurren dentro del paso también llevan su ``step``. Al terminar un paso
    de primer nivel se envían los eventos pendientes.

    Args:
        func: función a decorar.

    Returns:
        La función envuelta, con el mismo nombre y docstring.
    """
    step_logger = logging.getLogger(func.__module__)
    name = func.__name__

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        step_logger.info("Inicio: %s", name, extra={"step": name, "event": EVENT_START})
        outer = _steps.get()
        token = _steps.set((*outer, name))
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
        except Exception as exc:
            _steps.reset(token)
            elapsed = time.perf_counter() - start
            step_logger.error(
                "Error en %s tras %.2f s: %s: %s",
                name,
                elapsed,
                type(exc).__name__,
                exc,
                extra={
                    "step": name,
                    "event": EVENT_ERROR,
                    "duration_ms": round(elapsed * 1000),
                },
            )
            raise
        _steps.reset(token)
        elapsed = time.perf_counter() - start
        step_logger.info(
            "Fin: %s (%.2f s)",
            name,
            elapsed,
            extra={
                "step": name,
                "event": EVENT_END,
                "duration_ms": round(elapsed * 1000),
            },
        )
        if not outer:
            flush_logs()
        return result

    return wrapper
