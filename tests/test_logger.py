"""Tests de src/logger.py: eventos en MongoDB, por lote y diferenciados."""

import logging
import threading
from collections.abc import Iterator

import mongomock
import pytest
from pymongo.errors import AutoReconnect

from src import logger, mongo


@pytest.fixture
def logs(mongo_db: mongomock.Database) -> mongomock.Collection:
    """La colección ``logs`` de la base simulada."""
    return mongo_db[mongo.LOGS]


@pytest.fixture
def handler(logs: mongomock.Collection) -> logger.MongoHandler:
    """Un handler sobre la colección de prueba."""
    return logger.MongoHandler(logs)


@pytest.fixture
def log(handler: logger.MongoHandler) -> Iterator[logging.Logger]:
    """Logger de prueba con el MongoHandler enganchado."""
    test_logger = logging.getLogger("src.test_logger")
    test_logger.setLevel(logging.DEBUG)
    test_logger.addHandler(handler)
    yield test_logger
    test_logger.removeHandler(handler)


def stored(logs: mongomock.Collection) -> list[dict]:
    """Los eventos guardados, en orden de llegada."""
    return list(logs.find({}, {"_id": 0}).sort("_id", 1))


# Lote ---------------------------------------------------------------------------


def test_events_wait_in_memory_until_flush(
    log: logging.Logger, handler: logger.MongoHandler, logs: mongomock.Collection
) -> None:
    """Un evento común no sale solo: espera en el lote hasta enviarlo."""
    log.info("hola %s", "mundo")
    assert handler.pending == 1
    assert logs.count_documents({}) == 0
    handler.flush()
    assert handler.pending == 0
    [event] = stored(logs)
    assert event["message"] == "hola mundo"
    assert event["level"] == "INFO"
    assert event["module"] == "src.test_logger"
    assert event["ts"] is not None


def test_one_insert_sends_the_whole_batch(
    log: logging.Logger,
    handler: logger.MongoHandler,
    logs: mongomock.Collection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Varios eventos viajan en una sola operación, y cada uno es un documento."""
    calls: list[int] = []
    real = logs.insert_many
    monkeypatch.setattr(
        handler,
        "_collection",
        type(
            "C",
            (),
            {
                "insert_many": lambda self, docs: (calls.append(len(docs)), real(docs))[
                    1
                ]
            },
        )(),
    )
    for index in range(5):
        log.info("evento %d", index)
    handler.flush()
    assert calls == [5]
    assert [e["message"] for e in stored(logs)] == [f"evento {i}" for i in range(5)]


def test_an_error_sends_the_batch_immediately(
    log: logging.Logger, logs: mongomock.Collection
) -> None:
    """Un ERROR no espera: sale junto con lo que había pendiente."""
    log.info("antes")
    assert logs.count_documents({}) == 0
    log.error("falló")
    assert [e["message"] for e in stored(logs)] == ["antes", "falló"]


def test_a_full_batch_is_sent(
    log: logging.Logger, logs: mongomock.Collection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Al llegar al tamaño del lote, sale solo."""
    monkeypatch.setattr(logger, "BATCH_SIZE", 3)
    log.info("1")
    log.info("2")
    assert logs.count_documents({}) == 0
    log.info("3")
    assert logs.count_documents({}) == 3


def test_an_old_event_triggers_the_send(
    log: logging.Logger, logs: mongomock.Collection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Si el evento más viejo ya esperó demasiado, el próximo evento envía el lote."""
    clock = [1000.0]
    monkeypatch.setattr(logger.time, "monotonic", lambda: clock[0])
    log.info("viejo")
    clock[0] += logger.MAX_AGE_SECONDS + 1
    log.info("nuevo")
    assert [e["message"] for e in stored(logs)] == ["viejo", "nuevo"]


def test_close_sends_what_is_pending(
    log: logging.Logger, handler: logger.MongoHandler, logs: mongomock.Collection
) -> None:
    """Al cerrar el handler (también al salir del programa) no se pierde nada."""
    log.info("pendiente")
    handler.close()
    assert [e["message"] for e in stored(logs)] == ["pendiente"]


def test_flush_with_nothing_pending_does_nothing(
    handler: logger.MongoHandler, logs: mongomock.Collection
) -> None:
    """Enviar sin eventos no toca la base."""
    handler.flush()
    assert logs.count_documents({}) == 0


def test_failed_send_keeps_the_events_and_retries(
    log: logging.Logger,
    handler: logger.MongoHandler,
    logs: mongomock.Collection,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Si MongoDB no responde, no rompe el programa, avisa y reintenta después."""
    real = logs.insert_many

    def down(docs: list[dict]) -> None:
        raise AutoReconnect("sin red")

    monkeypatch.setattr(logs, "insert_many", down)
    log.info("uno")
    log.error("dos")  # intenta enviar y falla, sin lanzar
    assert handler.pending == 2
    assert "No se pudieron guardar 2 eventos" in capsys.readouterr().err
    monkeypatch.setattr(logs, "insert_many", real)
    log.info("tres")
    handler.flush()
    assert [e["message"] for e in stored(logs)] == ["uno", "dos", "tres"]


def test_pending_events_have_a_cap_when_mongodb_stays_down(
    log: logging.Logger,
    handler: logger.MongoHandler,
    logs: mongomock.Collection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Con MongoDB caído mucho tiempo la memoria no crece sin límite."""
    monkeypatch.setattr(logger, "MAX_PENDING", 5)
    monkeypatch.setattr(
        logs, "insert_many", lambda docs: (_ for _ in ()).throw(AutoReconnect("x"))
    )
    for index in range(20):
        log.error("evento %d", index)
    assert handler.pending == 5
    assert handler._pending[-1]["message"] == "evento 19"


def test_the_handler_does_not_connect_until_the_first_send(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Crear el handler no abre una conexión: solo al enviar el primer lote."""
    touched: list[str] = []
    monkeypatch.setattr(mongo, "get_database", lambda: touched.append("db"))
    created = logger.MongoHandler()
    assert touched == []
    assert created.pending == 0


def test_default_collection_is_logs_of_the_shared_database(
    mongo_db: mongomock.Database,
) -> None:
    """Sin colección explícita usa ``logs`` de la base compartida."""
    created = logger.MongoHandler()
    record = logging.LogRecord("src.x", logging.INFO, "f", 1, "hola", (), None)
    created.handle(record)
    created.flush()
    assert mongo_db[mongo.LOGS].count_documents({"message": "hola"}) == 1


def test_a_failure_while_building_the_event_is_reported_not_raised(
    log: logging.Logger, handler: logger.MongoHandler, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Si el evento no se puede armar, va al manejador de errores de logging."""
    failures: list[logging.LogRecord] = []
    monkeypatch.setattr(handler, "handleError", failures.append)
    monkeypatch.setattr(
        logger, "build_event", lambda record: (_ for _ in ()).throw(RuntimeError("x"))
    )
    log.info("no debe lanzar")
    assert [r.getMessage() for r in failures] == ["no debe lanzar"]


def test_concurrent_threads_do_not_lose_events(
    log: logging.Logger, handler: logger.MongoHandler, logs: mongomock.Collection
) -> None:
    """Muchos hilos escribiendo a la vez dejan todos sus eventos."""

    def work(thread: int) -> None:
        for index in range(25):
            log.info("hilo %d evento %d", thread, index)

    threads = [threading.Thread(target=work, args=(n,)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    handler.flush()
    assert logs.count_documents({}) == 200


# Documento ----------------------------------------------------------------------


def test_run_id_taken_from_context(
    log: logging.Logger, handler: logger.MongoHandler, logs: mongomock.Collection
) -> None:
    """El run_id del ContextVar queda en cada evento."""
    log.info("sin corrida")
    token = logger.current_run_id.set("abc12345")
    try:
        log.info("con corrida")
    finally:
        logger.current_run_id.reset(token)
    handler.flush()
    first, second = stored(logs)
    assert first["run_id"] is None
    assert second["run_id"] == "abc12345"


def test_plain_events_have_no_step_fields(
    log: logging.Logger, handler: logger.MongoHandler, logs: mongomock.Collection
) -> None:
    """Fuera de un paso, el documento no trae step, event ni duration_ms."""
    log.info("suelto")
    handler.flush()
    [event] = stored(logs)
    assert not {"step", "event", "duration_ms"} & set(event)


# @log_step ----------------------------------------------------------------------


@pytest.fixture
def step_logging(handler: logger.MongoHandler) -> Iterator[None]:
    """Conecta el logger de este módulo (el que usa @log_step) a la colección.

    También lo engancha al logger raíz de ``src``, que es donde ``flush_logs``
    busca los handlers; los eventos del módulo no pasan por ahí, así que no se
    duplican.
    """
    module_logger = logging.getLogger(__name__)
    root = logging.getLogger(logger.ROOT_LOGGER_NAME)
    old_level = module_logger.level
    module_logger.setLevel(logging.INFO)
    module_logger.addHandler(handler)
    root.addHandler(handler)
    yield
    root.removeHandler(handler)
    module_logger.removeHandler(handler)
    module_logger.setLevel(old_level)


@pytest.mark.usefixtures("step_logging")
def test_log_step_logs_start_and_end_with_their_fields(
    logs: mongomock.Collection,
) -> None:
    """Un paso exitoso registra inicio y fin, diferenciados, y devuelve su resultado."""

    @logger.log_step
    def suma(a: int, b: int) -> int:
        """Suma dos números."""
        return a + b

    assert suma(2, 3) == 5
    start, end = stored(logs)
    assert (start["level"], start["message"]) == ("INFO", "Inicio: suma")
    assert (start["step"], start["event"]) == ("suma", "inicio")
    assert "duration_ms" not in start
    assert end["message"].startswith("Fin: suma (")
    assert (end["step"], end["event"]) == ("suma", "fin")
    assert isinstance(end["duration_ms"], int) and end["duration_ms"] >= 0
    assert suma.__name__ == "suma"
    assert suma.__doc__ == "Suma dos números."


@pytest.mark.usefixtures("step_logging")
def test_log_step_logs_error_and_reraises(logs: mongomock.Collection) -> None:
    """Un paso que falla registra ERROR con su duración y relanza la excepción."""

    @logger.log_step
    def falla() -> None:
        raise ValueError("datos malos")

    with pytest.raises(ValueError, match="datos malos"):
        falla()
    start, error = stored(logs)  # el error sale solo, sin esperar al lote
    assert start["event"] == "inicio"
    assert (error["level"], error["step"], error["event"]) == (
        "ERROR",
        "falla",
        "error",
    )
    assert "ValueError: datos malos" in error["message"]
    assert "duration_ms" in error


@pytest.mark.usefixtures("step_logging")
def test_events_inside_a_step_carry_its_name(
    logs: mongomock.Collection, handler: logger.MongoHandler
) -> None:
    """Lo que se registra dentro de un paso queda etiquetado con ese paso."""
    inner_logger = logging.getLogger(__name__)

    @logger.log_step
    def entrenar() -> None:
        inner_logger.info("trabajando")

    inner_logger.info("antes")
    entrenar()
    events = {e["message"]: e for e in stored(logs)}
    assert "step" not in events["antes"]
    assert events["trabajando"]["step"] == "entrenar"
    assert "event" not in events["trabajando"]


@pytest.mark.usefixtures("step_logging")
def test_nested_steps_use_the_innermost_name_and_send_once(
    logs: mongomock.Collection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Un paso dentro de otro conserva su nombre; el envío sale al cerrar el externo."""
    inner_logger = logging.getLogger(__name__)
    sends: list[int] = []
    real_flush = logger.flush_logs
    monkeypatch.setattr(
        logger, "flush_logs", lambda: (sends.append(1), real_flush())[1]
    )

    @logger.log_step
    def interno() -> None:
        inner_logger.info("en el interno")

    @logger.log_step
    def externo() -> None:
        inner_logger.info("en el externo")
        interno()
        assert logs.count_documents({}) == 0  # todavía en el lote

    externo()
    assert sends == [1]
    by_message = {e["message"]: e["step"] for e in stored(logs) if "step" in e}
    assert by_message["en el externo"] == "externo"
    assert by_message["en el interno"] == "interno"
    assert [e["event"] for e in stored(logs) if e.get("event")] == [
        "inicio",
        "inicio",
        "fin",
        "fin",
    ]


@pytest.mark.usefixtures("step_logging")
def test_the_step_name_is_restored_after_a_failing_inner_step(
    logs: mongomock.Collection,
) -> None:
    """Si un paso interno falla y el externo sigue, el externo recupera su nombre."""
    inner_logger = logging.getLogger(__name__)

    @logger.log_step
    def roto() -> None:
        raise RuntimeError("x")

    @logger.log_step
    def externo() -> None:
        with pytest.raises(RuntimeError):
            roto()
        inner_logger.info("sigue")

    externo()
    follow = next(e for e in stored(logs) if e["message"] == "sigue")
    assert follow["step"] == "externo"


# muted --------------------------------------------------------------------------


def test_muted_blocks_writes_only_inside_the_block(
    log: logging.Logger, handler: logger.MongoHandler, logs: mongomock.Collection
) -> None:
    """Dentro de muted() no se guarda nada; antes y después sí."""
    log.info("antes")
    with logger.muted():
        log.info("silenciado")
        log.error("también silenciado")
    log.info("después")
    handler.flush()
    assert [e["message"] for e in stored(logs)] == ["antes", "después"]


def test_muted_is_restored_even_if_the_block_fails(
    log: logging.Logger, handler: logger.MongoHandler, logs: mongomock.Collection
) -> None:
    """Si el bloque lanza una excepción, el logger vuelve a guardar."""
    with pytest.raises(RuntimeError), logger.muted():
        raise RuntimeError("falla")
    log.info("sigue funcionando")
    handler.flush()
    assert [e["message"] for e in stored(logs)] == ["sigue funcionando"]


def test_muted_nests_and_silences_log_step(
    handler: logger.MongoHandler, logs: mongomock.Collection, step_logging: None
) -> None:
    """Un paso con @log_step dentro de muted() tampoco deja eventos."""

    @logger.log_step
    def paso() -> int:
        return 1

    with logger.muted():
        with logger.muted():
            paso()
        paso()
    handler.flush()
    assert stored(logs) == []
    paso()
    handler.flush()
    assert len(stored(logs)) == 2


# setup_logging y flush_logs -----------------------------------------------------


def test_setup_logging_is_idempotent() -> None:
    """setup_logging no duplica el handler."""
    root = logging.getLogger(logger.ROOT_LOGGER_NAME)
    before = len(root.handlers)
    logger.setup_logging()
    logger.setup_logging()
    ours = [h for h in root.handlers if isinstance(h, logger.MongoHandler)]
    assert len(ours) == 1
    assert len(root.handlers) == before + 1


def test_setup_logging_keeps_other_handlers_and_sets_the_level() -> None:
    """Otros handlers no cuentan como el nuestro y el nivel se aplica."""
    root = logging.getLogger(logger.ROOT_LOGGER_NAME)
    other = logging.StreamHandler()
    root.addHandler(other)
    returned = logger.setup_logging(logging.WARNING)
    assert returned is root and root.level == logging.WARNING
    assert other in root.handlers
    assert any(isinstance(h, logger.MongoHandler) for h in root.handlers)


def test_flush_logs_sends_pending_events_of_the_package_handlers(
    mongo_db: mongomock.Database,
) -> None:
    """``flush_logs`` envía lo pendiente del handler instalado por setup_logging."""
    logger.setup_logging()
    logging.getLogger("src.test_flush").info("pendiente")
    assert mongo_db[mongo.LOGS].count_documents({}) == 0
    logger.flush_logs()
    assert mongo_db[mongo.LOGS].count_documents({"message": "pendiente"}) == 1


def test_flush_logs_without_handlers_is_harmless() -> None:
    """Sin ningún MongoHandler instalado, no hace nada."""
    logger.flush_logs()


def test_flush_logs_skips_handlers_that_are_not_ours(
    mongo_db: mongomock.Database,
) -> None:
    """Un handler ajeno en el logger del paquete no se toca ni rompe el envío."""
    root = logging.getLogger(logger.ROOT_LOGGER_NAME)
    root.addHandler(logging.NullHandler())
    logger.setup_logging()
    logging.getLogger("src.test_flush").info("pendiente")
    logger.flush_logs()
    assert mongo_db[mongo.LOGS].count_documents({"message": "pendiente"}) == 1
