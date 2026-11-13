"""Acceso a las bases SQLite del proyecto.

Cada operación abre su propia conexión (los hilos de Streamlit no las
comparten) en modo WAL y la cierra al terminar.
"""

import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from src import config

SCHEMA_STRUCTURED = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    status      TEXT NOT NULL DEFAULT 'running',
    seed        INTEGER,
    notes       TEXT
);

CREATE TABLE IF NOT EXISTS logs (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        TEXT NOT NULL,
    level     TEXT NOT NULL,
    module    TEXT NOT NULL,
    message   TEXT NOT NULL,
    run_id    TEXT
);

CREATE INDEX IF NOT EXISTS idx_logs_run_id ON logs (run_id);

CREATE TABLE IF NOT EXISTS datos_limpios (
    fecha                TEXT PRIMARY KEY,
    demanda_total        REAL NOT NULL,
    demanda_residencial  REAL,
    comercio_e_industria REAL,
    grandes_usuarios     REAL,
    temperatura_promedio REAL NOT NULL,
    potencia_maxima      REAL,
    potencia_repetida    INTEGER NOT NULL DEFAULT 0
);
"""


def _raw_table_ddl() -> str:
    """DDL de ``datos_originales``: el CSV completo, la fecha como clave."""
    first, *rest = config.RAW_COLUMNS
    lines = [f"    {first} TEXT PRIMARY KEY"] + [f"    {c} REAL" for c in rest]
    return (
        "CREATE TABLE IF NOT EXISTS datos_originales (\n" + ",\n".join(lines) + "\n);\n"
    )


SCHEMA_RAW = _raw_table_ddl()

SCHEMA_UNSTRUCTURED = """
CREATE TABLE IF NOT EXISTS resultados_modelos (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     TEXT NOT NULL,
    scenario   TEXT NOT NULL,
    model      TEXT NOT NULL,
    created_at TEXT NOT NULL,
    doc        TEXT NOT NULL CHECK (json_valid(doc)),
    UNIQUE (run_id, scenario, model)
);

CREATE TABLE IF NOT EXISTS configuracion (
    run_id     TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    doc        TEXT NOT NULL CHECK (json_valid(doc))
);

CREATE TABLE IF NOT EXISTS predicciones (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     TEXT NOT NULL,
    created_at TEXT NOT NULL,
    doc        TEXT NOT NULL CHECK (json_valid(doc))
);

CREATE INDEX IF NOT EXISTS idx_predicciones_run_id ON predicciones (run_id);
"""


# Cuánto espera un checkpoint a que otras conexiones suelten la base (es de mejor
# esfuerzo: no debe trabar el final de una corrida).
CHECKPOINT_TIMEOUT_SECONDS = 2.0


def _configure(conn: sqlite3.Connection) -> None:
    """Ajusta una conexión nueva: filas con nombre, WAL y escrituras rápidas.

    - ``journal_mode=WAL``: lectores y escritor no se bloquean entre sí.
    - ``synchronous=NORMAL``: en WAL no corrompe la base ante un corte de luz
      (solo podrían perderse las últimas confirmaciones), y evita un ``fsync``
      por cada confirmación.
    - Sin checkpoint al cerrar: con una conexión por operación, SQLite haría un
      checkpoint (``fsync`` del archivo principal y borrado del ``-wal``) en
      CADA cierre, lo que costaba ~100 ms por evento de log en un disco lento.
      El volcado se hace al final de cada corrida con ``checkpoint``.
    """
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.setconfig(sqlite3.SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE, True)


@contextmanager
def get_connection(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    """Abre una conexión nueva en modo WAL y la cierra al salir.

    Confirma los cambios si el bloque termina bien y los revierte si falla.

    Args:
        db_path: ruta de la base; por defecto ``structured.db``.

    Yields:
        Conexión con ``row_factory`` en ``sqlite3.Row``.
    """
    path = db_path if db_path is not None else config.STRUCTURED_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    try:
        _configure(conn)
        yield conn
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def checkpoint(
    db_path: Path | None = None, timeout: float = CHECKPOINT_TIMEOUT_SECONDS
) -> bool:
    """Vuelca el WAL al archivo principal de la base y lo deja vacío.

    Se llama al final de cada corrida: así el ``.db`` queda completo por sí solo
    (sin su ``-wal``), que es lo que se entrega o se copia para revisar. Es de
    mejor esfuerzo: si la base está ocupada no falla, solo devuelve ``False``.

    Args:
        db_path: ruta de la base; por defecto ``structured.db``.
        timeout: segundos que espera a que otras conexiones la suelten.

    Returns:
        ``True`` si se volcó todo; ``False`` si la base no existe o estaba ocupada.
    """
    path = db_path if db_path is not None else config.STRUCTURED_DB_PATH
    if not path.is_file():
        return False
    conn = sqlite3.connect(path, timeout=timeout)
    try:
        busy, _log_frames, _moved = conn.execute(
            "PRAGMA wal_checkpoint(TRUNCATE)"
        ).fetchone()
    except sqlite3.Error:
        return False
    finally:
        conn.close()
    return busy == 0


def checkpoint_all(
    structured_path: Path | None = None, unstructured_path: Path | None = None
) -> bool:
    """Hace ``checkpoint`` de las dos bases (las que existan).

    Args:
        structured_path: ruta de ``structured.db``; por defecto la de ``config``.
        unstructured_path: ruta de ``unstructured.db``; por defecto la de
            ``config``.

    Returns:
        ``True`` si las dos se volcaron por completo.
    """
    unstructured = (
        unstructured_path
        if unstructured_path is not None
        else config.UNSTRUCTURED_DB_PATH
    )
    results = [checkpoint(structured_path), checkpoint(unstructured)]
    return all(results)


def init_structured_db(db_path: Path | None = None) -> None:
    """Crea ``runs``, ``logs``, ``datos_originales`` y ``datos_limpios`` si faltan.

    Args:
        db_path: ruta de la base; por defecto ``structured.db``.
    """
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA_STRUCTURED + SCHEMA_RAW)


def now_iso() -> str:
    """Devuelve la fecha y hora actual en UTC, formato ISO 8601."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def new_run_id() -> str:
    """Genera un ``run_id`` corto: 8 caracteres hexadecimales."""
    return uuid.uuid4().hex[:8]


def create_run(
    seed: int | None = None,
    notes: str | None = None,
    db_path: Path | None = None,
) -> str:
    """Registra una corrida nueva con estado ``running``.

    Args:
        seed: semilla de la corrida; por defecto ``config.SEED``.
        notes: comentario libre opcional.
        db_path: ruta de la base; por defecto ``structured.db``.

    Returns:
        El ``run_id`` generado.
    """
    run_id = new_run_id()
    seed = config.SEED if seed is None else seed
    with get_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO runs (run_id, started_at, status, seed, notes) "
            "VALUES (?, ?, 'running', ?, ?)",
            (run_id, now_iso(), seed, notes),
        )
    return run_id


def finish_run(run_id: str, status: str, db_path: Path | None = None) -> None:
    """Cierra una corrida con su estado final.

    Args:
        run_id: identificador de la corrida.
        status: ``ok`` o ``error``.
        db_path: ruta de la base; por defecto ``structured.db``.

    Raises:
        ValueError: si el estado no es válido o el ``run_id`` no existe.
    """
    if status not in ("ok", "error"):
        raise ValueError(f"Estado inválido: {status!r}")
    with get_connection(db_path) as conn:
        cur = conn.execute(
            "UPDATE runs SET finished_at = ?, status = ? WHERE run_id = ?",
            (now_iso(), status, run_id),
        )
        if cur.rowcount == 0:
            raise ValueError(f"run_id inexistente: {run_id}")


def get_run(run_id: str, db_path: Path | None = None) -> sqlite3.Row | None:
    """Devuelve la fila de una corrida, o ``None`` si no existe."""
    with get_connection(db_path) as conn:
        return conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()


def init_unstructured_db(db_path: Path | None = None) -> None:
    """Crea ``resultados_modelos``, ``configuracion`` y ``predicciones`` si faltan.

    Las tres guardan un documento JSON por fila (``doc``), que se consulta con
    ``json_extract``. El ``run_id`` las conecta con ``structured.db``. Es
    seguro llamarla sobre una base vieja: agrega las tablas que falten sin
    tocar los datos.

    Args:
        db_path: ruta de la base; por defecto ``unstructured.db``.
    """
    path = db_path if db_path is not None else config.UNSTRUCTURED_DB_PATH
    with get_connection(path) as conn:
        conn.executescript(SCHEMA_UNSTRUCTURED)


def init_databases(
    structured_path: Path | None = None, unstructured_path: Path | None = None
) -> None:
    """Crea las dos bases con todas sus tablas vacías si todavía no existen.

    Es lo que hace la app y la línea de comandos al arrancar: quien clona el
    proyecto no recibe ningún ``.db`` (están en el ``.gitignore``) y las bases
    aparecen solas en ``db/`` con su esquema completo. Es idempotente: sobre
    bases que ya existen no borra ni modifica nada.

    Args:
        structured_path: ruta de ``structured.db``; por defecto la de ``config``.
        unstructured_path: ruta de ``unstructured.db``; por defecto la de
            ``config``.
    """
    init_structured_db(structured_path)
    init_unstructured_db(unstructured_path)


def attach_unstructured(
    conn: sqlite3.Connection,
    unstructured_path: Path | None = None,
    alias: str = "unstructured",
) -> None:
    """Adjunta ``unstructured.db`` a una conexión de ``structured.db``.

    Permite consultas cruzadas por ``run_id``, por ejemplo unir ``runs`` con
    ``unstructured.resultados_modelos``.

    Args:
        conn: conexión abierta sobre ``structured.db``.
        unstructured_path: ruta de la base; por defecto la de ``config``.
        alias: nombre con el que se consulta la base adjunta.
    """
    path = unstructured_path or config.UNSTRUCTURED_DB_PATH
    conn.execute(f"ATTACH DATABASE ? AS {alias}", (str(path),))
