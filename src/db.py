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
"""


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
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        yield conn
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_structured_db(db_path: Path | None = None) -> None:
    """Crea las tablas ``runs`` y ``logs`` si no existen.

    Args:
        db_path: ruta de la base; por defecto ``structured.db``.
    """
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA_STRUCTURED)


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
