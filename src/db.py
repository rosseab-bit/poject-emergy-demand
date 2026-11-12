"""Acceso a las bases SQLite del proyecto.

Cada operación abre su propia conexión (los hilos de Streamlit no las
comparten) en modo WAL y la cierra al terminar.
"""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
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
