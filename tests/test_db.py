"""Tests de src/db.py."""

import sqlite3
from pathlib import Path

import pytest

from src import db


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Ruta de una base temporal."""
    return tmp_path / "test.db"


def test_init_creates_tables(db_path: Path) -> None:
    """init_structured_db crea runs y logs."""
    db.init_structured_db(db_path)
    with db.get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    names = {r["name"] for r in rows}
    assert {"runs", "logs"} <= names


def test_init_is_idempotent(db_path: Path) -> None:
    """Llamar dos veces no falla ni borra datos."""
    db.init_structured_db(db_path)
    with db.get_connection(db_path) as conn:
        conn.execute("INSERT INTO runs (run_id, started_at) VALUES ('a', 'now')")
    db.init_structured_db(db_path)
    with db.get_connection(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    assert count == 1


def test_wal_mode_enabled(db_path: Path) -> None:
    """La base queda en modo WAL."""
    db.init_structured_db(db_path)
    with db.get_connection(db_path) as conn:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode == "wal"


def test_rollback_on_error(db_path: Path) -> None:
    """Si el bloque falla, no se guarda nada."""
    db.init_structured_db(db_path)
    with pytest.raises(RuntimeError), db.get_connection(db_path) as conn:
        conn.execute("INSERT INTO runs (run_id, started_at) VALUES ('a', 'now')")
        raise RuntimeError("falla")
    with db.get_connection(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    assert count == 0


def test_connection_closed_after_block(db_path: Path) -> None:
    """La conexión se cierra al salir del bloque."""
    with db.get_connection(db_path) as conn:
        pass
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")
