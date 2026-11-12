"""Tests de src/db.py."""

import sqlite3
from pathlib import Path

import pytest

from src import config, db


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


def test_create_run_starts_running(db_path: Path) -> None:
    """create_run guarda la corrida en estado running con la semilla."""
    db.init_structured_db(db_path)
    run_id = db.create_run(seed=7, notes="prueba", db_path=db_path)
    run = db.get_run(run_id, db_path)
    assert run is not None
    assert run["status"] == "running"
    assert run["seed"] == 7
    assert run["notes"] == "prueba"
    assert run["finished_at"] is None


def test_create_run_default_seed_from_config(db_path: Path) -> None:
    """Sin semilla explícita se usa la de config."""
    db.init_structured_db(db_path)
    run_id = db.create_run(db_path=db_path)
    assert db.get_run(run_id, db_path)["seed"] == config.SEED


def test_run_ids_are_short_and_unique() -> None:
    """Los run_id tienen 8 caracteres hex y no se repiten."""
    ids = {db.new_run_id() for _ in range(100)}
    assert len(ids) == 100
    assert all(len(i) == 8 and int(i, 16) >= 0 for i in ids)


def test_finish_run_sets_status(db_path: Path) -> None:
    """finish_run guarda estado final y fecha de cierre."""
    db.init_structured_db(db_path)
    run_id = db.create_run(db_path=db_path)
    db.finish_run(run_id, "ok", db_path)
    run = db.get_run(run_id, db_path)
    assert run["status"] == "ok"
    assert run["finished_at"] is not None


def test_finish_run_rejects_bad_input(db_path: Path) -> None:
    """Estado inválido o run_id inexistente lanzan ValueError."""
    db.init_structured_db(db_path)
    run_id = db.create_run(db_path=db_path)
    with pytest.raises(ValueError):
        db.finish_run(run_id, "raro", db_path)
    with pytest.raises(ValueError):
        db.finish_run("noexiste", "ok", db_path)


def test_get_run_missing_returns_none(db_path: Path) -> None:
    """get_run devuelve None si no existe."""
    db.init_structured_db(db_path)
    assert db.get_run("noexiste", db_path) is None
