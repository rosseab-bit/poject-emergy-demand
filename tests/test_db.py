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


@pytest.fixture
def udb_path(tmp_path: Path) -> Path:
    """Ruta de una base no estructurada temporal."""
    path = tmp_path / "unstructured.db"
    db.init_unstructured_db(path)
    return path


def insert_result(path: Path, run_id: str, scenario: str, model: str, doc: str) -> None:
    """Inserta una fila en resultados_modelos."""
    with db.get_connection(path) as conn:
        conn.execute(
            "INSERT INTO resultados_modelos (run_id, scenario, model, created_at, doc) "
            "VALUES (?, ?, ?, 'now', ?)",
            (run_id, scenario, model, doc),
        )


def test_init_unstructured_creates_tables_and_wal(udb_path: Path) -> None:
    """Se crean las dos tablas y la base queda en modo WAL."""
    with db.get_connection(udb_path) as conn:
        names = {
            r["name"]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert {"resultados_modelos", "configuracion"} <= names
    assert mode == "wal"


def test_init_unstructured_is_idempotent(udb_path: Path) -> None:
    """Llamar dos veces no borra datos."""
    insert_result(udb_path, "r1", "A", "xgboost", '{"f1": 0.5}')
    db.init_unstructured_db(udb_path)
    with db.get_connection(udb_path) as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM resultados_modelos").fetchone()[0] == 1
        )


def test_json_extract_reads_nested_values(udb_path: Path) -> None:
    """Se puede recuperar un valor anidado con json_extract."""
    doc = '{"holdout": {"f1": 0.44}, "cv": {"scores": [0.1, 0.2, 0.3]}}'
    insert_result(udb_path, "r1", "A", "logistic_regression", doc)
    with db.get_connection(udb_path) as conn:
        row = conn.execute(
            "SELECT json_extract(doc, '$.holdout.f1') AS f1, "
            "json_extract(doc, '$.cv.scores[2]') AS last_cv "
            "FROM resultados_modelos WHERE run_id = 'r1' AND scenario = 'A'"
        ).fetchone()
    assert row["f1"] == 0.44
    assert row["last_cv"] == 0.3


def test_invalid_json_is_rejected(udb_path: Path) -> None:
    """Un documento que no es JSON válido no se puede guardar."""
    with pytest.raises(sqlite3.IntegrityError):
        insert_result(udb_path, "r1", "A", "xgboost", "esto no es json")
    with pytest.raises(sqlite3.IntegrityError), db.get_connection(udb_path) as conn:
        conn.execute(
            "INSERT INTO configuracion (run_id, created_at, doc) VALUES ('r1', 'now', '{')"
        )


def test_result_is_unique_per_run_scenario_and_model(udb_path: Path) -> None:
    """No se repite (run_id, escenario, modelo), pero sí varía cualquiera."""
    insert_result(udb_path, "r1", "A", "xgboost", "{}")
    insert_result(udb_path, "r1", "B", "xgboost", "{}")
    insert_result(udb_path, "r2", "A", "xgboost", "{}")
    with pytest.raises(sqlite3.IntegrityError):
        insert_result(udb_path, "r1", "A", "xgboost", "{}")


def test_one_configuration_per_run(udb_path: Path) -> None:
    """configuracion tiene una sola fila por run_id."""
    with db.get_connection(udb_path) as conn:
        conn.execute("INSERT INTO configuracion VALUES ('r1', 'now', '{}')")
    with pytest.raises(sqlite3.IntegrityError), db.get_connection(udb_path) as conn:
        conn.execute("INSERT INTO configuracion VALUES ('r1', 'now', '{}')")
