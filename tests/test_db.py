"""Tests de src/db.py."""

import sqlite3
import time
from pathlib import Path

import pytest
from support import (
    STRUCTURED_TABLES,
    UNSTRUCTURED_TABLES,
    count_rows,
    main_file_only,
    user_tables,
)

from src import config, db


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Ruta de una base temporal."""
    return tmp_path / "test.db"


def test_init_creates_tables(db_path: Path) -> None:
    """init_structured_db crea runs y los dos datasets, y ya no crea logs (van a MongoDB)."""
    db.init_structured_db(db_path)
    with db.get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    names = {r["name"] for r in rows}
    assert {"runs", "datos_originales", "datos_limpios"} <= names
    assert "logs" not in names


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


def insert_prediction(path: Path, run_id: str, doc: str) -> None:
    """Inserta una fila en predicciones."""
    with db.get_connection(path) as conn:
        conn.execute(
            "INSERT INTO predicciones (run_id, created_at, doc) VALUES (?, 'now', ?)",
            (run_id, doc),
        )


def test_predicciones_table_exists_after_init(udb_path: Path) -> None:
    """init_unstructured_db crea también la colección predicciones."""
    with db.get_connection(udb_path) as conn:
        names = {
            r["name"]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert "predicciones" in names


def test_predicciones_allows_many_per_run_and_reads_nested_json(udb_path: Path) -> None:
    """Una corrida puede tener varias predicciones; se leen con json_extract."""
    doc = (
        '{"inputs": {"fecha": "2026-09-01", "temperatura": 15.4},'
        ' "models": {"xgboost": {"probabilidad": 0.9, "clase": 1}}}'
    )
    insert_prediction(udb_path, "r1", doc)
    insert_prediction(udb_path, "r1", doc)
    with db.get_connection(udb_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM predicciones").fetchone()[0] == 2
        row = conn.execute(
            "SELECT json_extract(doc, '$.inputs.fecha') AS fecha, "
            "json_extract(doc, '$.models.xgboost.probabilidad') AS p "
            "FROM predicciones WHERE run_id = 'r1' ORDER BY id LIMIT 1"
        ).fetchone()
    assert (row["fecha"], row["p"]) == ("2026-09-01", 0.9)


def test_predicciones_rejects_invalid_json(udb_path: Path) -> None:
    """Un documento que no es JSON no se puede guardar."""
    with pytest.raises(sqlite3.IntegrityError):
        insert_prediction(udb_path, "r1", "no es json")


def test_init_adds_predicciones_to_an_old_database_without_losing_data(
    tmp_path: Path,
) -> None:
    """Una base creada antes de esta tabla la recibe y conserva sus datos."""
    path = tmp_path / "old.db"
    with db.get_connection(path) as conn:
        conn.executescript(
            "CREATE TABLE resultados_modelos (id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " run_id TEXT NOT NULL, scenario TEXT NOT NULL, model TEXT NOT NULL,"
            " created_at TEXT NOT NULL, doc TEXT NOT NULL CHECK (json_valid(doc)),"
            " UNIQUE (run_id, scenario, model));"
            "INSERT INTO resultados_modelos (run_id, scenario, model, created_at, doc)"
            " VALUES ('r1', 'A', 'xgboost', 'now', '{\"f1\": 0.5}');"
        )
    db.init_unstructured_db(path)
    with db.get_connection(path) as conn:
        tables = {
            r["name"]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        kept = conn.execute(
            "SELECT json_extract(doc, '$.f1') FROM resultados_modelos"
        ).fetchone()[0]
    assert {"predicciones", "configuracion", "resultados_modelos"} <= tables
    assert kept == 0.5


def write_events(path: Path, n: int, start: int = 0) -> None:
    """Escribe n filas en ``datos_originales``, cada una con su propia conexión."""
    for i in range(start, start + n):
        with db.get_connection(path) as conn:
            conn.execute(
                "INSERT INTO datos_originales (indice_tiempo, demanda_total) "
                "VALUES (?, ?)",
                (f"mes {i:03d}", float(i)),
            )


def test_connection_is_wal_with_normal_sync_and_no_checkpoint_on_close(
    db_path: Path,
) -> None:
    """Cada conexión usa WAL, synchronous=NORMAL y no hace checkpoint al cerrar."""
    db.init_structured_db(db_path)
    with db.get_connection(db_path) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 1  # NORMAL
        assert conn.getconfig(sqlite3.SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE) is True


def test_events_from_many_short_connections_are_all_readable(db_path: Path) -> None:
    """Lo escrito con una conexión por evento lo ve cualquier conexión nueva."""
    db.init_structured_db(db_path)
    write_events(db_path, 25)
    with db.get_connection(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM datos_originales").fetchone()[0] == 25
        last = conn.execute(
            "SELECT indice_tiempo FROM datos_originales ORDER BY demanda_total DESC"
        ).fetchone()
    assert last[0] == "mes 024"


def test_main_file_alone_is_incomplete_until_the_checkpoint(
    db_path: Path, tmp_path: Path
) -> None:
    """Sin checkpoint, copiar solo el .db pierde lo reciente; con checkpoint, no."""
    db.init_structured_db(db_path)
    write_events(db_path, 10)
    before = main_file_only(db_path, tmp_path / "antes")
    assert (
        count_rows(before, "datos_originales") or 0
    ) < 10  # lo reciente vive en el -wal

    assert db.checkpoint(db_path) is True
    after = main_file_only(db_path, tmp_path / "despues")
    assert count_rows(after, "datos_originales") == 10
    wal = Path(f"{db_path}-wal")
    assert not wal.exists() or wal.stat().st_size == 0


def test_checkpoint_keeps_the_data_and_the_database_usable(db_path: Path) -> None:
    """Después del checkpoint se puede seguir escribiendo y leyendo en WAL."""
    db.init_structured_db(db_path)
    write_events(db_path, 5)
    assert db.checkpoint(db_path) is True
    write_events(db_path, 3, start=5)
    with db.get_connection(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM datos_originales").fetchone()[0] == 8
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_checkpoint_of_a_missing_database_creates_nothing(tmp_path: Path) -> None:
    """Sin archivo no hay nada que volcar, y no se crea ninguno."""
    missing = tmp_path / "nada.db"
    assert db.checkpoint(missing) is False
    assert not missing.exists()
    assert db.checkpoint_all(missing, tmp_path / "tampoco.db") is False
    assert list(tmp_path.iterdir()) == []


def test_checkpoint_all_flushes_both_databases(tmp_path: Path) -> None:
    """checkpoint_all deja completos los dos archivos."""
    structured, unstructured = tmp_path / "s.db", tmp_path / "u.db"
    db.init_structured_db(structured)
    db.init_unstructured_db(unstructured)
    write_events(structured, 4)
    insert_prediction(unstructured, "r1", '{"a": 1}')
    assert db.checkpoint_all(structured, unstructured) is True
    assert (
        count_rows(main_file_only(structured, tmp_path / "c"), "datos_originales") == 4
    )
    assert count_rows(main_file_only(unstructured, tmp_path / "c"), "predicciones") == 1


def test_checkpoint_is_best_effort_when_another_reader_holds_the_database(
    db_path: Path, tmp_path: Path
) -> None:
    """Con un lector activo no lanza ni se cuelga: devuelve False y no pierde datos."""
    db.init_structured_db(db_path)
    write_events(db_path, 3)
    reader = sqlite3.connect(db_path)
    reader.execute("BEGIN")
    reader.execute(
        "SELECT * FROM datos_originales"
    ).fetchone()  # fija una vista de la base
    try:
        write_events(db_path, 3, start=3)
        started = time.perf_counter()
        assert db.checkpoint(db_path, timeout=0.2) is False
        assert time.perf_counter() - started < 5
        with db.get_connection(db_path) as conn:
            assert (
                conn.execute("SELECT COUNT(*) FROM datos_originales").fetchone()[0] == 6
            )
    finally:
        reader.close()
    assert db.checkpoint(db_path) is True
    assert (
        count_rows(main_file_only(db_path, tmp_path / "final"), "datos_originales") == 6
    )


def test_rollback_still_discards_the_write_with_the_new_settings(db_path: Path) -> None:
    """Un bloque que falla no deja nada, aunque no haya checkpoint al cerrar."""
    db.init_structured_db(db_path)
    with pytest.raises(RuntimeError), db.get_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO datos_originales (indice_tiempo, demanda_total) "
            "VALUES ('x', 1.0)"
        )
        raise RuntimeError("falla")
    assert db.checkpoint(db_path) is True
    assert count_rows(db_path, "datos_originales") == 0


def test_checkpoint_of_an_unreadable_file_is_best_effort(tmp_path: Path) -> None:
    """Un archivo que no es una base SQLite no hace fallar el final de una corrida."""
    broken = tmp_path / "roto.db"
    broken.write_bytes(b"esto no es una base de datos sqlite" * 50)
    assert db.checkpoint(broken) is False
    assert broken.read_bytes().startswith(b"esto no es")  # no se tocó


def test_init_databases_creates_both_files_with_every_table_empty(
    tmp_path: Path,
) -> None:
    """En una carpeta vacía (ni siquiera existe db/) aparecen las dos bases vacías."""
    structured = tmp_path / "db" / "structured.db"
    unstructured = tmp_path / "db" / "unstructured.db"
    assert not structured.parent.exists()
    db.init_databases(structured, unstructured)
    assert user_tables(structured) == dict.fromkeys(STRUCTURED_TABLES, 0)
    assert user_tables(unstructured) == dict.fromkeys(UNSTRUCTURED_TABLES, 0)


def test_init_databases_is_idempotent_and_keeps_existing_data(tmp_path: Path) -> None:
    """Sobre bases que ya existen no borra ni modifica nada."""
    structured, unstructured = tmp_path / "s.db", tmp_path / "u.db"
    db.init_databases(structured, unstructured)
    run_id = db.create_run(seed=7, db_path=structured)
    insert_prediction(unstructured, run_id, '{"a": 1}')
    db.init_databases(structured, unstructured)
    db.init_databases(structured, unstructured)
    assert user_tables(structured)["runs"] == 1
    assert user_tables(unstructured)["predicciones"] == 1
    assert db.get_run(run_id, structured)["seed"] == 7


def test_init_databases_defaults_to_the_paths_in_config(tmp_config: Path) -> None:
    """Sin argumentos usa las rutas de config (aquí apuntadas a una carpeta temporal)."""
    db.init_databases()
    assert config.STRUCTURED_DB_PATH.is_file() and config.UNSTRUCTURED_DB_PATH.is_file()
    assert set(user_tables(config.STRUCTURED_DB_PATH)) == STRUCTURED_TABLES
    assert set(user_tables(config.UNSTRUCTURED_DB_PATH)) == UNSTRUCTURED_TABLES
