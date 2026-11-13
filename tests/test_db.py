"""Tests de src/db.py."""

import sqlite3
import time
from pathlib import Path

import pytest
from support import (
    STRUCTURED_TABLES,
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
    """init_structured_db crea solo los dos datasets: runs y logs van a MongoDB."""
    db.init_structured_db(db_path)
    with db.get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    names = {r["name"] for r in rows}
    assert {"datos_originales", "datos_limpios"} <= names
    assert not {"runs", "logs"} & names  # van a MongoDB


def test_init_is_idempotent(db_path: Path) -> None:
    """Llamar dos veces no falla ni borra datos."""
    db.init_structured_db(db_path)
    with db.get_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO datos_originales (indice_tiempo, demanda_total) VALUES ('a', 1.0)"
        )
    db.init_structured_db(db_path)
    with db.get_connection(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM datos_originales").fetchone()[0]
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
        conn.execute(
            "INSERT INTO datos_originales (indice_tiempo, demanda_total) VALUES ('a', 1.0)"
        )
        raise RuntimeError("falla")
    with db.get_connection(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM datos_originales").fetchone()[0]
    assert count == 0


def test_connection_closed_after_block(db_path: Path) -> None:
    """La conexión se cierra al salir del bloque."""
    with db.get_connection(db_path) as conn:
        pass
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")


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
    assert list(tmp_path.iterdir()) == []


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


def test_init_structured_db_creates_the_file_with_every_table_empty(
    tmp_path: Path,
) -> None:
    """En una carpeta vacía (ni siquiera existe db/) aparece la base vacía."""
    structured = tmp_path / "db" / "structured.db"
    assert not structured.parent.exists()
    db.init_structured_db(structured)
    assert user_tables(structured) == dict.fromkeys(STRUCTURED_TABLES, 0)
    assert not list(structured.parent.glob("unstructured*"))  # ya no existe esa base


def test_init_structured_db_is_idempotent_and_keeps_existing_data(
    tmp_path: Path,
) -> None:
    """Sobre una base que ya existe no borra ni modifica nada."""
    structured = tmp_path / "s.db"
    db.init_structured_db(structured)
    write_events(structured, 3)
    db.init_structured_db(structured)
    db.init_structured_db(structured)
    assert user_tables(structured)["datos_originales"] == 3


def test_init_structured_db_defaults_to_the_path_in_config(tmp_config: Path) -> None:
    """Sin argumentos usa la ruta de config (aquí apuntada a una carpeta temporal)."""
    db.init_structured_db()
    assert config.STRUCTURED_DB_PATH.is_file()
    assert set(user_tables(config.STRUCTURED_DB_PATH)) == STRUCTURED_TABLES
