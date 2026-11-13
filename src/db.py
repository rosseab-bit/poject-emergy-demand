"""Acceso a la base SQLite del proyecto (``structured.db``).

Cada operación abre su propia conexión (los hilos de Streamlit no las
comparten) en modo WAL y la cierra al terminar.
"""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from src import config

SCHEMA_STRUCTURED = """
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


def init_structured_db(db_path: Path | None = None) -> None:
    """Crea ``datos_originales`` y ``datos_limpios`` si faltan.

    Args:
        db_path: ruta de la base; por defecto ``structured.db``.
    """
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA_STRUCTURED + SCHEMA_RAW)
