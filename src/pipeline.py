"""Ejecución del pipeline por línea de comandos: ``python -m src.pipeline``."""

import logging
from pathlib import Path

from src import data, db, logger

log = logging.getLogger(__name__)


@logger.log_step
def run_etl_steps(csv_path: Path | None, db_path: Path | None) -> int:
    """Carga, limpia y guarda los datos. Devuelve las filas guardadas."""
    df = data.load_raw(csv_path)
    data.check_sector_nulls(df)
    cleaned = data.flag_repeated_power(df)
    log.info("Filas completas tras la limpieza: %d", data.count_complete_rows(cleaned))
    return data.save_clean_data(cleaned, db_path)


def run_etl(csv_path: Path | None = None, db_path: Path | None = None) -> str:
    """Ejecuta el ETL completo como una corrida registrada.

    Crea el ``run_id``, lo deja activo para el logger, corre los pasos y cierra
    la corrida con estado ``ok`` o ``error`` (relanzando el error).

    Args:
        csv_path: CSV de entrada; por defecto el de ``data/raw/``.
        db_path: ruta de ``structured.db``; por defecto la de ``config``.

    Returns:
        El ``run_id`` de la corrida.
    """
    logger.setup_logging(db_path)
    db.init_structured_db(db_path)
    run_id = db.create_run(db_path=db_path)
    token = logger.current_run_id.set(run_id)
    try:
        run_etl_steps(csv_path, db_path)
        db.finish_run(run_id, "ok", db_path)
        log.info("Corrida terminada: run_id=%s", run_id)
    except Exception:
        db.finish_run(run_id, "error", db_path)
        raise
    finally:
        logger.current_run_id.reset(token)
    return run_id


def main() -> None:
    """Punto de entrada: corre el ETL y muestra el log por consola."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    run_etl()


if __name__ == "__main__":
    # Con ``python -m`` este módulo se llama ``__main__`` y sus eventos no
    # cuelgan del logger ``src``. Se usa la versión importada para que queden
    # registrados en ``logs``.
    from src.pipeline import main as _main

    _main()
