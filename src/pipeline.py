"""Ejecución del pipeline por línea de comandos: ``python -m src.pipeline``."""

import logging
from collections.abc import Callable
from pathlib import Path

from src import data, db, features, logger, results, train

log = logging.getLogger(__name__)


@logger.log_step
def run_etl_steps(csv_path: Path | None, db_path: Path | None) -> int:
    """Carga, limpia, guarda, deriva variables y target. Devuelve filas guardadas."""
    df = data.load_raw(csv_path)
    data.check_sector_nulls(df)
    cleaned = data.flag_repeated_power(df)
    log.info("Filas completas tras la limpieza: %d", data.count_complete_rows(cleaned))
    rows = data.save_clean_data(cleaned, db_path)
    dataset = features.build_dataset(data.load_clean_data(db_path))
    _, y = features.get_model_data(dataset, "A")
    features.log_class_balance(y)
    return rows


@logger.log_step
def run_training_steps(db_path: Path | None) -> list[train.ModelRun]:
    """Entrena los 3 modelos en los escenarios A y B desde ``datos_limpios``."""
    dataset = features.build_dataset(data.load_clean_data(db_path))
    return train.train_all(dataset)


@logger.log_step
def run_saving_steps(
    run_id: str,
    runs: list[train.ModelRun],
    csv_path: Path | None,
    unstructured_path: Path | None,
    models_dir: Path | None,
) -> None:
    """Guarda la configuración y cada resultado (con su modelo) de la corrida."""
    results.save_run_config(
        run_id, results.build_run_config(csv_path), unstructured_path
    )
    for run in runs:
        results.save_model_run(run_id, run, models_dir, unstructured_path)


def _run_registered(steps: Callable[[], object], db_path: Path | None) -> str:
    """Ejecuta ``steps`` como una corrida registrada en ``runs`` y ``logs``.

    Crea el ``run_id``, lo deja activo para el logger y cierra la corrida con
    estado ``ok`` o ``error`` (relanzando el error).
    """
    logger.setup_logging(db_path)
    db.init_structured_db(db_path)
    run_id = db.create_run(db_path=db_path)
    token = logger.current_run_id.set(run_id)
    try:
        steps()
        db.finish_run(run_id, "ok", db_path)
        log.info("Corrida terminada: run_id=%s", run_id)
    except Exception:
        db.finish_run(run_id, "error", db_path)
        raise
    finally:
        logger.current_run_id.reset(token)
    return run_id


def run_etl(csv_path: Path | None = None, db_path: Path | None = None) -> str:
    """Ejecuta solo el ETL como una corrida registrada.

    Args:
        csv_path: CSV de entrada; por defecto el de ``data/raw/``.
        db_path: ruta de ``structured.db``; por defecto la de ``config``.

    Returns:
        El ``run_id`` de la corrida.
    """
    return _run_registered(lambda: run_etl_steps(csv_path, db_path), db_path)


def run_pipeline(
    csv_path: Path | None = None,
    db_path: Path | None = None,
    unstructured_path: Path | None = None,
    models_dir: Path | None = None,
) -> str:
    """Ejecuta ETL, entrenamiento y guardado en una misma corrida registrada.

    Los resultados, la configuración y los modelos quedan atados al mismo
    ``run_id`` que ``runs`` y ``logs``.

    Args:
        csv_path: CSV de entrada; por defecto el de ``data/raw/``.
        db_path: ruta de ``structured.db``; por defecto la de ``config``.
        unstructured_path: ruta de ``unstructured.db``; por defecto la de
            ``config``.
        models_dir: carpeta de los ``.joblib``; por defecto ``config.MODELS_DIR``.

    Returns:
        El ``run_id`` de la corrida.
    """

    def steps() -> None:
        run_etl_steps(csv_path, db_path)
        runs = run_training_steps(db_path)
        run_id = logger.current_run_id.get()
        run_saving_steps(run_id, runs, csv_path, unstructured_path, models_dir)

    return _run_registered(steps, db_path)


def main() -> None:
    """Punto de entrada: corre el ETL, entrena y guarda; muestra el log."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    run_pipeline()


if __name__ == "__main__":
    # Con ``python -m`` este módulo se llama ``__main__`` y sus eventos no
    # cuelgan del logger ``src``. Se usa la versión importada para que queden
    # registrados en ``logs``.
    from src.pipeline import main as _main

    _main()
