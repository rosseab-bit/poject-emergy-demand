"""Ejecución del pipeline por línea de comandos: ``python -m src.pipeline``."""

import logging
from collections.abc import Callable
from pathlib import Path

from src import config, data, db, features, logger, models, results, train

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
def run_training_steps(
    db_path: Path | None,
    seed: int = config.SEED,
    scenarios: tuple[str, ...] = ("A", "B"),
    model_names: tuple[str, ...] = models.MODEL_NAMES,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> list[train.ModelRun]:
    """Entrena los modelos y escenarios pedidos desde ``datos_limpios``."""
    dataset = features.build_dataset(data.load_clean_data(db_path))
    return train.train_all(dataset, scenarios, model_names, seed, on_progress)


@logger.log_step
def run_saving_steps(
    run_id: str,
    runs: list[train.ModelRun],
    csv_path: Path | None,
    unstructured_path: Path | None,
    models_dir: Path | None,
    seed: int = config.SEED,
) -> None:
    """Guarda la configuración y cada resultado (con su modelo) de la corrida."""
    results.save_run_config(
        run_id, results.build_run_config(csv_path, seed), unstructured_path
    )
    for run in runs:
        results.save_model_run(run_id, run, models_dir, unstructured_path)


def _run_registered(
    steps: Callable[[], object],
    db_path: Path | None,
    seed: int = config.SEED,
    run_id: str | None = None,
    unstructured_path: Path | None = None,
    with_unstructured: bool = False,
) -> str:
    """Ejecuta ``steps`` como una corrida registrada en ``runs`` y ``logs``.

    Crea el ``run_id`` (o continúa el que se pasa), lo deja activo para el
    logger y cierra la corrida con estado ``ok`` o ``error`` (relanzando el
    error). Al terminar vuelca el WAL de las bases al archivo principal, así los
    ``.db`` quedan completos por sí solos; ``with_unstructured`` indica si la
    corrida también escribió en ``unstructured.db``.
    """
    logger.setup_logging(db_path)
    db.init_structured_db(db_path)
    if run_id is None:
        run_id = db.create_run(seed=seed, db_path=db_path)
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
        if with_unstructured:
            db.checkpoint_all(db_path, unstructured_path)
        else:
            db.checkpoint(db_path)
    return run_id


def run_etl(
    csv_path: Path | None = None,
    db_path: Path | None = None,
    seed: int = config.SEED,
) -> str:
    """Ejecuta solo el ETL como una corrida registrada.

    Args:
        csv_path: CSV de entrada; por defecto el de ``data/raw/``.
        db_path: ruta de ``structured.db``; por defecto la de ``config``.
        seed: semilla de la corrida; por defecto ``config.SEED``.

    Returns:
        El ``run_id`` de la corrida.
    """
    return _run_registered(lambda: run_etl_steps(csv_path, db_path), db_path, seed)


def run_training(
    run_id: str,
    db_path: Path | None = None,
    unstructured_path: Path | None = None,
    models_dir: Path | None = None,
    csv_path: Path | None = None,
    scenarios: tuple[str, ...] = ("A", "B"),
    model_names: tuple[str, ...] = models.MODEL_NAMES,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> str:
    """Entrena y guarda los resultados dentro de una corrida ya iniciada.

    Continúa el ``run_id`` que dejó el ETL: usa la semilla de esa corrida y
    deja el entrenamiento, los resultados y los modelos atados al mismo
    ``run_id`` en ``runs``, ``logs`` y ``unstructured.db``.

    Args:
        run_id: corrida existente, con el ETL ya ejecutado.
        db_path: ruta de ``structured.db``; por defecto la de ``config``.
        unstructured_path: ruta de ``unstructured.db``; por defecto la de
            ``config``.
        models_dir: carpeta de los ``.joblib``; por defecto ``config.MODELS_DIR``.
        csv_path: CSV original (para el hash de la configuración).
        scenarios: escenarios a entrenar.
        model_names: modelos a entrenar.
        on_progress: función opcional ``(terminados, total, etiqueta)``.

    Returns:
        El mismo ``run_id``.

    Raises:
        ValueError: si la corrida no existe, no tiene el ETL hecho o ya tiene
            resultados (para volver a entrenar hay que iniciar otra corrida).
    """
    db.init_structured_db(db_path)
    run = db.get_run(run_id, db_path)
    if run is None:
        raise ValueError(f"No existe la corrida {run_id}")
    if data.load_clean_data(db_path).empty:
        raise ValueError("Primero ejecutá el ETL: datos_limpios está vacía")
    if results.has_results(run_id, unstructured_path):
        raise ValueError(f"La corrida {run_id} ya tiene resultados; iniciá otra")
    seed = run["seed"]

    def steps() -> None:
        runs = run_training_steps(db_path, seed, scenarios, model_names, on_progress)
        run_saving_steps(run_id, runs, csv_path, unstructured_path, models_dir, seed)

    return _run_registered(
        steps,
        db_path,
        seed,
        run_id=run_id,
        unstructured_path=unstructured_path,
        with_unstructured=True,
    )


def run_pipeline(
    csv_path: Path | None = None,
    db_path: Path | None = None,
    unstructured_path: Path | None = None,
    models_dir: Path | None = None,
    seed: int = config.SEED,
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
        seed: semilla de la corrida; por defecto ``config.SEED``.

    Returns:
        El ``run_id`` de la corrida.
    """

    def steps() -> None:
        run_etl_steps(csv_path, db_path)
        runs = run_training_steps(db_path, seed)
        run_id = logger.current_run_id.get()
        run_saving_steps(run_id, runs, csv_path, unstructured_path, models_dir, seed)

    return _run_registered(
        steps,
        db_path,
        seed,
        unstructured_path=unstructured_path,
        with_unstructured=True,
    )


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
