"""Ejecución del pipeline por línea de comandos: ``python -m src.pipeline``."""

import argparse
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


def _non_negative_int(text: str) -> int:
    """Convierte un argumento de la línea de comandos en un entero no negativo."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"no es un número entero: {text!r}") from None
    if value < 0:
        raise argparse.ArgumentTypeError(f"no puede ser negativo: {value}")
    return value


def build_parser() -> argparse.ArgumentParser:
    """Arma el parser de la línea de comandos de ``python -m src.pipeline``."""
    parser = argparse.ArgumentParser(
        prog="python -m src.pipeline",
        add_help=False,
        description=(
            "Ejecuta el ETL, entrena los tres modelos en los escenarios A y B y "
            "guarda los resultados como una corrida registrada en las bases."
        ),
    )
    parser.add_argument(
        "-h", "--help", action="help", help="muestra esta ayuda y termina"
    )
    parser.add_argument(
        "--seed",
        type=_non_negative_int,
        default=config.SEED,
        help=f"semilla de la corrida (por defecto {config.SEED}, la de config.py)",
    )
    parser.add_argument(
        "--solo-etl",
        action="store_true",
        help="solo ejecuta el ETL (carga, limpieza y datos_limpios), sin entrenar",
    )
    parser.add_argument(
        "--reporte",
        action="store_true",
        help="al terminar, genera el PDF de la corrida en reports/",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada: corre la corrida pedida y muestra el log por consola.

    Args:
        argv: argumentos de la línea de comandos; por defecto ``sys.argv[1:]``.

    Returns:
        Código de salida: 0 si la corrida terminó bien y 1 si falló.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.solo_etl and args.reporte:
        parser.error("--reporte necesita entrenar: no se puede usar con --solo-etl")
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    db.init_databases()  # en el primer arranque de un clon crea las bases vacías
    try:
        run_id = (
            run_etl(seed=args.seed) if args.solo_etl else run_pipeline(seed=args.seed)
        )
        if args.reporte:
            from src import report  # solo si se pide: carga ReportLab y matplotlib

            log.info("Reporte guardado en %s", report.save_report(run_id))
    except Exception as error:  # noqa: BLE001 - el detalle ya quedó en logs
        log.error("La corrida falló: %s: %s", type(error).__name__, error)
        return 1
    return 0


if __name__ == "__main__":
    # Con ``python -m`` este módulo se llama ``__main__`` y sus eventos no
    # cuelgan del logger ``src``. Se usa la versión importada para que queden
    # registrados en ``logs``.
    from src.pipeline import main as _main

    raise SystemExit(_main())
