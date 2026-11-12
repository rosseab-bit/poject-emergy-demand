"""Datos de reproducibilidad y guardado de resultados en ``unstructured.db``."""

import hashlib
import json
import logging
import platform
import subprocess
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import config, db, features, models
from src.logger import log_step
from src.train import EvalResult, ModelRun

logger = logging.getLogger(__name__)

TRACKED_PACKAGES = (
    "pandas",
    "numpy",
    "scikit-learn",
    "xgboost",
    "streamlit",
    "reportlab",
    "matplotlib",
    "joblib",
)
GIT_TIMEOUT_SECONDS = 10


def file_sha256(path: Path) -> str:
    """Calcula el SHA-256 de un archivo, leyéndolo por bloques.

    Args:
        path: ruta del archivo.

    Returns:
        El hash en hexadecimal (64 caracteres).
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def library_versions(packages: tuple[str, ...] = TRACKED_PACKAGES) -> dict[str, str]:
    """Devuelve las versiones de Python y de las librerías del proyecto.

    Args:
        packages: nombres de paquetes a consultar.

    Returns:
        Diccionario nombre a versión; ``"no instalado"`` si falta alguna.
    """
    versions = {"python": platform.python_version()}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "no instalado"
    return versions


def _git(args: list[str], repo_dir: Path) -> str | None:
    """Ejecuta un comando de git de solo lectura; ``None`` si no se puede."""
    try:
        done = subprocess.run(
            ["git", *args],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def git_info(repo_dir: Path | None = None) -> dict[str, str | bool | None]:
    """Devuelve el hash del commit actual y si hay cambios sin commitear.

    Usa solo ``git log`` y ``git status`` (lectura). Si no hay git o la
    carpeta no es un repositorio, ambos valores son ``None``.

    Args:
        repo_dir: carpeta del repositorio; por defecto la raíz del proyecto.

    Returns:
        ``{"commit": <hash o None>, "dirty": <bool o None>}``.
    """
    repo = repo_dir if repo_dir is not None else config.ROOT_DIR
    commit = _git(["log", "-1", "--format=%H"], repo)
    status = _git(["status", "--porcelain"], repo)
    if not commit:
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": None if status is None else bool(status)}


@log_step
def reproducibility_info(
    csv_path: Path | None = None, seed: int = config.SEED
) -> dict[str, object]:
    """Reúne lo necesario para repetir una corrida.

    Args:
        csv_path: CSV original; por defecto ``config.RAW_CSV_PATH``.
        seed: semilla de la corrida; por defecto ``config.SEED``.

    Returns:
        Diccionario con ``seed``, ``dataset_sha256``, ``versions``,
        ``git_commit`` y ``git_dirty``.
    """
    path = csv_path if csv_path is not None else config.RAW_CSV_PATH
    git = git_info()
    info = {
        "seed": seed,
        "dataset_sha256": file_sha256(path),
        "versions": library_versions(),
        "git_commit": git["commit"],
        "git_dirty": git["dirty"],
    }
    logger.info(
        "Reproducibilidad: semilla %s, dataset %s, commit %s",
        info["seed"],
        str(info["dataset_sha256"])[:12],
        str(info["git_commit"])[:12],
    )
    return info


def _json_default(value: object) -> object:
    """Convierte tipos de numpy a tipos nativos para serializar a JSON."""
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"No se puede serializar {type(value).__name__}")


def to_json(doc: dict) -> str:
    """Serializa un documento a JSON (UTF-8, sin escapar acentos)."""
    return json.dumps(doc, ensure_ascii=False, default=_json_default)


def build_run_config(
    csv_path: Path | None = None, seed: int = config.SEED
) -> dict[str, object]:
    """Arma el documento de configuración de una corrida (Tabla 3).

    Args:
        csv_path: CSV original; por defecto ``config.RAW_CSV_PATH``.
        seed: semilla de la corrida; por defecto ``config.SEED``.

    Returns:
        Reproducibilidad (semilla, hash, versiones, commit) más los parámetros
        del experimento: hold-out, cortes, target, temperatura base, variables
        de cada escenario y grillas de hiperparámetros.
    """
    return {
        **reproducibility_info(csv_path, seed),
        "holdout_months": config.HOLDOUT_MONTHS,
        "cv_splits": config.CV_SPLITS,
        "target": {
            "years_back": config.TARGET_YEARS_BACK,
            "window_months": config.TARGET_WINDOW_MONTHS,
            "quantile": config.TARGET_QUANTILE,
        },
        "base_temp_c": config.BASE_TEMP_C,
        "scenarios": {name: list(cols) for name, cols in features.SCENARIOS.items()},
        "param_grids": models.PARAM_GRIDS,
    }


@log_step
def save_run_config(
    run_id: str, doc: dict[str, object], db_path: Path | None = None
) -> None:
    """Guarda la configuración de una corrida en la tabla ``configuracion``.

    Args:
        run_id: identificador de la corrida (uno por corrida).
        doc: documento de ``build_run_config``.
        db_path: ruta de ``unstructured.db``; por defecto la de ``config``.

    Raises:
        sqlite3.IntegrityError: si ya hay una configuración para ese ``run_id``.
    """
    path = db_path if db_path is not None else config.UNSTRUCTURED_DB_PATH
    db.init_unstructured_db(path)
    with db.get_connection(path) as conn:
        conn.execute(
            "INSERT INTO configuracion (run_id, created_at, doc) VALUES (?, ?, ?)",
            (run_id, db.now_iso(), to_json(doc)),
        )


def model_filename(run_id: str, scenario: str, model: str) -> str:
    """Nombre del archivo del modelo: ``<run_id>_<escenario>_<modelo>.joblib``."""
    return f"{run_id}_{scenario}_{model}.joblib"


def _metrics_doc(result: EvalResult) -> dict[str, object]:
    """Métricas y matriz de confusión de una evaluación, sin predicciones."""
    doc = asdict(result)
    del doc["y_pred"], doc["proba"]
    return doc


def build_result_doc(run: ModelRun, model_name: str, model_sha256: str) -> dict:
    """Arma el documento JSON de un resultado (Tabla 2).

    Args:
        run: resultado de entrenar un modelo en un escenario.
        model_name: nombre del archivo ``.joblib`` guardado.
        model_sha256: hash SHA-256 de ese archivo.

    Returns:
        Documento con parámetros, validación cruzada, métricas del hold-out,
        referencias, predicciones por mes y datos del archivo del modelo.
    """
    predictions = [
        {"fecha": date, "real": real, "predicho": pred, "probabilidad": proba}
        for date, real, pred, proba in zip(
            run.test_dates, run.y_test, run.test.y_pred, run.test.proba, strict=True
        )
    ]
    return {
        "scenario": run.scenario,
        "model": run.model,
        "features": run.features,
        "n_train": run.n_train,
        "n_test": run.n_test,
        "params": run.search.best_params,
        "cv": {
            "scores": run.search.cv_scores,
            "mean": run.search.cv_mean,
            "std": run.search.cv_std,
        },
        "holdout": _metrics_doc(run.test),
        "baselines": {n: _metrics_doc(r) for n, r in run.baselines.items()},
        "predictions": predictions,
        "model_file": {"name": model_name, "sha256": model_sha256},
    }


@log_step
def save_model_run(
    run_id: str,
    run: ModelRun,
    models_dir: Path | None = None,
    db_path: Path | None = None,
) -> Path:
    """Guarda el modelo (``.joblib``) y su resultado (Tabla 2) de una corrida.

    Si falla el guardado en la base, borra el archivo recién escrito.

    Args:
        run_id: identificador de la corrida.
        run: resultado de entrenar un modelo en un escenario.
        models_dir: carpeta de modelos; por defecto ``config.MODELS_DIR``.
        db_path: ruta de ``unstructured.db``; por defecto la de ``config``.

    Returns:
        Ruta del archivo ``.joblib``.

    Raises:
        ValueError: si ya hay un resultado para ese ``run_id``, escenario y
            modelo.
        FileExistsError: si el archivo del modelo ya existe.
    """
    directory = models_dir if models_dir is not None else config.MODELS_DIR
    path = db_path if db_path is not None else config.UNSTRUCTURED_DB_PATH
    db.init_unstructured_db(path)
    with db.get_connection(path) as conn:
        exists = conn.execute(
            "SELECT 1 FROM resultados_modelos "
            "WHERE run_id = ? AND scenario = ? AND model = ?",
            (run_id, run.scenario, run.model),
        ).fetchone()
    if exists:
        raise ValueError(f"Ya existe el resultado {run_id}/{run.scenario}/{run.model}")

    directory.mkdir(parents=True, exist_ok=True)
    name = model_filename(run_id, run.scenario, run.model)
    model_path = directory / name
    if model_path.exists():
        raise FileExistsError(model_path)
    joblib.dump(run.search.estimator, model_path)
    try:
        doc = build_result_doc(run, name, file_sha256(model_path))
        with db.get_connection(path) as conn:
            conn.execute(
                "INSERT INTO resultados_modelos "
                "(run_id, scenario, model, created_at, doc) VALUES (?, ?, ?, ?, ?)",
                (run_id, run.scenario, run.model, db.now_iso(), to_json(doc)),
            )
    except BaseException:
        model_path.unlink(missing_ok=True)
        raise
    logger.info("Resultado guardado: %s/%s en %s", run.scenario, run.model, name)
    return model_path


def load_result(
    run_id: str, scenario: str, model: str, db_path: Path | None = None
) -> dict:
    """Recupera el documento JSON de un resultado guardado.

    Args:
        run_id: identificador de la corrida.
        scenario: ``"A"`` o ``"B"``.
        model: nombre del modelo.
        db_path: ruta de ``unstructured.db``; por defecto la de ``config``.

    Returns:
        El documento decodificado.

    Raises:
        LookupError: si no existe ese resultado.
    """
    path = db_path if db_path is not None else config.UNSTRUCTURED_DB_PATH
    with db.get_connection(path) as conn:
        row = conn.execute(
            "SELECT doc FROM resultados_modelos "
            "WHERE run_id = ? AND scenario = ? AND model = ?",
            (run_id, scenario, model),
        ).fetchone()
    if row is None:
        raise LookupError(f"No hay resultado para {run_id}/{scenario}/{model}")
    return json.loads(row["doc"])


def load_model(doc: dict, models_dir: Path | None = None) -> object:
    """Carga el modelo de un resultado, verificando antes su hash SHA-256.

    Los ``.joblib`` son pickles: cargar uno ajeno puede ejecutar código. Por
    eso solo se carga si el archivo coincide con el hash guardado al entrenar.

    Args:
        doc: documento de ``load_result``.
        models_dir: carpeta de modelos; por defecto ``config.MODELS_DIR``.

    Returns:
        El pipeline entrenado.

    Raises:
        FileNotFoundError: si el archivo no existe.
        ValueError: si el hash no coincide con el guardado.
    """
    directory = models_dir if models_dir is not None else config.MODELS_DIR
    info = doc["model_file"]
    path = directory / info["name"]
    if not path.is_file():
        raise FileNotFoundError(path)
    if file_sha256(path) != info["sha256"]:
        raise ValueError(f"El hash de {path.name} no coincide con el guardado")
    return joblib.load(path)


def has_results(run_id: str, db_path: Path | None = None) -> bool:
    """Indica si una corrida ya tiene resultados guardados en la Tabla 2.

    Args:
        run_id: identificador de la corrida.
        db_path: ruta de ``unstructured.db``; por defecto la de ``config``.
    """
    path = db_path if db_path is not None else config.UNSTRUCTURED_DB_PATH
    db.init_unstructured_db(path)
    with db.get_connection(path) as conn:
        row = conn.execute(
            "SELECT 1 FROM resultados_modelos WHERE run_id = ? LIMIT 1", (run_id,)
        ).fetchone()
    return row is not None


def feature_importance(estimator: object, feature_names: list[str]) -> pd.DataFrame:
    """Importancia de cada variable en un pipeline ya entrenado.

    - Regresión logística: coeficientes (las variables están estandarizadas
      dentro del pipeline, así que son comparables; el signo indica el sentido).
    - Random Forest y XGBoost: ``feature_importances_`` (suman 1, sin signo).

    Args:
        estimator: ``Pipeline`` con un paso final llamado ``clf``.
        feature_names: nombres de las variables, en el orden de entrenamiento.

    Returns:
        DataFrame con ``feature``, ``importancia`` y ``tipo`` (``coeficiente``
        o ``importancia``), ordenado de mayor a menor valor absoluto.

    Raises:
        ValueError: si el clasificador no expone coeficientes ni importancias o
            su cantidad no coincide con las variables.
    """
    clf = estimator.named_steps["clf"]
    if hasattr(clf, "coef_"):
        values, kind = np.ravel(clf.coef_), "coeficiente"
    elif hasattr(clf, "feature_importances_"):
        values, kind = np.asarray(clf.feature_importances_), "importancia"
    else:
        raise ValueError(f"{type(clf).__name__} no expone importancias de variables")
    if len(values) != len(feature_names):
        raise ValueError("La cantidad de variables no coincide con el modelo")
    table = pd.DataFrame(
        {"feature": feature_names, "importancia": values.astype(float), "tipo": kind}
    )
    order = table["importancia"].abs().sort_values(ascending=False).index
    return table.loc[order].reset_index(drop=True)
