"""Lectura de las bases para la interfaz: corridas, pasos, resultados y logs.

La app consulta siempre estas funciones y no guarda nada en la memoria de
Streamlit, así los resultados siguen ahí al cerrar y reabrir la app.
"""

from pathlib import Path

import pandas as pd

from src import config, data, db, features, logger, models, mongo, results, runs

STEP_ETL = "ETL"
STEP_TRAINING = "Entrenamiento"
STEP_PREDICTION = "Predicción"
STEP_REPORT = "Reporte"
STEPS = (STEP_ETL, STEP_TRAINING, STEP_PREDICTION, STEP_REPORT)

RUN_COLUMNS = ["run_id", "started_at", "finished_at", "status", "seed", "notes"]
LOG_COLUMNS = [
    "ts",
    "level",
    "module",
    "message",
    "run_id",
    "step",
    "event",
    "duration_ms",
]


def list_runs() -> pd.DataFrame:
    """Lista las corridas de MongoDB, la más reciente primero.

    Returns:
        DataFrame con ``RUN_COLUMNS``; vacío si no hay corridas.
    """
    cursor = (
        mongo.get_database()[mongo.RUNS]
        .find({}, {"_id": 0})
        .sort([("started_at", -1), ("_id", -1)])
    )
    return pd.DataFrame(list(cursor), columns=RUN_COLUMNS)


def report_path(run_id: str, reports_dir: Path | None = None) -> Path:
    """Ruta del PDF de una corrida: ``reports/reporte_<run_id>.pdf``."""
    directory = reports_dir if reports_dir is not None else config.REPORTS_DIR
    return directory / f"reporte_{run_id}.pdf"


def has_report(run_id: str, reports_dir: Path | None = None) -> bool:
    """Indica si ya se generó el PDF de una corrida (el archivo existe)."""
    return report_path(run_id, reports_dir).is_file()


def run_steps(
    run_id: str,
    reports_dir: Path | None = None,
) -> dict[str, bool]:
    """Indica qué pasos de una corrida están hechos.

    El ETL se deduce de sus eventos en la colección ``logs`` de MongoDB, el
    entrenamiento de los resultados guardados, la predicción de la colección
    ``predicciones`` y el reporte de la existencia del PDF en ``reports/``.

    Args:
        run_id: identificador de la corrida.
        reports_dir: carpeta de los PDF; por defecto ``config.REPORTS_DIR``.

    Returns:
        Diccionario paso a ``True`` (hecho) o ``False`` (pendiente).
    """
    etl = mongo.get_database()[mongo.LOGS].find_one(
        {"run_id": run_id, "message": {"$regex": "^Tabla datos_limpios guardada"}},
        {"_id": 1},
    )
    return {
        STEP_ETL: etl is not None,
        STEP_TRAINING: results.has_results(run_id),
        STEP_PREDICTION: has_predictions(run_id),
        STEP_REPORT: has_report(run_id, reports_dir),
    }


def has_predictions(run_id: str) -> bool:
    """Indica si una corrida tiene al menos una predicción guardada.

    Args:
        run_id: identificador de la corrida.
    """
    found = mongo.get_database()[mongo.PREDICTIONS].find_one(
        {"run_id": run_id}, {"_id": 1}
    )
    return found is not None


def run_summary(run_id: str) -> dict[str, object] | None:
    """Resume una corrida para la barra lateral.

    Args:
        run_id: identificador de la corrida.

    Returns:
        Diccionario con los datos de ``runs``, el hash del dataset (``None``
        hasta que se entrena) y el estado de los pasos; ``None`` si la corrida
        no existe.
    """
    run = runs.get_run(run_id)
    if run is None:
        return None
    summary = {key: run.get(key) for key in RUN_COLUMNS}
    saved = mongo.get_database()[mongo.CONFIGURATION].find_one(
        {"run_id": run_id}, {"dataset_sha256": 1}
    )
    summary["dataset_sha256"] = None if saved is None else saved["dataset_sha256"]
    summary["steps"] = run_steps(run_id)
    return summary


# Campo del documento de cada columna de ``results_table`` (con punto = anidado).
RESULT_FIELDS = [
    "scenario",
    "model",
    "n_train",
    "n_test",
    "cv.mean",
    "cv.std",
    "holdout.accuracy",
    "holdout.precision",
    "holdout.recall",
    "holdout.f1",
    "baselines.siempre_alta.f1",
    "baselines.aleatorio_proporcional.f1",
    "params",
]
RESULT_COLUMNS = [
    "scenario",
    "model",
    "n_train",
    "n_test",
    "cv_mean",
    "cv_std",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "f1_siempre_alta",
    "f1_aleatorio",
    "params",
]


def _nested(doc: dict, dotted: str) -> object:
    """Valor de un campo anidado (``"cv.mean"``) de un documento, o ``None``."""
    value: object = doc
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def results_table(run_id: str) -> pd.DataFrame:
    """Tabla modelo por escenario con las métricas guardadas de una corrida.

    Args:
        run_id: identificador de la corrida.

    Returns:
        DataFrame con ``RESULT_COLUMNS``, ordenado por escenario y modelo;
        vacío si la corrida no tiene resultados.
    """
    cursor = mongo.get_database()[mongo.RESULTS].find(
        {"run_id": run_id}, {field: 1 for field in RESULT_FIELDS} | {"_id": 0}
    )
    rows = []
    for doc in cursor:
        rows.append([_nested(doc, field) for field in RESULT_FIELDS])
    table = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    order = {name: i for i, name in enumerate(models.MODEL_NAMES)}
    table["_order"] = table["model"].map(order)
    table = table.sort_values(["scenario", "_order"]).drop(columns="_order")
    return table.reset_index(drop=True)


def result_docs(run_id: str) -> list[dict[str, object]]:
    """Devuelve los documentos JSON completos de los resultados de una corrida.

    Sirven para las matrices de confusión, las predicciones por mes y las
    variables usadas.

    Args:
        run_id: identificador de la corrida.

    Returns:
        Lista de documentos, ordenados por escenario y modelo.
    """
    table = results_table(run_id)
    return [
        results.load_result(run_id, row.scenario, row.model)
        for row in table.itertuples()
    ]


def get_logs(
    run_id: str | None = None,
    levels: list[str] | None = None,
    modules: list[str] | None = None,
) -> pd.DataFrame:
    """Devuelve eventos de la colección ``logs`` de MongoDB, con filtros.

    Args:
        run_id: si se indica, solo los eventos de esa corrida.
        levels: niveles a incluir (``INFO``, ``WARNING``, ``ERROR``...).
        modules: módulos a incluir (por ejemplo ``src.train``).

    Returns:
        DataFrame con ``LOG_COLUMNS``, en orden cronológico. ``step``, ``event``
        y ``duration_ms`` son nulos en los eventos que no los tienen.
    """
    query: dict[str, object] = {}
    if run_id is not None:
        query["run_id"] = run_id
    for field, values in (("level", levels), ("module", modules)):
        if values:
            query[field] = {"$in": list(values)}
    cursor = (
        mongo.get_database()[mongo.LOGS]
        .find(query, {"_id": 0})
        .sort([("ts", 1), ("_id", 1)])
    )
    return pd.DataFrame(list(cursor), columns=LOG_COLUMNS)


def list_log_modules(run_id: str | None = None) -> list[str]:
    """Lista los módulos que registraron eventos, para el filtro de la página.

    Args:
        run_id: si se indica, solo los de esa corrida.
    """
    query = {} if run_id is None else {"run_id": run_id}
    return sorted(mongo.get_database()[mongo.LOGS].distinct("module", query))


def data_overview(db_path: Path | None = None) -> dict[str, object] | None:
    """Resume la tabla ``datos_limpios`` y el target para la página Datos.

    Args:
        db_path: ruta de ``structured.db``; por defecto la de ``config``.

    Returns:
        ``None`` si el ETL todavía no guardó datos. Si no, un diccionario con
        ``rows``, ``rows_complete``, ``rows_sector_nulls``, ``repeated_months``,
        ``first_month``, ``last_month`` y ``target`` (filas con target,
        positivos, negativos, proporción y primer mes con target).
    """
    db.init_structured_db(db_path)
    clean = data.load_clean_data(db_path)
    if clean.empty:
        return None
    with logger.muted():  # mirar la página no debe agregar eventos a logs
        dataset = features.build_dataset(clean)
    _, y = features.get_model_data(dataset, "A")
    labeled = dataset.dropna(subset=[features.TARGET_COLUMN])
    balance = features.class_balance(y)
    month = clean[data.DATE_COLUMN].dt.strftime("%Y-%m")
    return {
        "rows": len(clean),
        "rows_complete": data.count_complete_rows(clean),
        "rows_sector_nulls": int(clean[data.SECTOR_COLUMNS].isna().any(axis=1).sum()),
        "repeated_months": month[clean["potencia_repetida"] == 1].tolist(),
        "first_month": month.iloc[0],
        "last_month": month.iloc[-1],
        "target": {
            **balance,
            "first_month": labeled[data.DATE_COLUMN].iloc[0].strftime("%Y-%m"),
        },
    }


def add_demand_index(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega ``indice_demanda``: demanda del mes sobre el promedio de su año.

    Quita la tendencia de crecimiento para que se vea la estacionalidad. Solo
    se calcula en años con los 12 meses; en el resto queda en nulo.

    Args:
        df: datos con ``fecha`` y ``demanda_total``.

    Returns:
        Copia de ``df`` con la columna ``indice_demanda`` (1,0 = promedio del
        año).
    """
    out = df.copy()
    year = out[data.DATE_COLUMN].dt.year
    months = out.groupby(year)[data.DATE_COLUMN].transform("count")
    mean = out.groupby(year)["demanda_total"].transform("mean")
    out["indice_demanda"] = (out["demanda_total"] / mean).where(months == 12)
    return out


def seasonality_by_month(df: pd.DataFrame) -> pd.DataFrame:
    """Promedio del ``indice_demanda`` por mes calendario (1 a 12).

    Args:
        df: datos con ``fecha`` y ``demanda_total``.

    Returns:
        DataFrame indexado por mes con la columna ``indice_medio``.
    """
    indexed = add_demand_index(df)
    month = indexed[data.DATE_COLUMN].dt.month
    mean = indexed.groupby(month)["indice_demanda"].mean()
    return mean.rename("indice_medio").to_frame().rename_axis("mes")


def feature_importance_for(
    run_id: str,
    scenario: str,
    model: str,
    models_dir: Path | None = None,
) -> pd.DataFrame:
    """Importancia de variables de un modelo guardado (verifica su hash).

    Args:
        run_id: identificador de la corrida.
        scenario: ``"A"`` o ``"B"``.
        model: nombre del modelo.
        models_dir: carpeta de modelos; por defecto ``config.MODELS_DIR``.

    Returns:
        Salida de ``results.feature_importance``.
    """
    doc = results.load_result(run_id, scenario, model)
    estimator = results.load_model(doc, models_dir)
    return results.feature_importance(estimator, doc["features"])


def scenario_comparison(table: pd.DataFrame) -> pd.DataFrame:
    """Compara el escenario B (con potencia máxima) contra el A por modelo.

    Args:
        table: salida de ``results_table``.

    Returns:
        Una fila por modelo con los dos escenarios entrenados: F1 de hold-out y
        de validación en A y B, y la diferencia B menos A. Vacío si falta alguno.
    """
    columns = ["model", "f1_A", "f1_B", "dif_f1", "cv_A", "cv_B", "dif_cv"]
    a = table[table["scenario"] == "A"].set_index("model")
    b = table[table["scenario"] == "B"].set_index("model")
    common = [
        name for name in models.MODEL_NAMES if name in a.index and name in b.index
    ]
    rows = [
        (
            name,
            a.at[name, "f1"],
            b.at[name, "f1"],
            b.at[name, "f1"] - a.at[name, "f1"],
            a.at[name, "cv_mean"],
            b.at[name, "cv_mean"],
            b.at[name, "cv_mean"] - a.at[name, "cv_mean"],
        )
        for name in common
    ]
    return pd.DataFrame(rows, columns=columns)


def noise_summary(
    table: pd.DataFrame, docs: list[dict[str, object]]
) -> dict[str, float] | None:
    """Datos para advertir cuánto ruido tienen las métricas de una corrida.

    Args:
        table: salida de ``results_table``.
        docs: salida de ``result_docs``.

    Returns:
        ``None`` si no hay resultados. Si no, ``n_test`` (meses del hold-out),
        ``positives`` (meses altos reales en el hold-out), ``max_cv_std`` (mayor
        desvío de validación) y ``always_high_f1`` (F1 de la referencia).
    """
    if table.empty or not docs:
        return None
    predictions = docs[0]["predictions"]
    return {
        "n_test": len(predictions),
        "positives": sum(p["real"] for p in predictions),
        "max_cv_std": float(table["cv_std"].max()),
        "always_high_f1": float(table["f1_siempre_alta"].iloc[0]),
    }


PREDICTION_COLUMNS = ["id", "run_id", "created_at", "fecha", "temperatura"]


def prediction_history(run_id: str | None = None) -> pd.DataFrame:
    """Lista las predicciones guardadas, la más reciente primero.

    Args:
        run_id: si se indica, solo las de esa corrida.

    Returns:
        DataFrame con ``PREDICTION_COLUMNS`` y, por cada modelo, las columnas
        ``p_<modelo>`` (probabilidad de demanda alta) y ``clase_<modelo>``.
    """
    columns = PREDICTION_COLUMNS + [
        column
        for name in models.MODEL_NAMES
        for column in (f"p_{name}", f"clase_{name}")
    ]
    rows = []
    for doc in _predictions(run_id, newest_first=True):
        row = [doc.get(column) for column in PREDICTION_COLUMNS]
        for name in models.MODEL_NAMES:
            entry = doc.get("modelos", {}).get(name, {})
            row += [entry.get("probabilidad"), entry.get("clase")]
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def _predictions(run_id: str | None, newest_first: bool) -> list[dict]:
    """Documentos de ``predicciones`` (de una corrida o de todas), ordenados."""
    direction = -1 if newest_first else 1
    cursor = (
        mongo.get_database()[mongo.PREDICTIONS]
        .find({} if run_id is None else {"run_id": run_id})
        .sort([("created_at", direction), ("_id", direction)])
    )
    return list(cursor)


def latest_prediction(run_id: str) -> dict[str, object] | None:
    """Devuelve el documento de la última predicción de una corrida, o ``None``."""
    found = _predictions(run_id, newest_first=True)[:1]
    return results.strip_metadata(found[0]) if found else None


def prediction_docs(run_id: str) -> list[dict[str, object]]:
    """Devuelve los documentos de todas las predicciones de una corrida.

    Args:
        run_id: identificador de la corrida.

    Returns:
        Documentos con su ``id`` y ``created_at``, del más viejo al más nuevo.
    """
    return [
        {**results.strip_metadata(doc), "created_at": doc["created_at"]}
        for doc in _predictions(run_id, newest_first=False)
    ]
