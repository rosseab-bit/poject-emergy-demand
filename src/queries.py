"""Lectura de las bases para la interfaz: corridas, pasos, resultados y logs.

La app consulta siempre estas funciones y no guarda nada en la memoria de
Streamlit, así los resultados siguen ahí al cerrar y reabrir la app.
"""

from pathlib import Path

import pandas as pd

from src import config, data, db, features, models, results

STEP_ETL = "ETL"
STEP_TRAINING = "Entrenamiento"
STEP_PREDICTION = "Predicción"
STEP_REPORT = "Reporte"
STEPS = (STEP_ETL, STEP_TRAINING, STEP_PREDICTION, STEP_REPORT)

RUN_COLUMNS = ["run_id", "started_at", "finished_at", "status", "seed", "notes"]
LOG_COLUMNS = ["id", "ts", "level", "module", "message", "run_id"]


def _read(query: str, params: tuple, path: Path | None, columns: list[str]):
    """Ejecuta una consulta sobre ``structured.db`` y devuelve un DataFrame."""
    db.init_structured_db(path)
    with db.get_connection(path) as conn:
        rows = conn.execute(query, params).fetchall()
    return pd.DataFrame([tuple(r) for r in rows], columns=columns)


def list_runs(db_path: Path | None = None) -> pd.DataFrame:
    """Lista las corridas, la más reciente primero.

    Args:
        db_path: ruta de ``structured.db``; por defecto la de ``config``.

    Returns:
        DataFrame con ``RUN_COLUMNS``; vacío si no hay corridas.
    """
    return _read(
        f"SELECT {', '.join(RUN_COLUMNS)} FROM runs "
        "ORDER BY started_at DESC, rowid DESC",
        (),
        db_path,
        RUN_COLUMNS,
    )


def run_steps(
    run_id: str,
    db_path: Path | None = None,
    unstructured_path: Path | None = None,
) -> dict[str, bool]:
    """Indica qué pasos de una corrida están hechos.

    El ETL se deduce de sus eventos en ``logs`` y el entrenamiento de los
    resultados guardados. Predicción y reporte quedan pendientes hasta que
    existan esas etapas.

    Args:
        run_id: identificador de la corrida.
        db_path: ruta de ``structured.db``; por defecto la de ``config``.
        unstructured_path: ruta de ``unstructured.db``; por defecto la de
            ``config``.

    Returns:
        Diccionario paso a ``True`` (hecho) o ``False`` (pendiente).
    """
    etl = _read(
        "SELECT 1 FROM logs WHERE run_id = ? AND message LIKE ? LIMIT 1",
        (run_id, "Tabla datos_limpios guardada%"),
        db_path,
        ["x"],
    )
    return {
        STEP_ETL: not etl.empty,
        STEP_TRAINING: results.has_results(run_id, unstructured_path),
        STEP_PREDICTION: False,
        STEP_REPORT: False,
    }


def run_summary(
    run_id: str,
    db_path: Path | None = None,
    unstructured_path: Path | None = None,
) -> dict[str, object] | None:
    """Resume una corrida para la barra lateral.

    Args:
        run_id: identificador de la corrida.
        db_path: ruta de ``structured.db``; por defecto la de ``config``.
        unstructured_path: ruta de ``unstructured.db``; por defecto la de
            ``config``.

    Returns:
        Diccionario con los datos de ``runs``, el hash del dataset (``None``
        hasta que se entrena) y el estado de los pasos; ``None`` si la corrida
        no existe.
    """
    db.init_structured_db(db_path)
    run = db.get_run(run_id, db_path)
    if run is None:
        return None
    summary = {key: run[key] for key in RUN_COLUMNS}
    summary["dataset_sha256"] = _config_value(
        run_id, "$.dataset_sha256", unstructured_path
    )
    summary["steps"] = run_steps(run_id, db_path, unstructured_path)
    return summary


def _config_value(run_id: str, json_path: str, path: Path | None) -> object:
    """Lee un valor de la configuración de una corrida con ``json_extract``."""
    path = path if path is not None else config.UNSTRUCTURED_DB_PATH
    db.init_unstructured_db(path)
    with db.get_connection(path) as conn:
        row = conn.execute(
            "SELECT json_extract(doc, ?) FROM configuracion WHERE run_id = ?",
            (json_path, run_id),
        ).fetchone()
    return None if row is None else row[0]


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


def results_table(run_id: str, unstructured_path: Path | None = None) -> pd.DataFrame:
    """Tabla modelo por escenario con las métricas guardadas de una corrida.

    Args:
        run_id: identificador de la corrida.
        unstructured_path: ruta de ``unstructured.db``; por defecto la de
            ``config``.

    Returns:
        DataFrame con ``RESULT_COLUMNS``, ordenado por escenario y modelo;
        vacío si la corrida no tiene resultados.
    """
    path = unstructured_path or config.UNSTRUCTURED_DB_PATH
    db.init_unstructured_db(path)
    with db.get_connection(path) as conn:
        rows = conn.execute(
            "SELECT scenario, model, "
            "json_extract(doc, '$.n_train'), json_extract(doc, '$.n_test'), "
            "json_extract(doc, '$.cv.mean'), json_extract(doc, '$.cv.std'), "
            "json_extract(doc, '$.holdout.accuracy'), "
            "json_extract(doc, '$.holdout.precision'), "
            "json_extract(doc, '$.holdout.recall'), "
            "json_extract(doc, '$.holdout.f1'), "
            "json_extract(doc, '$.baselines.siempre_alta.f1'), "
            "json_extract(doc, '$.baselines.aleatorio_proporcional.f1'), "
            "json_extract(doc, '$.params') "
            "FROM resultados_modelos WHERE run_id = ?",
            (run_id,),
        ).fetchall()
    table = pd.DataFrame([tuple(r) for r in rows], columns=RESULT_COLUMNS)
    order = {name: i for i, name in enumerate(models.MODEL_NAMES)}
    table["_order"] = table["model"].map(order)
    table = table.sort_values(["scenario", "_order"]).drop(columns="_order")
    return table.reset_index(drop=True)


def result_docs(
    run_id: str, unstructured_path: Path | None = None
) -> list[dict[str, object]]:
    """Devuelve los documentos JSON completos de los resultados de una corrida.

    Sirven para las matrices de confusión, las predicciones por mes y las
    variables usadas.

    Args:
        run_id: identificador de la corrida.
        unstructured_path: ruta de ``unstructured.db``; por defecto la de
            ``config``.

    Returns:
        Lista de documentos, ordenados por escenario y modelo.
    """
    table = results_table(run_id, unstructured_path)
    return [
        results.load_result(run_id, row.scenario, row.model, unstructured_path)
        for row in table.itertuples()
    ]


def get_logs(
    run_id: str | None = None,
    levels: list[str] | None = None,
    modules: list[str] | None = None,
    db_path: Path | None = None,
) -> pd.DataFrame:
    """Devuelve eventos de ``logs`` con filtros opcionales.

    Args:
        run_id: si se indica, solo los eventos de esa corrida.
        levels: niveles a incluir (``INFO``, ``WARNING``, ``ERROR``...).
        modules: módulos a incluir (por ejemplo ``src.train``).
        db_path: ruta de ``structured.db``; por defecto la de ``config``.

    Returns:
        DataFrame con ``LOG_COLUMNS``, en orden cronológico.
    """
    conditions, params = [], []
    if run_id is not None:
        conditions.append("run_id = ?")
        params.append(run_id)
    for column, values in (("level", levels), ("module", modules)):
        if values:
            conditions.append(f"{column} IN ({', '.join('?' * len(values))})")
            params.extend(values)
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    return _read(
        f"SELECT {', '.join(LOG_COLUMNS)} FROM logs {where} ORDER BY id",
        tuple(params),
        db_path,
        LOG_COLUMNS,
    )


def list_log_modules(
    run_id: str | None = None, db_path: Path | None = None
) -> list[str]:
    """Lista los módulos que registraron eventos, para el filtro de la página.

    Args:
        run_id: si se indica, solo los de esa corrida.
        db_path: ruta de ``structured.db``; por defecto la de ``config``.
    """
    logs = get_logs(run_id, db_path=db_path)
    return sorted(logs["module"].unique().tolist())


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
    unstructured_path: Path | None = None,
    models_dir: Path | None = None,
) -> pd.DataFrame:
    """Importancia de variables de un modelo guardado (verifica su hash).

    Args:
        run_id: identificador de la corrida.
        scenario: ``"A"`` o ``"B"``.
        model: nombre del modelo.
        unstructured_path: ruta de ``unstructured.db``; por defecto la de
            ``config``.
        models_dir: carpeta de modelos; por defecto ``config.MODELS_DIR``.

    Returns:
        Salida de ``results.feature_importance``.
    """
    doc = results.load_result(run_id, scenario, model, unstructured_path)
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
