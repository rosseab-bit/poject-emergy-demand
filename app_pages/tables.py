"""Tablas listas para mostrar en pantalla."""

import pandas as pd

from app_pages.labels import model_label, month_label
from src import features, models


def results_for_display(table: pd.DataFrame) -> pd.DataFrame:
    """Tabla de resultados con columnas y valores listos para mostrar."""
    shown = pd.DataFrame(
        {
            "Escenario": table["scenario"],
            "Modelo": table["model"].map(model_label),
            "F1 validación (promedio)": table["cv_mean"],
            "± desvío": table["cv_std"],
            "Accuracy hold-out": table["accuracy"],
            "Precision": table["precision"],
            "Recall": table["recall"],
            "F1 hold-out": table["f1"],
            "F1 «siempre alta»": table["f1_siempre_alta"],
            "F1 aleatorio": table["f1_aleatorio"],
        }
    )
    return shown


LEVEL_ICONS = {"ERROR": "❌", "CRITICAL": "❌", "WARNING": "⚠️"}
STATUS_ICONS = {"ok": "✅", "running": "⏳", "error": "❌"}
# Fondos translúcidos: se leen bien en el tema claro y en el oscuro.
ROW_STYLES = {
    "ERROR": "background-color: rgba(220, 38, 38, 0.22)",
    "CRITICAL": "background-color: rgba(220, 38, 38, 0.22)",
    "WARNING": "background-color: rgba(245, 158, 11, 0.20)",
}


def with_icon(value: str, icons: dict[str, str]) -> str:
    """Antepone el ícono del estado o nivel; así no se depende solo del color."""
    icon = icons.get(value)
    return f"{icon} {value}" if icon else value


def logs_for_display(logs: pd.DataFrame) -> pd.DataFrame:
    """Tabla de eventos con nombres de columna y niveles con ícono."""
    return pd.DataFrame(
        {
            "Fecha (UTC)": logs["ts"],
            "Nivel": logs["level"].map(lambda v: with_icon(v, LEVEL_ICONS)),
            "Módulo": logs["module"],
            "Mensaje": logs["message"],
            "Corrida": logs["run_id"],
        }
    )


def runs_for_display(runs: pd.DataFrame) -> pd.DataFrame:
    """Tabla de corridas con el estado acompañado de un ícono."""
    return pd.DataFrame(
        {
            "Corrida": runs["run_id"],
            "Estado": runs["status"].map(lambda v: with_icon(v, STATUS_ICONS)),
            "Inicio (UTC)": runs["started_at"],
            "Fin (UTC)": runs["finished_at"],
            "Semilla": runs["seed"],
        }
    )


def highlight_row(row: pd.Series) -> list[str]:
    """Estilo de una fila: rojo para errores, ámbar para advertencias."""
    text = str(row.get("Nivel", row.get("Estado", "")))
    key = text.split(" ")[-1].upper()
    return [ROW_STYLES.get(key, "")] * len(row)


LAG_ROW_LABELS = {
    "demanda_total": "Demanda total",
    "demanda_residencial": "Demanda residencial",
    "comercio_e_industria": "Comercio e industria",
    "grandes_usuarios": "Grandes usuarios",
}


def lags_for_display(lags: dict[str, float], month: pd.Timestamp) -> pd.DataFrame:
    """Rezagos que se completan solos desde ``datos_limpios``, por variable."""
    one = f"1 mes antes ({month_label(month - pd.DateOffset(months=1))})"
    twelve = f"12 meses antes ({month_label(month - pd.DateOffset(months=12))})"
    return pd.DataFrame(
        {
            "Variable": [LAG_ROW_LABELS[c] for c in features.LAG_SOURCE_COLUMNS],
            one: [
                lags[features.lag_column_name(c, 1)]
                for c in features.LAG_SOURCE_COLUMNS
            ],
            twelve: [
                lags[features.lag_column_name(c, 12)]
                for c in features.LAG_SOURCE_COLUMNS
            ],
        }
    )


def prediction_for_display(doc: dict) -> pd.DataFrame:
    """Resultado de una predicción: una fila por modelo, con su referencia."""
    rows = []
    for name in models.MODEL_NAMES:
        entry = doc["modelos"].get(name)
        if entry is None:
            continue
        rows.append(
            {
                "Modelo": model_label(name),
                "Probabilidad de alta": entry["probabilidad"],
                "Clasificación": "alta" if entry["clase"] == 1 else "normal",
                "F1 del modelo (hold-out)": entry["f1_holdout"],
                "F1 «siempre alta»": entry["f1_siempre_alta"],
                "¿Supera la referencia?": "Sí" if entry["supera_referencia"] else "No",
            }
        )
    return pd.DataFrame(rows)


def history_for_display(history: pd.DataFrame, with_run: bool) -> pd.DataFrame:
    """Historial de predicciones guardadas; oculta los modelos nunca entrenados."""
    shown = pd.DataFrame(
        {
            "Guardada (UTC)": history["created_at"],
            "Mes": history["fecha"].str[:7],
            "Temperatura (°C)": history["temperatura"],
        }
    )
    if with_run:
        shown.insert(0, "Corrida", history["run_id"])
    for name in models.MODEL_NAMES:
        column = history[f"p_{name}"]
        if column.notna().any():
            shown[f"P(alta) {model_label(name)}"] = column
    return shown
