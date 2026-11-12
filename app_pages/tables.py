"""Tablas listas para mostrar en pantalla."""

import pandas as pd

from app_pages.labels import model_label


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
