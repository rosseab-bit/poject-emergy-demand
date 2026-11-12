"""Etiquetas en castellano de modelos y escenarios para mostrar en pantalla."""

import pandas as pd

from src import models

MODEL_LABELS = {
    models.LOGISTIC_REGRESSION: "Regresión logística",
    models.RANDOM_FOREST: "Random Forest",
    models.XGBOOST: "XGBoost",
}
SCENARIO_LABELS = {
    "A": "A · anticipación (sin potencia máxima)",
    "B": "B · diagnóstico (con potencia máxima)",
}


def model_label(name: str) -> str:
    """Nombre legible de un modelo; devuelve el mismo texto si no lo conoce."""
    return MODEL_LABELS.get(name, name)


MONTH_ABBR = (
    "ene",
    "feb",
    "mar",
    "abr",
    "may",
    "jun",
    "jul",
    "ago",
    "sep",
    "oct",
    "nov",
    "dic",
)
MONTH_NAMES = (
    "enero",
    "febrero",
    "marzo",
    "abril",
    "mayo",
    "junio",
    "julio",
    "agosto",
    "septiembre",
    "octubre",
    "noviembre",
    "diciembre",
)


def month_label(month: pd.Timestamp) -> str:
    """Mes y año abreviados, por ejemplo ``sep 2025``."""
    return f"{MONTH_ABBR[month.month - 1]} {month.year}"
