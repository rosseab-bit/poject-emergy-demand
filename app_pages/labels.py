"""Etiquetas en castellano de modelos y escenarios para mostrar en pantalla."""

import pandas as pd

from src import models

MODEL_LABELS = models.MODEL_LABELS
SCENARIO_LABELS = {
    "A": "A · anticipación (sin potencia máxima)",
    "B": "B · diagnóstico (con potencia máxima)",
}


model_label = models.model_label


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
