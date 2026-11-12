"""Etiquetas en castellano de modelos y escenarios para mostrar en pantalla."""

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
