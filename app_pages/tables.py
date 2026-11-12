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
