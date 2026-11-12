"""Validación, búsqueda de hiperparámetros, evaluación y guardado de modelos."""

import pandas as pd

from src import config


def split_holdout(
    x: pd.DataFrame, y: pd.Series, holdout: int = config.HOLDOUT_MONTHS
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Separa los últimos ``holdout`` meses como conjunto de prueba final.

    El corte es temporal: el entrenamiento es siempre anterior a la prueba.
    ``x`` e ``y`` deben estar ordenados por fecha, como los devuelve
    ``features.get_model_data``.

    Args:
        x: variables predictoras, ordenadas por fecha.
        y: target, con el mismo orden e índice que ``x``.
        holdout: cantidad de meses finales para la prueba.

    Returns:
        Tupla ``(x_train, x_test, y_train, y_test)``.

    Raises:
        ValueError: si ``x`` e ``y`` no coinciden o ``holdout`` no deja datos
            de entrenamiento.
    """
    if not x.index.equals(y.index):
        raise ValueError("x e y deben tener el mismo índice.")
    if not 0 < holdout < len(x):
        raise ValueError(f"holdout inválido: {holdout} (hay {len(x)} filas).")
    return x.iloc[:-holdout], x.iloc[-holdout:], y.iloc[:-holdout], y.iloc[-holdout:]
