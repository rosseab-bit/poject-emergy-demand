"""Variables derivadas: temperatura, calendario, rezagos y target."""

import numpy as np
import pandas as pd

from src import config
from src.data import DATE_COLUMN
from src.logger import log_step

TEMPERATURE_COLUMN = "temperatura_promedio"
COLD_COLUMN = "grados_dia_frio"
HEAT_COLUMN = "grados_dia_calor"


@log_step
def add_temperature_features(
    df: pd.DataFrame, base: float = config.BASE_TEMP_C
) -> pd.DataFrame:
    """Agrega los grados-día de frío y de calor respecto de una temperatura base.

    Se calculan con la temperatura media mensual (aproximación: el dataset
    no trae temperaturas diarias). Frío = ``max(0, base - T)`` y calor =
    ``max(0, T - base)``.

    Args:
        df: datos con ``temperatura_promedio``.
        base: temperatura base en °C; por defecto ``config.BASE_TEMP_C``.

    Returns:
        Copia de ``df`` con ``grados_dia_frio`` y ``grados_dia_calor``.
    """
    out = df.copy()
    temp = out[TEMPERATURE_COLUMN]
    out[COLD_COLUMN] = (base - temp).clip(lower=0)
    out[HEAT_COLUMN] = (temp - base).clip(lower=0)
    return out


@log_step
def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega el mes en forma cíclica (seno y coseno) y el año.

    El seno y el coseno hacen que diciembre quede cerca de enero.

    Args:
        df: datos con la columna ``fecha``.

    Returns:
        Copia de ``df`` con ``mes_sin``, ``mes_cos`` y ``anio``.
    """
    out = df.copy()
    month = out[DATE_COLUMN].dt.month
    angle = 2 * np.pi * (month - 1) / 12
    out["mes_sin"] = np.sin(angle)
    out["mes_cos"] = np.cos(angle)
    out["anio"] = out[DATE_COLUMN].dt.year
    return out
