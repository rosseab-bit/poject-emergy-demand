"""Variables derivadas: temperatura, calendario, rezagos y target."""

import numpy as np
import pandas as pd

from src import config
from src.data import DATE_COLUMN, SECTOR_COLUMNS, validate_monthly_dates
from src.logger import log_step

TEMPERATURE_COLUMN = "temperatura_promedio"
COLD_COLUMN = "grados_dia_frio"
HEAT_COLUMN = "grados_dia_calor"
TOTAL_COLUMN = "demanda_total"
LAG_SOURCE_COLUMNS = [TOTAL_COLUMN, *SECTOR_COLUMNS]
LAGS = (1, 12)


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


def lag_column_name(column: str, lag: int) -> str:
    """Nombre de la columna de rezago, por ejemplo ``demanda_total_lag12``."""
    return f"{column}_lag{lag}"


@log_step
def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega los rezagos de 1 y 12 meses de la demanda total y sectorial.

    Cada rezago toma el valor de ``k`` meses antes, así que solo usa el
    pasado. Los primeros meses quedan en nulo por falta de historia.

    Args:
        df: datos ordenados por fecha, un mes tras otro, con
            ``LAG_SOURCE_COLUMNS``.

    Returns:
        Copia de ``df`` con una columna ``<variable>_lag<k>`` por variable y
        rezago.

    Raises:
        ValueError: si las fechas no son mensuales y consecutivas.
    """
    validate_monthly_dates(df[DATE_COLUMN])
    out = df.copy()
    for column in LAG_SOURCE_COLUMNS:
        for lag in LAGS:
            out[lag_column_name(column, lag)] = out[column].shift(lag)
    return out
