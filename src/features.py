"""Variables derivadas: temperatura, calendario, rezagos y target."""

import logging

import numpy as np
import pandas as pd

from src import config
from src.data import DATE_COLUMN, SECTOR_COLUMNS, validate_monthly_dates
from src.logger import log_step

logger = logging.getLogger(__name__)

TEMPERATURE_COLUMN = "temperatura_promedio"
COLD_COLUMN = "grados_dia_frio"
HEAT_COLUMN = "grados_dia_calor"
RATIO_COLUMN = "cociente_demanda"
THRESHOLD_COLUMN = "umbral_demanda"
TARGET_COLUMN = "demanda_alta"
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


def compute_demand_ratio(
    demand: pd.Series, years_back: int = config.TARGET_YEARS_BACK
) -> pd.Series:
    """Cociente entre la demanda del mes y el promedio del mismo mes previo.

    El promedio es el de los mismos meses calendario de los ``years_back``
    años anteriores (12, 24 y 36 meses atrás). Es nulo si falta alguno.

    Args:
        demand: demanda total mensual, un mes tras otro.
        years_back: cantidad de años previos a promediar.

    Returns:
        Serie con el cociente, nula en los primeros ``12 * years_back`` meses.
    """
    previous = pd.concat(
        [demand.shift(12 * k) for k in range(1, years_back + 1)], axis=1
    )
    return demand / previous.mean(axis=1, skipna=False)


@log_step
def add_target(
    df: pd.DataFrame,
    window: int = config.TARGET_WINDOW_MONTHS,
    quantile: float = config.TARGET_QUANTILE,
) -> pd.DataFrame:
    """Agrega el target binario ``demanda_alta`` con umbral móvil.

    Un mes es "alto" (1) si su cociente supera el cuantil de los cocientes de
    los ``window`` meses anteriores (sin incluir el propio mes). Usa solo el
    pasado: el valor de la fila ``t`` depende únicamente de filas ``<= t``.
    Los meses sin historia suficiente quedan en nulo.

    Las columnas ``cociente_demanda`` y ``umbral_demanda`` salen de la demanda
    del propio mes: sirven para auditar el target y **no deben usarse como
    predictores**.

    Args:
        df: datos ordenados por fecha, un mes tras otro, con ``demanda_total``.
        window: meses de cocientes que forman el umbral.
        quantile: cuantil que define "alta" (0,75 = 25 % superior).

    Returns:
        Copia de ``df`` con ``cociente_demanda``, ``umbral_demanda`` y
        ``demanda_alta`` (entero con nulos).

    Raises:
        ValueError: si las fechas no son mensuales y consecutivas.
    """
    validate_monthly_dates(df[DATE_COLUMN])
    out = df.copy()
    ratio = compute_demand_ratio(out[TOTAL_COLUMN])
    threshold = ratio.rolling(window, min_periods=window).quantile(quantile).shift(1)
    target = (ratio > threshold).astype("Int64").where(threshold.notna())
    out[RATIO_COLUMN] = ratio
    out[THRESHOLD_COLUMN] = threshold
    out[TARGET_COLUMN] = target
    return out


LAG_COLUMNS = [lag_column_name(c, k) for c in LAG_SOURCE_COLUMNS for k in LAGS]

# Escenario A (anticipación): todo lo que se conoce antes de que termine el mes.
FEATURES_A = [
    TEMPERATURE_COLUMN,
    COLD_COLUMN,
    HEAT_COLUMN,
    "mes_sin",
    "mes_cos",
    "anio",
    *LAG_COLUMNS,
]
# Escenario B (diagnóstico): suma la potencia máxima, que ocurre en el mismo mes.
FEATURES_B = [*FEATURES_A, "potencia_maxima"]
SCENARIOS = {"A": FEATURES_A, "B": FEATURES_B}


@log_step
def build_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula todas las variables derivadas y el target (reglas fijas).

    Args:
        df: datos limpios (tabla ``datos_limpios``), ordenados por fecha.

    Returns:
        DataFrame con las variables de temperatura, calendario, rezagos y el
        target. Los meses sin target (``demanda_alta`` nulo) se conservan.
    """
    out = add_temperature_features(df)
    out = add_calendar_features(out)
    out = add_lag_features(out)
    return add_target(out)


def get_model_data(df: pd.DataFrame, scenario: str) -> tuple[pd.DataFrame, pd.Series]:
    """Devuelve ``X`` e ``y`` de un escenario, solo con los meses con target.

    Args:
        df: salida de ``build_dataset``.
        scenario: ``"A"`` (anticipación) o ``"B"`` (diagnóstico).

    Returns:
        Tupla ``(X, y)``, con ``y`` entera (0 o 1) y el mismo índice que ``X``.

    Raises:
        ValueError: si el escenario no existe.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"Escenario inválido: {scenario!r}; usar 'A' o 'B'")
    labeled = df.dropna(subset=[TARGET_COLUMN])
    return labeled[SCENARIOS[scenario]], labeled[TARGET_COLUMN].astype(int)


def class_balance(y: pd.Series) -> dict[str, float]:
    """Resume el balance de clases del target.

    Args:
        y: target binario (0 o 1).

    Returns:
        Diccionario con ``total``, ``positivos``, ``negativos`` y
        ``proporcion_positivos``.
    """
    positives = int(y.sum())
    total = len(y)
    return {
        "total": total,
        "positivos": positives,
        "negativos": total - positives,
        "proporcion_positivos": positives / total if total else 0.0,
    }


def log_class_balance(y: pd.Series) -> dict[str, float]:
    """Calcula el balance de clases y lo registra en el log.

    Args:
        y: target binario (0 o 1).

    Returns:
        El mismo diccionario que ``class_balance``.
    """
    balance = class_balance(y)
    logger.info(
        "Balance de clases: %d meses con target, %d positivos (%.1f %%), %d negativos",
        balance["total"],
        balance["positivos"],
        100 * balance["proporcion_positivos"],
        balance["negativos"],
    )
    return balance


FEATURE_LABELS = {
    "temperatura_promedio": "la temperatura media del mes",
    "grados_dia_frio": "los grados-día de frío",
    "grados_dia_calor": "los grados-día de calor",
    "mes_sin": "el componente seno del mes",
    "mes_cos": "el componente coseno del mes",
    "anio": "el año",
    "potencia_maxima": "la potencia máxima",
    **{
        lag_column_name(column, 1): f"{label} del mes anterior"
        for column, label in (
            ("demanda_total", "la demanda total"),
            ("demanda_residencial", "la demanda residencial"),
            ("comercio_e_industria", "la demanda de comercio e industria"),
            ("grandes_usuarios", "la demanda de grandes usuarios"),
        )
    },
    **{
        lag_column_name(column, 12): f"{label} del mismo mes del año anterior"
        for column, label in (
            ("demanda_total", "la demanda total"),
            ("demanda_residencial", "la demanda residencial"),
            ("comercio_e_industria", "la demanda de comercio e industria"),
            ("grandes_usuarios", "la demanda de grandes usuarios"),
        )
    },
}


def feature_label(name: str) -> str:
    """Descripción legible de una variable; devuelve el mismo texto si no la conoce."""
    return FEATURE_LABELS.get(name, name)
