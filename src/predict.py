"""Predicción del mes siguiente con los modelos guardados de una corrida."""

import math
from dataclasses import dataclass

import pandas as pd

from src import features, logger
from src.data import DATE_COLUMN, validate_monthly_dates

# Rango razonable de la temperatura media mensual (°C) para el formulario.
TEMPERATURE_RANGE = (-5.0, 45.0)


@dataclass
class PredictionInput:
    """Entrada de una predicción: el mes, la temperatura y las variables del modelo.

    Attributes:
        month: primer día del mes a predecir.
        temperature: temperatura media esperada del mes (°C).
        features: una fila con las variables del escenario A.
        lags: rezagos completados desde ``datos_limpios`` (nombre a valor).
    """

    month: pd.Timestamp
    temperature: float
    features: pd.DataFrame
    lags: dict[str, float]


def next_month(clean: pd.DataFrame) -> pd.Timestamp:
    """Devuelve el mes siguiente al último dato disponible.

    Es el único mes que se puede predecir: sus rezagos de 1 y 12 meses ya se
    conocen. Para meses más lejanos faltaría la demanda del mes anterior.

    Args:
        clean: datos de ``datos_limpios``, ordenados por fecha.

    Returns:
        Primer día del mes siguiente al último dato.

    Raises:
        ValueError: si no hay datos.
    """
    if clean.empty:
        raise ValueError("No hay datos: primero ejecutá el ETL.")
    return clean[DATE_COLUMN].iloc[-1] + pd.offsets.MonthBegin(1)


def default_temperature(clean: pd.DataFrame, month: int) -> float:
    """Temperatura media histórica de un mes calendario, como valor inicial.

    Args:
        clean: datos de ``datos_limpios``.
        month: mes calendario (1 a 12).

    Returns:
        Promedio de ``temperatura_promedio`` de ese mes en todos los años.
    """
    same_month = clean[clean[DATE_COLUMN].dt.month == month]
    return float(same_month[features.TEMPERATURE_COLUMN].mean())


def validate_temperature(temperature: float) -> float:
    """Verifica que la temperatura sea un número dentro del rango razonable.

    Args:
        temperature: temperatura media esperada (°C).

    Returns:
        La temperatura como ``float``.

    Raises:
        ValueError: si no es un número finito o está fuera de ``TEMPERATURE_RANGE``.
    """
    value = float(temperature)
    low, high = TEMPERATURE_RANGE
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(
            f"La temperatura debe estar entre {low} y {high} °C: {temperature!r}"
        )
    return value


def build_input(clean: pd.DataFrame, temperature: float) -> PredictionInput:
    """Arma la fila de entrada del mes siguiente, con los rezagos desde la base.

    Agrega al historial una fila del mes siguiente con la temperatura esperada y
    calcula las variables con las mismas funciones del entrenamiento
    (``features``), así los rezagos salen solo de datos ya conocidos.

    Args:
        clean: datos de ``datos_limpios``, ordenados y sin meses faltantes.
        temperature: temperatura media esperada del mes (°C).

    Returns:
        ``PredictionInput`` con las variables del escenario A.

    Raises:
        ValueError: si no hay datos, la temperatura es inválida, faltan meses o
            el historial no alcanza para algún rezago.
    """
    month = next_month(clean)
    temperature = validate_temperature(temperature)
    validate_monthly_dates(clean[DATE_COLUMN])
    new_row = pd.DataFrame(
        {DATE_COLUMN: [month], features.TEMPERATURE_COLUMN: [temperature]}
    )
    history = pd.concat([clean, new_row], ignore_index=True)
    with logger.muted():  # las funciones de features no ensucian el historial
        history = features.add_temperature_features(history)
        history = features.add_calendar_features(history)
        history = features.add_lag_features(history)
    row = history.iloc[[-1]][features.FEATURES_A].reset_index(drop=True)
    missing = row.columns[row.isna().any()].tolist()
    if missing:
        raise ValueError(f"El historial no alcanza para calcular: {missing}")
    lags = {name: float(row.at[0, name]) for name in features.LAG_COLUMNS}
    return PredictionInput(month, temperature, row, lags)
