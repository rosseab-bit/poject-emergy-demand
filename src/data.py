"""Carga y limpieza del dataset de demanda eléctrica mensual.

El CSV de ``data/raw/`` es de solo lectura: nunca se modifica.
"""

import logging
from pathlib import Path

import pandas as pd

from src import config
from src.logger import log_step

logger = logging.getLogger(__name__)

DATE_COLUMN = "fecha"
RAW_DATE_COLUMN = "indice_tiempo"
SECTOR_COLUMNS = ["demanda_residencial", "comercio_e_industria", "grandes_usuarios"]
VALUE_COLUMNS = [
    "demanda_total",
    *SECTOR_COLUMNS,
    "temperatura_promedio",
    "potencia_maxima",
]


def validate_monthly_dates(dates: pd.Series) -> None:
    """Verifica que las fechas sean únicas, ordenadas y un mes tras otro.

    Args:
        dates: serie de fechas (``datetime64``).

    Raises:
        ValueError: si hay nulos, duplicados, desorden, días distintos del 1
            o meses faltantes.
    """
    if dates.isna().any():
        raise ValueError("Hay fechas nulas o no interpretables.")
    if dates.duplicated().any():
        raise ValueError("Hay fechas duplicadas.")
    if not dates.is_monotonic_increasing:
        raise ValueError("Las fechas no están ordenadas.")
    if (dates.dt.day != 1).any():
        raise ValueError("Hay fechas que no son el primer día del mes.")
    expected = pd.date_range(dates.iloc[0], dates.iloc[-1], freq="MS")
    if len(expected) != len(dates):
        raise ValueError("Faltan meses en la serie.")


@log_step
def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Lee el CSV original y devuelve solo las columnas que usa el proyecto.

    Las columnas por distribuidora no se usan (la temperatura es nacional).

    Args:
        path: ruta del CSV; por defecto ``config.RAW_CSV_PATH``.

    Returns:
        DataFrame con ``fecha`` y ``VALUE_COLUMNS``, una fila por mes.

    Raises:
        ValueError: si faltan columnas o la serie de fechas no es válida.
    """
    path = path if path is not None else config.RAW_CSV_PATH
    df = pd.read_csv(path)
    missing = {RAW_DATE_COLUMN, *VALUE_COLUMNS} - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en el CSV: {sorted(missing)}")
    df = df[[RAW_DATE_COLUMN, *VALUE_COLUMNS]].rename(
        columns={RAW_DATE_COLUMN: DATE_COLUMN}
    )
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    validate_monthly_dates(df[DATE_COLUMN])
    logger.info(
        "CSV leído: %d filas, de %s a %s",
        len(df),
        df[DATE_COLUMN].iloc[0].strftime("%Y-%m"),
        df[DATE_COLUMN].iloc[-1].strftime("%Y-%m"),
    )
    return df


@log_step
def flag_repeated_power(df: pd.DataFrame) -> pd.DataFrame:
    """Marca y anula los valores de potencia máxima repetidos del mes anterior.

    Un valor idéntico al del mes anterior se toma como error de carga: se
    pone en nulo (lo imputa después el ``Pipeline``, solo con datos de
    entrenamiento) y se deja la marca ``potencia_repetida`` (0 o 1).

    Args:
        df: datos con ``potencia_maxima``, ordenados por fecha.

    Returns:
        Copia de ``df`` con la columna ``potencia_repetida`` agregada.
    """
    out = df.copy()
    power = out["potencia_maxima"]
    repeated = power.notna() & power.eq(power.shift())
    out["potencia_repetida"] = repeated.astype(int)
    out.loc[repeated, "potencia_maxima"] = pd.NA
    for date in out.loc[repeated, DATE_COLUMN]:
        logger.warning(
            "Potencia máxima repetida del mes anterior en %s: se pone en nulo",
            date.strftime("%Y-%m"),
        )
    logger.info("Valores de potencia máxima repetidos: %d", int(repeated.sum()))
    return out
