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


SECTOR_START = pd.Timestamp("2005-01-01")


@log_step
def check_sector_nulls(df: pd.DataFrame) -> int:
    """Verifica que los nulos de los sectores sean solo anteriores a 2005.

    Los sectores (residencial, comercio e industria, grandes usuarios) no
    existen en la fuente antes de 2005-01. No se imputan: esos meses quedan
    con nulos y se usan solo por su ``demanda_total`` como historia.

    Args:
        df: datos con ``fecha`` y las columnas de ``SECTOR_COLUMNS``.

    Returns:
        Cantidad de filas con algún sector nulo.

    Raises:
        ValueError: si hay sectores nulos desde 2005-01 en adelante.
    """
    null_rows = df[SECTOR_COLUMNS].isna().any(axis=1)
    unexpected = null_rows & (df[DATE_COLUMN] >= SECTOR_START)
    if unexpected.any():
        months = df.loc[unexpected, DATE_COLUMN].dt.strftime("%Y-%m").tolist()
        raise ValueError(f"Sectores nulos desde 2005-01: {months}")
    logger.info(
        "Filas con sectores nulos (anteriores a 2005): %d", int(null_rows.sum())
    )
    return int(null_rows.sum())


def count_complete_rows(df: pd.DataFrame) -> int:
    """Cuenta las filas sin ningún nulo en las columnas de valores.

    Args:
        df: datos con ``VALUE_COLUMNS``.

    Returns:
        Cantidad de filas completas.
    """
    return int(df[VALUE_COLUMNS].notna().all(axis=1).sum())
