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
