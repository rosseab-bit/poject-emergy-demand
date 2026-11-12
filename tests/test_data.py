"""Tests de src/data.py."""

from pathlib import Path

import pandas as pd
import pytest

from src import data


def write_csv(path: Path, rows: list[dict]) -> Path:
    """Escribe un CSV mínimo con las columnas del dataset."""
    cols = [data.RAW_DATE_COLUMN, *data.VALUE_COLUMNS, "extra_distribuidora"]
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)
    return path


def make_rows(dates: list[str]) -> list[dict]:
    """Arma filas de prueba para las fechas dadas."""
    return [
        {
            data.RAW_DATE_COLUMN: d,
            "demanda_total": 100.0 + i,
            "demanda_residencial": 40.0,
            "comercio_e_industria": 30.0,
            "grandes_usuarios": 30.0,
            "temperatura_promedio": 20.0,
            "potencia_maxima": 200.0 + i,
            "extra_distribuidora": 1.0,
        }
        for i, d in enumerate(dates)
    ]


def test_load_raw_keeps_only_used_columns(tmp_path: Path) -> None:
    """Devuelve fecha y columnas de valores, sin las de distribuidoras."""
    path = write_csv(tmp_path / "a.csv", make_rows(["2020-01-01", "2020-02-01"]))
    df = data.load_raw(path)
    assert list(df.columns) == [data.DATE_COLUMN, *data.VALUE_COLUMNS]
    assert len(df) == 2
    assert pd.api.types.is_datetime64_any_dtype(df[data.DATE_COLUMN])


def test_load_raw_real_csv() -> None:
    """El CSV real se lee completo: 308 filas de 2001-01 a 2026-08."""
    df = data.load_raw()
    assert len(df) == 308
    assert df[data.DATE_COLUMN].iloc[0] == pd.Timestamp("2001-01-01")
    assert df[data.DATE_COLUMN].iloc[-1] == pd.Timestamp("2026-08-01")


def test_load_raw_missing_column(tmp_path: Path) -> None:
    """Si falta una columna requerida lanza ValueError."""
    pd.DataFrame({data.RAW_DATE_COLUMN: ["2020-01-01"]}).to_csv(
        tmp_path / "b.csv", index=False
    )
    with pytest.raises(ValueError, match="Faltan columnas"):
        data.load_raw(tmp_path / "b.csv")


@pytest.mark.parametrize(
    ("dates", "message"),
    [
        (["2020-01-01", "2020-01-01"], "duplicadas"),
        (["2020-02-01", "2020-01-01"], "ordenadas"),
        (["2020-01-15", "2020-02-15"], "primer día"),
        (["2020-01-01", "2020-03-01"], "Faltan meses"),
        (["2020-01-01", "no es fecha"], "nulas"),
    ],
)
def test_load_raw_invalid_dates(tmp_path: Path, dates: list[str], message: str) -> None:
    """Fechas duplicadas, desordenadas, irregulares o ilegibles fallan."""
    path = write_csv(tmp_path / "c.csv", make_rows(dates))
    with pytest.raises(ValueError, match=message):
        data.load_raw(path)


def make_power_frame(values: list[float | None]) -> pd.DataFrame:
    """DataFrame mínimo con fecha y potencia máxima."""
    dates = pd.date_range("2020-01-01", periods=len(values), freq="MS")
    return pd.DataFrame({data.DATE_COLUMN: dates, "potencia_maxima": values})


def test_flag_repeated_power_nulls_the_repeat() -> None:
    """El valor igual al del mes anterior queda nulo y marcado."""
    df = make_power_frame([100.0, 110.0, 110.0, 120.0])
    out = data.flag_repeated_power(df)
    assert out["potencia_repetida"].tolist() == [0, 0, 1, 0]
    assert out["potencia_maxima"].isna().tolist() == [False, False, True, False]
    assert out["potencia_maxima"].iloc[1] == 110.0


def test_flag_repeated_power_does_not_mutate_input() -> None:
    """La función devuelve una copia."""
    df = make_power_frame([100.0, 100.0])
    data.flag_repeated_power(df)
    assert df["potencia_maxima"].tolist() == [100.0, 100.0]
    assert "potencia_repetida" not in df.columns


def test_flag_repeated_power_without_repeats() -> None:
    """Sin repetidos no cambia nada."""
    df = make_power_frame([100.0, 101.0, 102.0])
    out = data.flag_repeated_power(df)
    assert out["potencia_repetida"].sum() == 0
    assert out["potencia_maxima"].notna().all()


def test_flag_repeated_power_ignores_existing_nulls() -> None:
    """Dos nulos seguidos no cuentan como repetición."""
    df = make_power_frame([100.0, None, None, 120.0])
    out = data.flag_repeated_power(df)
    assert out["potencia_repetida"].sum() == 0


def test_flag_repeated_power_real_csv() -> None:
    """En el CSV real se detectan exactamente 2002-03 y 2021-08."""
    out = data.flag_repeated_power(data.load_raw())
    flagged = out.loc[out["potencia_repetida"] == 1, data.DATE_COLUMN]
    assert flagged.dt.strftime("%Y-%m").tolist() == ["2002-03", "2021-08"]
    assert out["potencia_maxima"].isna().sum() == 2


def make_sector_frame(dates: list[str], null_dates: list[str]) -> pd.DataFrame:
    """DataFrame con sectores completos salvo en las fechas indicadas."""
    df = pd.DataFrame({data.DATE_COLUMN: pd.to_datetime(dates)})
    for col in data.SECTOR_COLUMNS:
        df[col] = 1.0
    df.loc[
        df[data.DATE_COLUMN].isin(pd.to_datetime(null_dates)), data.SECTOR_COLUMNS
    ] = None
    return df


def test_check_sector_nulls_accepts_pre_2005() -> None:
    """Nulos antes de 2005 son esperados y se cuentan."""
    df = make_sector_frame(
        ["2004-11-01", "2004-12-01", "2005-01-01"], ["2004-11-01", "2004-12-01"]
    )
    assert data.check_sector_nulls(df) == 2


def test_check_sector_nulls_rejects_unexpected() -> None:
    """Un nulo desde 2005 en adelante lanza ValueError."""
    df = make_sector_frame(["2004-12-01", "2005-01-01"], ["2005-01-01"])
    with pytest.raises(ValueError, match="2005-01"):
        data.check_sector_nulls(df)


def test_count_complete_rows() -> None:
    """Cuenta solo las filas sin ningún nulo."""
    df = pd.DataFrame({c: [1.0, 2.0, 3.0] for c in data.VALUE_COLUMNS})
    df.loc[1, "temperatura_promedio"] = None
    assert data.count_complete_rows(df) == 2


def test_real_csv_null_summary() -> None:
    """En el CSV real: 48 filas con sectores nulos y 260 completas."""
    df = data.load_raw()
    assert data.check_sector_nulls(df) == 48
    assert data.count_complete_rows(df) == 260
    cleaned = data.flag_repeated_power(df)
    assert data.count_complete_rows(cleaned) == 259
