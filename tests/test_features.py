"""Tests de src/features.py."""

import numpy as np
import pandas as pd
import pytest

from src import config, features


def make_frame(temps: list[float], start: str = "2020-01-01") -> pd.DataFrame:
    """DataFrame mínimo con fecha y temperatura."""
    return pd.DataFrame(
        {
            "fecha": pd.date_range(start, periods=len(temps), freq="MS"),
            "temperatura_promedio": temps,
        }
    )


def test_degree_days_split_cold_and_heat() -> None:
    """Frío y calor se separan respecto de la base de 18 °C."""
    out = features.add_temperature_features(make_frame([8.0, 18.0, 28.0]))
    assert out["grados_dia_frio"].tolist() == [10.0, 0.0, 0.0]
    assert out["grados_dia_calor"].tolist() == [0.0, 0.0, 10.0]


def test_degree_days_default_base_is_config() -> None:
    """La base por defecto sale de config.BASE_TEMP_C."""
    assert config.BASE_TEMP_C == 18.0
    out = features.add_temperature_features(make_frame([config.BASE_TEMP_C]))
    assert out["grados_dia_frio"].iloc[0] == 0.0


def test_degree_days_custom_base_and_no_mutation() -> None:
    """Acepta otra base y no modifica el DataFrame original."""
    df = make_frame([20.0])
    out = features.add_temperature_features(df, base=22.0)
    assert out["grados_dia_frio"].iloc[0] == 2.0
    assert "grados_dia_frio" not in df.columns


def test_degree_days_never_negative_and_one_is_zero() -> None:
    """Nunca son negativos y a lo sumo uno de los dos es distinto de cero."""
    out = features.add_temperature_features(make_frame(list(np.arange(0, 35, 0.5))))
    assert (out["grados_dia_frio"] >= 0).all()
    assert (out["grados_dia_calor"] >= 0).all()
    assert ((out["grados_dia_frio"] * out["grados_dia_calor"]) == 0).all()


def test_calendar_features_cyclic_and_year() -> None:
    """Enero queda en (0, 1), julio en (0, -1) y se agrega el año."""
    out = features.add_calendar_features(make_frame([20.0] * 12))
    assert out["mes_sin"].iloc[0] == pytest.approx(0.0)
    assert out["mes_cos"].iloc[0] == pytest.approx(1.0)
    assert out["mes_sin"].iloc[6] == pytest.approx(0.0, abs=1e-9)
    assert out["mes_cos"].iloc[6] == pytest.approx(-1.0)
    assert out["anio"].tolist() == [2020] * 12


def test_calendar_december_is_close_to_january() -> None:
    """Diciembre y enero están más cerca que diciembre y junio."""
    out = features.add_calendar_features(make_frame([20.0] * 12))
    pts = out[["mes_sin", "mes_cos"]].to_numpy()
    assert np.linalg.norm(pts[11] - pts[0]) < np.linalg.norm(pts[11] - pts[5])
