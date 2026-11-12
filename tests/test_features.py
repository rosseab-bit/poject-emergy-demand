"""Tests de src/features.py."""

import numpy as np
import pandas as pd
import pytest

from src import config, data, features


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


def make_demand_frame(n: int = 30) -> pd.DataFrame:
    """DataFrame con demanda total y sectores crecientes (valores distintos)."""
    df = pd.DataFrame({"fecha": pd.date_range("2005-01-01", periods=n, freq="MS")})
    df["demanda_total"] = np.arange(n, dtype=float) + 1000
    df["demanda_residencial"] = np.arange(n, dtype=float) + 100
    df["comercio_e_industria"] = np.arange(n, dtype=float) + 200
    df["grandes_usuarios"] = np.arange(n, dtype=float) + 300
    return df


def test_lags_take_value_from_k_months_before() -> None:
    """El rezago k de la fila t es el valor de la fila t-k."""
    df = make_demand_frame()
    out = features.add_lag_features(df)
    for col in features.LAG_SOURCE_COLUMNS:
        for lag in features.LAGS:
            name = features.lag_column_name(col, lag)
            assert out[name].iloc[lag:].tolist() == df[col].iloc[:-lag].tolist()


def test_lags_are_null_without_history() -> None:
    """Los primeros k meses de cada rezago quedan en nulo."""
    out = features.add_lag_features(make_demand_frame())
    assert out["demanda_total_lag1"].iloc[:1].isna().all()
    assert out["demanda_total_lag12"].iloc[:12].isna().all()
    assert out["demanda_total_lag12"].iloc[12:].notna().all()


def test_lags_do_not_use_future_values() -> None:
    """Cambiar un valor futuro no altera los rezagos de meses anteriores."""
    df = make_demand_frame()
    base = features.add_lag_features(df)
    changed = df.copy()
    changed.loc[20:, features.LAG_SOURCE_COLUMNS] = 999999.0
    out = features.add_lag_features(changed)
    lag_cols = [c for c in out.columns if "_lag" in c]
    # Los rezagos de las filas 0..20 dependen solo de filas <= 19.
    pd.testing.assert_frame_equal(base.loc[:20, lag_cols], out.loc[:20, lag_cols])


def test_lags_reject_non_consecutive_months() -> None:
    """Con un mes faltante falla en vez de desalinear los rezagos."""
    df = make_demand_frame().drop(index=5).reset_index(drop=True)
    with pytest.raises(ValueError, match="Faltan meses"):
        features.add_lag_features(df)


def test_lags_do_not_mutate_input() -> None:
    """Devuelve una copia."""
    df = make_demand_frame()
    features.add_lag_features(df)
    assert "demanda_total_lag1" not in df.columns


def test_lags_real_data_sector_lag12_starts_in_2006() -> None:
    """Con los datos reales, el rezago 12 de los sectores existe desde 2006-01."""
    out = features.add_lag_features(data.flag_repeated_power(data.load_raw()))
    col = features.lag_column_name("demanda_residencial", 12)
    first_valid = out.loc[out[col].notna(), "fecha"].iloc[0]
    assert first_valid == pd.Timestamp("2006-01-01")
    assert out[features.lag_column_name("demanda_total", 12)].iloc[:12].isna().all()
