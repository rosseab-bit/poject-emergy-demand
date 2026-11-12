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


def make_target_frame(n: int = 80, seed: int = 0) -> pd.DataFrame:
    """Demanda mensual sintética con estacionalidad, tendencia y ruido."""
    rng = np.random.default_rng(seed)
    months = np.arange(n)
    demand = 1000 + 5 * months + 80 * np.sin(2 * np.pi * months / 12)
    demand = demand + rng.normal(0, 25, n)
    return pd.DataFrame(
        {
            "fecha": pd.date_range("2001-01-01", periods=n, freq="MS"),
            "demanda_total": demand,
        }
    )


def test_ratio_is_demand_over_mean_of_same_month_previous_years() -> None:
    """El cociente usa los meses 12, 24 y 36 atrás."""
    df = make_target_frame()
    d = df["demanda_total"]
    ratio = features.compute_demand_ratio(d)
    t = 50
    expected = d[t] / np.mean([d[t - 12], d[t - 24], d[t - 36]])
    assert ratio[t] == pytest.approx(expected)
    assert ratio.iloc[:36].isna().all()
    assert ratio.iloc[36:].notna().all()


def test_threshold_is_quantile_of_previous_ratios_only() -> None:
    """El umbral de t es el cuantil de los 36 cocientes anteriores, sin t."""
    out = features.add_target(make_target_frame())
    t = 75
    previous = out["cociente_demanda"].iloc[t - 36 : t]
    assert out["umbral_demanda"].iloc[t] == pytest.approx(previous.quantile(0.75))
    assert out["demanda_alta"].iloc[t] == int(
        out["cociente_demanda"].iloc[t] > out["umbral_demanda"].iloc[t]
    )


def test_target_is_null_without_enough_history() -> None:
    """Sin 36 meses de historia de cocientes el target es nulo."""
    out = features.add_target(make_target_frame())
    first_valid = 36 + 36
    assert out["demanda_alta"].iloc[:first_valid].isna().all()
    assert out["demanda_alta"].iloc[first_valid:].notna().all()
    assert set(out["demanda_alta"].dropna().unique()) <= {0, 1}


def test_target_does_not_use_future_data_truncation() -> None:
    """El target de cada mes es el mismo con la serie completa o cortada ahí."""
    df = make_target_frame()
    full = features.add_target(df)
    for t in (72, 75, 79):
        cut = features.add_target(df.iloc[: t + 1])
        assert cut["demanda_alta"].iloc[t] == full["demanda_alta"].iloc[t]
        pd.testing.assert_series_equal(
            cut["demanda_alta"], full["demanda_alta"].iloc[: t + 1]
        )


def test_target_does_not_change_when_future_values_change() -> None:
    """Alterar la demanda futura no cambia el target de meses anteriores."""
    df = make_target_frame()
    base = features.add_target(df)
    changed = df.copy()
    changed.loc[70:, "demanda_total"] *= 5
    out = features.add_target(changed)
    cols = ["cociente_demanda", "umbral_demanda", "demanda_alta"]
    pd.testing.assert_frame_equal(base.loc[:69, cols], out.loc[:69, cols])


def test_target_rejects_non_consecutive_months() -> None:
    """Con un mes faltante falla."""
    df = make_target_frame().drop(index=10).reset_index(drop=True)
    with pytest.raises(ValueError, match="Faltan meses"):
        features.add_target(df)


def test_target_real_data_matches_plan() -> None:
    """Con los datos reales: 236 meses con target desde 2007-01, 60 positivos."""
    out = features.add_target(data.flag_repeated_power(data.load_raw()))
    valid = out.dropna(subset=["demanda_alta"])
    assert len(valid) == 236
    assert valid["fecha"].iloc[0] == pd.Timestamp("2007-01-01")
    assert int(valid["demanda_alta"].sum()) == 60
    assert int(valid["demanda_alta"].iloc[-36:].sum()) == 7
