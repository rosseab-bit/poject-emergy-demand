"""Tests de src/predict.py: la fila de entrada del mes siguiente."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import data, db, features, logger, predict


@pytest.fixture(scope="module")
def clean() -> pd.DataFrame:
    """Datos limpios reales (308 meses), tal como los guarda el ETL."""
    return data.flag_repeated_power(data.load_raw())


def test_next_month_is_the_month_after_the_last_data(clean: pd.DataFrame) -> None:
    """Con los datos reales (hasta 2026-08) el mes siguiente es 2026-09."""
    assert predict.next_month(clean) == pd.Timestamp("2026-09-01")


def test_next_month_rolls_over_the_year() -> None:
    """Después de diciembre viene enero del año siguiente."""
    frame = pd.DataFrame({"fecha": pd.to_datetime(["2025-11-01", "2025-12-01"])})
    assert predict.next_month(frame) == pd.Timestamp("2026-01-01")


def test_next_month_without_data_asks_for_etl() -> None:
    """Sin datos pide ejecutar el ETL."""
    with pytest.raises(ValueError, match="ejecutá el ETL"):
        predict.next_month(pd.DataFrame({"fecha": pd.to_datetime([])}))


def test_default_temperature_is_the_historical_mean_of_that_month(
    clean: pd.DataFrame,
) -> None:
    """Es el promedio de ese mes calendario en todos los años."""
    september = clean[clean["fecha"].dt.month == 9]["temperatura_promedio"]
    assert predict.default_temperature(clean, 9) == pytest.approx(september.mean())
    assert 14 < predict.default_temperature(clean, 9) < 17


@pytest.mark.parametrize("bad", [-50.0, 99.0, float("nan"), float("inf")])
def test_validate_temperature_rejects_out_of_range_and_non_finite(bad: float) -> None:
    """Valores fuera de rango o no finitos lanzan ValueError."""
    with pytest.raises(ValueError, match="temperatura"):
        predict.validate_temperature(bad)


def test_validate_temperature_accepts_the_limits() -> None:
    """Los extremos del rango son válidos."""
    low, high = predict.TEMPERATURE_RANGE
    assert predict.validate_temperature(low) == low
    assert predict.validate_temperature(high) == high
    assert predict.validate_temperature("18.5") == 18.5


def test_build_input_real_data_for_the_next_month(clean: pd.DataFrame) -> None:
    """La fila del mes siguiente trae las variables del escenario A, sin nulos."""
    result = predict.build_input(clean, 15.0)
    row = result.features
    assert result.month == pd.Timestamp("2026-09-01")
    assert list(row.columns) == features.FEATURES_A
    assert len(row) == 1
    assert row.isna().sum().sum() == 0
    assert row.at[0, "temperatura_promedio"] == 15.0
    assert row.at[0, "grados_dia_frio"] == pytest.approx(3.0)
    assert row.at[0, "grados_dia_calor"] == 0.0
    assert row.at[0, "anio"] == 2026
    angle = 2 * np.pi * (9 - 1) / 12
    assert row.at[0, "mes_sin"] == pytest.approx(np.sin(angle))
    assert row.at[0, "mes_cos"] == pytest.approx(np.cos(angle))


def test_lags_come_from_the_last_month_and_the_same_month_last_year(
    clean: pd.DataFrame,
) -> None:
    """El rezago 1 es agosto de 2026 y el rezago 12 es septiembre de 2025."""
    result = predict.build_input(clean, 15.0)
    by_date = clean.set_index("fecha")
    for column in features.LAG_SOURCE_COLUMNS:
        assert (
            result.lags[f"{column}_lag1"]
            == by_date.at[pd.Timestamp("2026-08-01"), column]
        )
        assert (
            result.lags[f"{column}_lag12"]
            == by_date.at[pd.Timestamp("2025-09-01"), column]
        )
    assert set(result.lags) == set(features.LAG_COLUMNS)


def test_input_matches_exactly_what_training_computes_for_a_known_month(
    clean: pd.DataFrame,
) -> None:
    """Para un mes conocido, la fila armada con solo el pasado es idéntica a la
    que usó el entrenamiento: no hay diferencia entre predecir y entrenar."""
    dataset = features.build_dataset(clean)
    for target in ("2025-03-01", "2024-12-01", "2023-07-01"):
        position = dataset.index[dataset["fecha"] == pd.Timestamp(target)][0]
        past = clean.iloc[:position]
        real_temperature = clean.at[position, "temperatura_promedio"]
        built = predict.build_input(past, real_temperature)
        assert built.month == pd.Timestamp(target)
        expected = dataset.loc[[position], features.FEATURES_A].reset_index(drop=True)
        pd.testing.assert_frame_equal(built.features, expected, check_dtype=False)


def test_only_the_expected_history_months_influence_the_row(
    clean: pd.DataFrame,
) -> None:
    """Cambiar un mes viejo no altera la fila; cambiar agosto altera solo el rezago 1."""
    base = predict.build_input(clean, 15.0).features
    old = clean.copy()
    old.loc[old["fecha"] == pd.Timestamp("2010-05-01"), "demanda_total"] *= 3
    pd.testing.assert_frame_equal(predict.build_input(old, 15.0).features, base)

    recent = clean.copy()
    recent.loc[recent["fecha"] == pd.Timestamp("2026-08-01"), "demanda_total"] += 1000
    changed = predict.build_input(recent, 15.0).features
    differing = [
        c for c in base.columns if not np.isclose(base.at[0, c], changed.at[0, c])
    ]
    assert differing == ["demanda_total_lag1"]


def test_build_input_does_not_mutate_its_input(clean: pd.DataFrame) -> None:
    """Devuelve una fila nueva y deja intacta la tabla recibida."""
    before = clean.copy()
    predict.build_input(clean, 15.0)
    pd.testing.assert_frame_equal(clean, before)


def test_build_input_rejects_bad_history(clean: pd.DataFrame) -> None:
    """Con un mes faltante o poca historia falla en vez de inventar valores."""
    with_gap = clean.drop(index=clean.index[-5]).reset_index(drop=True)
    with pytest.raises(ValueError, match="Faltan meses"):
        predict.build_input(with_gap, 15.0)
    short = clean.iloc[-6:].reset_index(drop=True)
    with pytest.raises(ValueError, match="no alcanza"):
        predict.build_input(short, 15.0)
    with pytest.raises(ValueError, match="temperatura"):
        predict.build_input(clean, 200.0)


def test_build_input_adds_no_events_to_logs(
    clean: pd.DataFrame, tmp_path: Path
) -> None:
    """Armar la fila no escribe eventos en la tabla logs."""
    path = tmp_path / "structured.db"
    logger.setup_logging(path)
    logging.getLogger("src.predict").info("control: este sí se guarda")
    predict.build_input(clean, 15.0)
    with db.get_connection(path) as conn:
        messages = [r["message"] for r in conn.execute("SELECT message FROM logs")]
    assert messages == ["control: este sí se guarda"]
