"""Tests de la página Predicción (AppTest sobre una corrida real entrenada)."""

from functools import partial
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from support import open_page as open_page_in
from support import start_app

from src import data, db, pipeline, predict, queries

open_page = partial(open_page_in, "Predicción")


def predict_click(at: AppTest) -> AppTest:
    """Hace clic en «Predecir»."""
    return at.button[0].click().run()


def test_prediction_without_runs_asks_to_train(tmp_config: Path) -> None:
    """Sin corridas pide entrenar y no ofrece predecir."""
    at = open_page()
    assert not at.exception
    assert at.header[0].value == "Predicción"
    assert any("Primero entrená" in w.value for w in at.warning)
    assert len(at.button) == 0


def test_prediction_with_untrained_run_asks_to_train(tmp_config: Path) -> None:
    """Una corrida con solo el ETL tampoco permite predecir."""
    pipeline.run_etl()
    at = open_page()
    assert any("Primero entrená" in w.value for w in at.warning)
    assert len(at.button) == 0


def test_prediction_requires_scenario_a(tmp_config: Path) -> None:
    """Si solo se entrenó el escenario B, avisa que falta el A."""
    run_id = pipeline.run_etl()
    pipeline.run_training(
        run_id, scenarios=("B",), model_names=("logistic_regression",)
    )
    at = open_page()
    assert any("escenario A" in w.value for w in at.warning)
    assert len(at.button) == 0


def test_form_shows_the_fixed_month_the_default_temperature_and_the_lags(
    use_trained: dict,
) -> None:
    """Mes y año fijos y deshabilitados, temperatura por defecto y rezagos reales."""
    at = open_page()
    assert not at.exception
    month, year, temperature = at.main.number_input
    assert (month.value, month.disabled) == (9, True)
    assert (year.value, year.disabled) == (2026, True)
    clean = data.load_clean_data(use_trained["db_path"])
    assert temperature.value == pytest.approx(
        predict.default_temperature(clean, 9), abs=0.05
    )
    assert (temperature.min, temperature.max) == predict.TEMPERATURE_RANGE
    assert any("septiembre de 2026" in m.value for m in at.markdown)
    assert any("nivel mensual" in m.value for m in at.markdown)

    lags = at.dataframe[0].value
    assert lags["Variable"].tolist() == [
        "Demanda total",
        "Demanda residencial",
        "Comercio e industria",
        "Grandes usuarios",
    ]
    assert "ago 2026" in lags.columns[1] and "sep 2025" in lags.columns[2]
    by_date = clean.set_index("fecha")
    assert lags.iloc[0, 1] == pytest.approx(
        by_date.at[pd.Timestamp("2026-08-01"), "demanda_total"], abs=0.1
    )
    assert lags.iloc[0, 2] == pytest.approx(
        by_date.at[pd.Timestamp("2025-09-01"), "demanda_total"], abs=0.1
    )


def test_predict_button_saves_and_shows_the_result(use_trained: dict) -> None:
    """Predecir guarda la predicción, avisa y muestra una fila por modelo."""
    before = len(queries.prediction_history(None))
    at = open_page()
    at.main.number_input[2].set_value(21.5).run()
    at = predict_click(at)
    assert not at.exception
    assert any("Predicción guardada" in s.value for s in at.success)
    assert len(queries.prediction_history(None)) == before + 1

    latest = queries.latest_prediction(use_trained["run_id"])
    assert latest["temperatura"] == 21.5
    assert [s.value for s in at.main.subheader][-2:] == [
        "Último resultado",
        "Historial de predicciones guardadas",
    ]
    result = at.dataframe[1].value
    assert result["Modelo"].tolist() == ["Regresión logística", "Random Forest"]
    assert result["Probabilidad de alta"].between(0, 1).all()
    assert set(result["Clasificación"]) <= {"alta", "normal"}
    assert set(result["¿Supera la referencia?"]) <= {"Sí", "No"}
    assert result["F1 «siempre alta»"].tolist() == pytest.approx([0.4, 0.4])
    stored = latest["modelos"]["logistic_regression"]["probabilidad"]
    assert result["Probabilidad de alta"].iloc[0] == pytest.approx(stored, abs=1e-3)


def test_result_explains_calibration_and_the_reference(use_trained: dict) -> None:
    """Junto al resultado se advierte que no está calibrado y qué pasa con la referencia."""
    at = predict_click(open_page())
    warnings = " ".join(w.value for w in at.warning)
    assert "No están calibradas" in warnings
    assert "«90 % de chances»" in warnings
    verdict = " ".join(x.value for x in [*at.info, *at.error])
    assert "«siempre alta»" in verdict
    assert "solo 12" in verdict or "12 de" in verdict


def test_result_and_history_survive_reopening_the_app(use_trained: dict) -> None:
    """Una sesión nueva ve el resultado y el historial sin volver a predecir."""
    predict_click(open_page())
    at = open_page()  # sesión nueva: no hay nada en la memoria de Streamlit
    assert not at.exception
    assert any(s.value == "Último resultado" for s in at.main.subheader)
    history = at.dataframe[-1].value
    assert len(history) >= 1
    assert "P(alta) Regresión logística" in history.columns
    assert "P(alta) XGBoost" not in history.columns  # no se entrenó en esta corrida


def test_history_scope_can_show_all_runs(use_trained: dict) -> None:
    """El historial puede mostrar solo la corrida actual o todas."""
    at = predict_click(open_page())
    assert "Corrida" not in at.dataframe[-1].value.columns
    at.radio[0].set_value("Todas las corridas").run()
    assert not at.exception
    assert "Corrida" in at.dataframe[-1].value.columns


def test_prediction_failure_is_shown_not_raised(
    use_trained: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Si falla la predicción, se muestra el error y la app sigue."""

    def boom(*args, **kwargs):
        raise RuntimeError("modelo ilegible")

    monkeypatch.setattr(predict, "predict_next_month", boom)
    at = predict_click(open_page())
    assert not at.exception
    assert any("modelo ilegible" in e.value for e in at.error)


def test_viewing_the_page_adds_no_events_to_logs(use_trained: dict) -> None:
    """Mirar la página no escribe en logs; solo el clic en Predecir lo hace."""

    def count() -> int:
        return len(queries.get_logs(db_path=use_trained["db_path"]))

    before = count()
    at = open_page()
    assert count() == before
    predict_click(at)
    assert count() > before
    assert queries.get_logs(db_path=use_trained["db_path"])["run_id"].isna().sum() == 0


def test_page_does_not_break_other_pages_without_data(tmp_config: Path) -> None:
    """Con bases vacías todas las páginas siguen abriendo."""
    at = start_app(60)
    for page in ("Predicción", "Reporte", "Datos"):
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, page
    assert db.get_run("noexiste") is None


def test_sidebar_marks_the_prediction_step_as_done(use_trained: dict) -> None:
    """Tras predecir, la barra lateral muestra el paso Predicción como hecho."""
    at = open_page()
    side = lambda: "\n".join(m.value for m in at.sidebar.markdown)
    assert "⬜ Reporte — pendiente" in side()
    at = predict_click(at)
    assert not at.exception
    assert "✅ Predicción — hecho" in "\n".join(m.value for m in at.sidebar.markdown)
    assert "⬜ Reporte — pendiente" in "\n".join(m.value for m in at.sidebar.markdown)
