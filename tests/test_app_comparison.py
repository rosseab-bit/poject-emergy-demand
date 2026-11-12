"""Tests de la página Comparación (AppTest sobre una corrida real entrenada)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app_pages import comparacion
from src import config, queries, results

APP_PATH = config.ROOT_DIR / "app.py"


def open_page() -> AppTest:
    """Abre la página Comparación."""
    at = AppTest.from_file(str(APP_PATH), default_timeout=120).run()
    return at.sidebar.radio[0].set_value("Comparación").run()


def test_comparison_without_results_asks_to_train(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sin resultados pide entrenar y no dibuja nada más."""
    monkeypatch.setattr(config, "STRUCTURED_DB_PATH", tmp_path / "s.db")
    monkeypatch.setattr(config, "UNSTRUCTURED_DB_PATH", tmp_path / "u.db")
    at = open_page()
    assert not at.exception
    assert any("Primero entrená" in w.value for w in at.warning)
    assert len(at.dataframe) == 0
    assert len(at.get("vega_lite_chart")) == 0


def test_comparison_shows_every_block(use_trained: dict) -> None:
    """Muestra tabla, gráfico, 4 matrices, variables y comparación A contra B."""
    at = open_page()
    assert not at.exception
    assert at.header[0].value == "Comparación"
    assert [s.value for s in at.main.subheader] == [
        "Métricas",
        "Matrices de confusión (hold-out)",
        "Variables que usa cada modelo",
        "Escenario A contra B: ¿cuánto aporta la potencia máxima?",
    ]
    # tabla de métricas + 4 matrices + comparación A contra B
    assert len(at.dataframe) == 6
    assert len(at.dataframe[0].value) == 4
    # gráfico de barras del F1 + gráfico de variables
    assert len(at.get("vega_lite_chart")) == 2


def test_comparison_does_not_declare_a_winner_and_explains_the_noise(
    use_trained: dict,
) -> None:
    """Avisa que no hay ganador, con los números reales del hold-out."""
    at = open_page()
    notice = " ".join(w.value for w in at.warning)
    assert "no se declara un modelo ganador" in notice
    assert "48 meses" in notice
    assert "solo 12" in notice
    assert "F1 = 0.40" in notice
    assert not any("recomendado" in m.value.lower() for m in at.markdown)


def test_confusion_matrices_match_the_stored_results(use_trained: dict) -> None:
    """Las matrices mostradas son las guardadas y suman los 48 meses."""
    at = open_page()
    docs = queries.result_docs(use_trained["run_id"])
    shown = [df.value for df in at.dataframe[1:5]]
    assert len(shown) == len(docs) == 4
    for frame, doc in zip(shown, docs, strict=True):
        assert frame.to_numpy().tolist() == doc["holdout"]["confusion_matrix"]
        assert frame.to_numpy().sum() == 48
        assert list(frame.index) == ["Real: normal", "Real: alta"]
        assert list(frame.columns) == ["Predicho: normal", "Predicho: alta"]


def test_scenario_table_shows_b_minus_a(use_trained: dict) -> None:
    """La tabla A contra B tiene un modelo por fila y la diferencia B menos A."""
    at = open_page()
    shown = at.dataframe[-1].value
    assert shown["Modelo"].tolist() == ["Regresión logística", "Random Forest"]
    diff = shown["F1 hold-out B"] - shown["F1 hold-out A"]
    assert shown["Diferencia (B − A)"].tolist() == pytest.approx(diff.tolist())


def test_variable_selector_switches_between_coefficients_and_importances(
    use_trained: dict,
) -> None:
    """Regresión logística muestra coeficientes; Random Forest, importancias."""
    at = open_page()
    captions = lambda: " ".join(c.value for c in at.caption)
    assert at.selectbox[0].value == ("A", "logistic_regression")
    assert "Coeficientes sobre variables estandarizadas" in captions()
    at.selectbox[0].set_value(("B", "random_forest")).run()
    assert not at.exception
    assert "Importancias (suman 1)" in captions()
    assert "Coeficientes sobre variables estandarizadas" not in captions()


def test_f1_chart_has_one_series_and_the_reference_line() -> None:
    """El gráfico no usa color (una sola serie) y dibuja la referencia."""
    spec = comparacion.f1_chart_spec(0.4)
    layers = spec["spec"]["layer"]
    assert [layer["mark"]["type"] for layer in layers] == ["bar", "rule", "text"]
    assert all("color" not in layer["encoding"] for layer in layers)
    assert layers[1]["encoding"]["y"]["datum"] == 0.4
    assert "0.40" in layers[2]["encoding"]["text"]["value"]
    assert layers[0]["encoding"]["y"]["scale"]["domain"] == [0, 1]


def test_comparison_uses_the_saved_model_not_memory(use_trained: dict) -> None:
    """Las importancias salen del .joblib guardado, verificando su hash."""
    doc = results.load_result(use_trained["run_id"], "A", "logistic_regression")
    path = use_trained["models_dir"] / doc["model_file"]["name"]
    assert path.is_file()
    table = queries.feature_importance_for(
        use_trained["run_id"], "A", "logistic_regression"
    )
    assert len(table) == len(doc["features"])
