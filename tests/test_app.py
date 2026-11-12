"""Tests de la interfaz de Streamlit con AppTest (sin abrir un navegador)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app_pages.sidebar import PAGE_NAMES
from src import config, db, models, pipeline, queries, train

APP_PATH = config.ROOT_DIR / "app.py"


@pytest.fixture
def tmp_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Apunta las bases y los modelos de config a una carpeta temporal."""
    monkeypatch.setattr(config, "STRUCTURED_DB_PATH", tmp_path / "structured.db")
    monkeypatch.setattr(config, "UNSTRUCTURED_DB_PATH", tmp_path / "unstructured.db")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    return tmp_path


def run_app() -> AppTest:
    """Ejecuta app.py y devuelve el resultado."""
    return AppTest.from_file(str(APP_PATH), default_timeout=60).run()


def sidebar_text(at: AppTest) -> str:
    """Todo el texto markdown de la barra lateral, en un solo string."""
    return "\n".join(m.value for m in at.sidebar.markdown)


def test_app_starts_without_runs(tmp_config: Path) -> None:
    """Sin corridas la app arranca, avisa y muestra la semilla por defecto."""
    at = run_app()
    assert not at.exception
    assert any("Todavía no hay corridas" in i.value for i in at.sidebar.info)
    assert at.sidebar.number_input[0].value == config.SEED
    assert at.sidebar.radio[0].options == list(PAGE_NAMES)
    assert at.sidebar.radio[0].value == "Datos"


@pytest.mark.usefixtures("clean_root_logger")
def test_sidebar_shows_current_run_and_steps(tmp_config: Path) -> None:
    """Con una corrida con ETL muestra run_id, semilla, hash pendiente y pasos."""
    run_id = pipeline.run_etl(seed=7)
    at = run_app()
    assert not at.exception
    text = sidebar_text(at)
    assert run_id in text
    assert "**Semilla de esta corrida:** 7" in text
    assert "Hash del dataset:** —" in text
    assert "✅ ETL — hecho" in text
    assert "⬜ Entrenamiento — pendiente" in text
    assert "⬜ Reporte — pendiente" in text
    assert at.sidebar.selectbox[0].value == run_id


@pytest.mark.usefixtures("clean_root_logger")
def test_selector_opens_an_older_run(tmp_config: Path) -> None:
    """El selector deja abrir una corrida anterior y la barra se actualiza."""
    old = pipeline.run_etl(seed=3)
    new = pipeline.run_etl(seed=9)
    at = run_app()
    assert at.sidebar.selectbox[0].value == new
    assert "Semilla de esta corrida:** 9" in sidebar_text(at)
    at.sidebar.selectbox[0].select(old).run()
    assert not at.exception
    assert "Semilla de esta corrida:** 3" in sidebar_text(at)


def test_seed_input_keeps_the_value_typed(tmp_config: Path) -> None:
    """Cambiar la semilla en la barra se conserva entre ejecuciones."""
    at = run_app()
    at.sidebar.number_input[0].set_value(123).run()
    assert not at.exception
    assert at.sidebar.number_input[0].value == 123


def test_every_page_renders(tmp_config: Path) -> None:
    """Se puede navegar a las seis páginas sin errores."""
    at = run_app()
    for name in PAGE_NAMES:
        at.sidebar.radio[0].set_value(name).run()
        assert not at.exception, name
        assert at.header[0].value == name


def test_pending_run_is_applied_before_the_selector(tmp_config: Path) -> None:
    """Una página puede pedir otra corrida sin chocar con el selector."""
    db.init_structured_db()
    first = db.create_run(seed=1)
    second = db.create_run(seed=2)
    at = run_app()
    assert at.sidebar.selectbox[0].value == second
    at.session_state["_pending_run"] = first
    at.run()
    assert not at.exception
    assert at.sidebar.selectbox[0].value == first


def go_to(at: AppTest, page: str) -> AppTest:
    """Navega a una página de la barra lateral."""
    return at.sidebar.radio[0].set_value(page).run()


def metrics(at: AppTest) -> dict[str, str]:
    """Etiqueta y valor de todas las métricas en pantalla."""
    return {m.label: m.value for m in at.metric}


def test_data_page_before_etl_asks_to_run_it(tmp_config: Path) -> None:
    """Sin ETL la página avisa y ofrece el botón, sin gráficos ni tabla."""
    at = go_to(run_app(), "Datos")
    assert not at.exception
    assert [b.label for b in at.button] == ["Ejecutar ETL"]
    assert any("Primero ejecutá el ETL" in w.value for w in at.warning)
    assert len(at.get("vega_lite_chart")) == 0
    assert len(at.dataframe) == 0


@pytest.mark.usefixtures("clean_root_logger")
def test_etl_button_creates_a_run_and_shows_results(tmp_config: Path) -> None:
    """El botón corre el ETL con la semilla de la barra y muestra todo."""
    at = run_app()
    at.sidebar.number_input[0].set_value(11).run()
    at = go_to(at, "Datos")
    at.button[0].click().run()
    assert not at.exception

    # La corrida nueva quedó elegida, con la semilla escrita y el ETL hecho.
    run_id = at.sidebar.selectbox[0].value
    text = sidebar_text(at)
    assert run_id in text
    assert "Semilla de esta corrida:** 11" in text
    assert "✅ ETL — hecho" in text
    assert db.get_run(run_id)["seed"] == 11

    shown = metrics(at)
    assert shown["Meses guardados"] == "308"
    assert shown["Meses con todas las variables"] == "259"
    assert shown["Meses sin sectores (antes de 2005)"] == "48"
    assert shown["Potencia máxima repetida"] == "2"
    assert shown["Meses con target"] == "236"
    assert shown["Demanda alta"] == "60"
    assert shown["Demanda normal"] == "176"
    assert shown["Proporción de altas"] == "25.4%"

    page_text = "\n".join(m.value for m in at.markdown)
    assert "2002-03, 2021-08" in page_text
    assert "cuantil 75%" in page_text
    assert len(at.get("vega_lite_chart")) == 3
    assert len(at.dataframe[0].value) == 308


@pytest.mark.usefixtures("clean_root_logger")
def test_data_page_keeps_results_after_reopening_the_app(tmp_config: Path) -> None:
    """Los resultados salen de las bases: una sesión nueva los sigue viendo."""
    pipeline.run_etl()
    at = go_to(run_app(), "Datos")
    assert not at.exception
    assert metrics(at)["Meses guardados"] == "308"
    assert len(at.get("vega_lite_chart")) == 3


@pytest.mark.usefixtures("clean_root_logger")
def test_data_page_shows_the_temperature_assumption(tmp_config: Path) -> None:
    """El supuesto de la zona de la temperatura está a la vista."""
    at = go_to(run_app(), "Datos")
    assert any("Buenos Aires" in i.value for i in at.info)
    assert any("nivel mensual" in m.value for m in at.markdown)


@pytest.mark.usefixtures("clean_root_logger")
def test_etl_failure_is_shown_not_raised(
    tmp_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Si el ETL falla, la app muestra el error y no se rompe."""

    def boom(*args, **kwargs):
        raise RuntimeError("csv ilegible")

    monkeypatch.setattr(pipeline, "run_etl", boom)
    at = go_to(run_app(), "Datos")
    at.button[0].click().run()
    assert not at.exception
    assert any("csv ilegible" in e.value for e in at.error)


LR = "logistic_regression"


def select_models(at: AppTest, scenarios: list[str], model_names: list[str]) -> AppTest:
    """Elige escenarios y modelos en la página Entrenamiento."""
    at.multiselect[0].set_value(scenarios)
    at.multiselect[1].set_value(model_names)
    return at.run()


def test_training_page_without_run_asks_for_etl(tmp_config: Path) -> None:
    """Sin corrida pide ejecutar el ETL y no ofrece entrenar."""
    at = go_to(run_app(), "Entrenamiento")
    assert not at.exception
    assert any("Primero ejecutá el ETL" in w.value for w in at.warning)
    assert len(at.button) == 0


def test_training_page_with_run_without_etl_asks_for_etl(tmp_config: Path) -> None:
    """Una corrida sin ETL hecho tampoco permite entrenar."""
    db.init_structured_db()
    db.create_run()
    at = go_to(run_app(), "Entrenamiento")
    assert any("Primero ejecutá el ETL" in w.value for w in at.warning)
    assert len(at.button) == 0


@pytest.mark.usefixtures("clean_root_logger")
def test_training_page_offers_all_options_after_etl(tmp_config: Path) -> None:
    """Con el ETL hecho ofrece los 2 escenarios y los 3 modelos y la semilla."""
    pipeline.run_etl(seed=5)
    at = go_to(run_app(), "Entrenamiento")
    assert not at.exception
    assert at.multiselect[0].value == ["A", "B"]
    assert at.multiselect[1].value == list(models.MODEL_NAMES)
    assert at.button[0].label == "Entrenar"
    assert at.button[0].disabled is False
    assert any("Semilla de esta corrida: **5**" in c.value for c in at.caption)


@pytest.mark.usefixtures("clean_root_logger")
def test_train_button_is_disabled_without_selection(tmp_config: Path) -> None:
    """Sin escenarios o sin modelos el botón queda deshabilitado."""
    pipeline.run_etl()
    at = go_to(run_app(), "Entrenamiento")
    at = select_models(at, [], [LR])
    assert at.button[0].disabled is True
    at = select_models(at, ["A"], [])
    assert at.button[0].disabled is True


@pytest.mark.usefixtures("clean_root_logger")
def test_training_button_trains_and_shows_saved_results(tmp_config: Path) -> None:
    """Entrenar A con regresión logística guarda el resultado y lo muestra."""
    run_id = pipeline.run_etl(seed=7)
    at = go_to(run_app(), "Entrenamiento")
    at = select_models(at, ["A"], [LR])
    at.button[0].click().run()
    assert not at.exception
    assert any("Entrenamiento terminado" in s.value for s in at.success)

    assert "✅ Entrenamiento — hecho" in sidebar_text(at)
    assert queries.run_steps(run_id)["Entrenamiento"] is True
    table = queries.results_table(run_id)
    assert table[["scenario", "model"]].values.tolist() == [["A", LR]]

    assert any("ya está entrenada" in i.value for i in at.info)
    assert len(at.button) == 0
    shown = at.dataframe[0].value
    assert shown["Modelo"].tolist() == ["Regresión logística"]
    assert shown["F1 hold-out"].iloc[0] == pytest.approx(table["f1"].iloc[0])
    assert shown["F1 «siempre alta»"].iloc[0] == pytest.approx(0.4)


@pytest.mark.usefixtures("clean_root_logger")
def test_training_progress_and_choices_reach_the_pipeline(
    tmp_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """La elección llega a run_training y se le pasa la función de progreso."""
    run_id = pipeline.run_etl()
    seen = {}

    def fake_training(rid, scenarios, model_names, on_progress, **kwargs):
        seen.update(rid=rid, scenarios=scenarios, models=model_names)
        on_progress(1, 2, "A/random_forest")
        on_progress(2, 2, "B/random_forest")
        return rid

    monkeypatch.setattr(pipeline, "run_training", fake_training)
    at = go_to(run_app(), "Entrenamiento")
    at = select_models(at, ["A", "B"], ["random_forest"])
    at.button[0].click().run()
    assert not at.exception
    assert seen == {
        "rid": run_id,
        "scenarios": ("A", "B"),
        "models": ("random_forest",),
    }


@pytest.mark.usefixtures("clean_root_logger")
def test_training_failure_is_shown_and_logged(
    tmp_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Si entrenar falla, se muestra el error y la corrida queda en error."""
    run_id = pipeline.run_etl()

    def boom(*args, **kwargs):
        raise RuntimeError("sin memoria")

    monkeypatch.setattr(train, "train_all", boom)
    at = go_to(run_app(), "Entrenamiento")
    at = select_models(at, ["A"], [LR])
    at.button[0].click().run()
    assert not at.exception
    assert any("sin memoria" in e.value for e in at.error)
    assert db.get_run(run_id)["status"] == "error"


@pytest.mark.usefixtures("clean_root_logger")
def test_flash_after_etl_is_shown_once(tmp_config: Path) -> None:
    """El aviso 'ETL ejecutado' aparece tras el botón y no se repite."""
    at = go_to(run_app(), "Datos")
    at.button[0].click().run()
    assert any("ETL ejecutado" in s.value for s in at.success)
    at.sidebar.radio[0].set_value("Logs").run()
    assert len(at.success) == 0
