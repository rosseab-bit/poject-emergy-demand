"""Tests de la interfaz de Streamlit con AppTest (sin abrir un navegador)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app_pages.sidebar import PAGE_NAMES
from src import config, db, pipeline

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
