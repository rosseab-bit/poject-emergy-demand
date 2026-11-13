"""Tests de la página Logs."""

from functools import partial
from pathlib import Path

import pandas as pd
import pytest
from support import open_page as open_page_in
from support import start_app

from app_pages import tables
from src import config, pipeline, queries

open_page = partial(open_page_in, "Logs", timeout=60)


@pytest.fixture
def two_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_root_logger
) -> dict:
    """Una corrida fallida (CSV inválido) y otra exitosa, la más reciente."""
    monkeypatch.setattr(config, "STRUCTURED_DB_PATH", tmp_path / "structured.db")
    monkeypatch.setattr(config, "UNSTRUCTURED_DB_PATH", tmp_path / "unstructured.db")
    bad = tmp_path / "malo.csv"
    bad.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="Faltan columnas"):
        pipeline.run_etl(csv_path=bad, seed=1)
    failed = queries.list_runs()["run_id"].iloc[0]
    good = pipeline.run_etl(seed=2)
    return {"failed": failed, "good": good}


def test_logs_page_without_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sin corridas ni eventos avisa, sin errores."""
    monkeypatch.setattr(config, "STRUCTURED_DB_PATH", tmp_path / "s.db")
    monkeypatch.setattr(config, "UNSTRUCTURED_DB_PATH", tmp_path / "u.db")
    at = open_page()
    assert not at.exception
    assert any("No hay eventos" in i.value for i in at.info)
    assert any("Todavía no hay corridas" in i.value for i in at.info)


def test_logs_default_shows_only_the_current_run(two_runs: dict) -> None:
    """Por defecto muestra los eventos de la corrida elegida, sin errores."""
    at = open_page()
    assert not at.exception
    assert at.radio[0].value == "Corrida actual"
    events = at.dataframe[0].value
    assert set(events["Corrida"]) == {two_runs["good"]}
    expected = queries.get_logs(two_runs["good"])
    assert len(events) == len(expected)
    assert {m.label: m.value for m in at.metric} == {
        "Eventos": str(len(expected)),
        "Advertencias": "2",
        "Errores": "0",
    }


def test_logs_all_runs_shows_the_error_of_the_failed_run(two_runs: dict) -> None:
    """Con 'Todas las corridas' aparece el error de la corrida fallida."""
    at = open_page()
    at.radio[0].set_value("Todas las corridas").run()
    assert not at.exception
    events = at.dataframe[0].value
    assert set(events["Corrida"]) == {two_runs["failed"], two_runs["good"]}
    errors = events[events["Nivel"] == "❌ ERROR"]
    # @log_step registra la falla en load_raw y de nuevo en el paso que lo contiene.
    assert len(errors) == 2
    assert set(errors["Corrida"]) == {two_runs["failed"]}
    assert all("Faltan columnas" in m for m in errors["Mensaje"])
    assert at.metric[2].value == "2"


def test_logs_filter_by_level_and_module(two_runs: dict) -> None:
    """Los filtros de nivel y módulo acotan la tabla."""
    at = open_page()
    at.radio[0].set_value("Todas las corridas").run()
    at.multiselect[0].set_value(["ERROR"]).run()
    assert set(at.dataframe[0].value["Nivel"]) == {"❌ ERROR"}

    at.multiselect[0].set_value([]).run()
    at.multiselect[1].set_value(["src.data"]).run()
    assert not at.exception
    assert set(at.dataframe[0].value["Módulo"]) == {"src.data"}


def test_logs_filter_without_matches_says_so(two_runs: dict) -> None:
    """Si ningún evento cumple los filtros, lo avisa."""
    at = open_page()
    at.multiselect[0].set_value(["ERROR"]).run()  # la corrida actual no tiene errores
    assert not at.exception
    assert any("No hay eventos con esos filtros" in i.value for i in at.info)


def test_runs_table_lists_both_runs_with_status_icons(two_runs: dict) -> None:
    """La lista de corridas muestra las dos con su estado e ícono."""
    at = open_page()
    runs = at.dataframe[-1].value
    by_id = runs.set_index("Corrida")["Estado"].to_dict()
    assert by_id == {two_runs["good"]: "✅ ok", two_runs["failed"]: "❌ error"}
    assert runs["Corrida"].iloc[0] == two_runs["good"]  # la más reciente primero


def test_level_and_status_always_carry_an_icon_not_only_color() -> None:
    """Errores y advertencias llevan ícono y texto, no solo color."""
    logs = pd.DataFrame(
        {
            "ts": ["t"] * 3,
            "level": ["INFO", "WARNING", "ERROR"],
            "module": ["m"] * 3,
            "message": ["a", "b", "c"],
            "run_id": ["r"] * 3,
        }
    )
    assert tables.logs_for_display(logs)["Nivel"].tolist() == [
        "INFO",
        "⚠️ WARNING",
        "❌ ERROR",
    ]


def test_highlight_row_colors_errors_and_warnings_only() -> None:
    """Rojo para errores, ámbar para advertencias y nada para el resto."""

    def style(level: str) -> list[str]:
        row = pd.Series({"Nivel": tables.with_icon(level, tables.LEVEL_ICONS), "x": 1})
        return tables.highlight_row(row)

    assert style("ERROR") == [tables.ROW_STYLES["ERROR"]] * 2
    assert style("WARNING") == [tables.ROW_STYLES["WARNING"]] * 2
    assert style("INFO") == ["", ""]
    failed_run = pd.Series({"Estado": "❌ error", "x": 1})
    assert tables.highlight_row(failed_run) == [tables.ROW_STYLES["ERROR"]] * 2
    ok_run = pd.Series({"Estado": "✅ ok", "x": 1})
    assert tables.highlight_row(ok_run) == ["", ""]


def test_viewing_pages_does_not_add_events_to_logs(two_runs: dict) -> None:
    """Abrir Datos y las demás páginas no escribe nada en la tabla logs."""
    before = len(queries.get_logs())
    at = start_app(60)  # página Datos
    for page in ("Entrenamiento", "Comparación", "Datos", "Logs"):
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, page
    assert len(queries.get_logs()) == before
    assert queries.get_logs()["run_id"].isna().sum() == 0
