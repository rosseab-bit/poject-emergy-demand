"""Tests de la página Reporte (AppTest sobre una corrida real entrenada)."""

import shutil
import subprocess
from functools import partial
from pathlib import Path

import pytest
from support import open_page as open_page_in
from support import sidebar_text

from app_pages import reporte
from src import config, pipeline, queries, report

open_page = partial(open_page_in, "Reporte", timeout=300)


@pytest.fixture
def reports_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Carpeta temporal de reportes (los tests no escriben en reports/ del proyecto)."""
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
    return tmp_path / "reports"


def stub_save_report(calls: list, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reemplaza la generación por una que registra la llamada y escribe un PDF falso."""

    def fake(run_id, include_predictions, *args, **kwargs):
        calls.append((run_id, include_predictions))
        path = queries.report_path(run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"%PDF-falso")
        return path

    monkeypatch.setattr(report, "save_report", fake)


def test_report_page_without_runs_asks_to_train(
    tmp_config: Path, reports_dir: Path
) -> None:
    """Sin corridas pide entrenar y no ofrece generar nada."""
    at = open_page()
    assert not at.exception
    assert at.header[0].value == "Reporte"
    assert any("Primero entrená" in w.value for w in at.warning)
    assert len(at.button) == 0 and len(at.checkbox) == 0


def test_report_page_with_an_untrained_run_asks_to_train(
    tmp_config: Path, reports_dir: Path
) -> None:
    """Una corrida con solo el ETL tampoco permite generar el reporte."""
    pipeline.run_etl()
    at = open_page()
    assert any("Primero entrená" in w.value for w in at.warning)
    assert len(at.button) == 0


def test_report_page_lists_the_contents_and_the_option(
    use_trained: dict, reports_dir: Path
) -> None:
    """Muestra el índice de lo que contiene, la opción de predicciones y el botón."""
    at = open_page()
    assert not at.exception
    page = "\n".join(m.value for m in at.main.markdown)
    for n, item in enumerate(reporte.CONTENTS, start=1):
        assert f"{n}. {item}" in page
    assert at.checkbox[0].label == "Incluir las predicciones guardadas"
    assert at.checkbox[0].value is True
    assert [b.label for b in at.button] == ["Generar PDF"]
    expected = len(queries.prediction_docs(use_trained["run_id"]))
    assert any(
        f"Predicciones guardadas para esta corrida: {expected}." in c.value
        for c in at.caption
    )
    assert len(at.get("download_button")) == 0
    assert "⬜ Reporte — pendiente" in sidebar_text(at)


def test_generating_creates_the_pdf_and_offers_the_download(
    use_trained: dict, reports_dir: Path
) -> None:
    """El botón genera el PDF real, lo guarda, avisa y ofrece descargarlo."""
    at = open_page().button[0].click().run(timeout=300)
    assert not at.exception
    path = queries.report_path(use_trained["run_id"])
    assert path.parent == reports_dir and path.is_file()
    assert path.read_bytes().startswith(b"%PDF-")
    assert any("Reporte generado" in s.value for s in at.success)
    assert "✅ Reporte — hecho" in sidebar_text(at)
    assert [s.value for s in at.main.subheader][-1] == "Reporte generado"
    assert len(at.get("download_button")) == 1
    assert any(path.name in m.value for m in at.main.markdown)
    assert any("generalo de nuevo" in c.value for c in at.caption)


def test_generating_leaves_an_event_in_logs_under_the_run(
    use_trained: dict, reports_dir: Path
) -> None:
    """Generar el reporte queda registrado en logs, atado al run_id."""
    open_page().button[0].click().run(timeout=300)
    logs = queries.get_logs(use_trained["run_id"], db_path=use_trained["db_path"])
    assert logs["message"].str.startswith("Reporte generado: reporte_").any()
    assert logs["run_id"].notna().all()


def test_the_checkbox_choice_reaches_the_generator(
    use_trained: dict, reports_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Con la casilla tildada o no, la generación recibe esa elección."""
    calls: list = []
    stub_save_report(calls, monkeypatch)
    open_page().button[0].click().run()
    assert calls == [(use_trained["run_id"], True)]
    at = open_page()
    at.checkbox[0].uncheck().run()
    at.button[0].click().run()
    assert calls[-1] == (use_trained["run_id"], False)


def test_a_new_session_still_sees_the_generated_report(
    use_trained: dict, reports_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """El estado sale del disco: al reabrir la app el PDF sigue disponible."""
    stub_save_report([], monkeypatch)
    open_page().button[0].click().run()
    at = open_page()  # sesión nueva, sin nada en la memoria de Streamlit
    assert not at.exception
    assert len(at.get("download_button")) == 1
    assert "✅ Reporte — hecho" in sidebar_text(at)
    assert any("generado el" in m.value for m in at.main.markdown)


def test_a_failed_generation_is_shown_and_leaves_the_step_pending(
    use_trained: dict, reports_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Si falla la generación se muestra el error y no queda ningún PDF."""

    def boom(*args, **kwargs):
        raise RuntimeError("sin espacio en disco")

    monkeypatch.setattr(report, "save_report", boom)
    at = open_page().button[0].click().run()
    assert not at.exception
    assert any("sin espacio en disco" in e.value for e in at.error)
    assert len(at.get("download_button")) == 0
    assert "⬜ Reporte — pendiente" in sidebar_text(at)
    assert not queries.has_report(use_trained["run_id"], reports_dir)


def test_viewing_the_page_adds_no_events_to_logs(
    use_trained: dict, reports_dir: Path
) -> None:
    """Mirar la página no escribe en logs; solo generar el reporte lo hace."""
    before = len(queries.get_logs(db_path=use_trained["db_path"]))
    open_page()
    assert len(queries.get_logs(db_path=use_trained["db_path"])) == before


@pytest.mark.skipif(
    shutil.which("pdftotext") is None, reason="pdftotext no está instalado"
)
def test_unchecking_the_option_really_leaves_the_predictions_out(
    use_trained: dict, reports_dir: Path
) -> None:
    """De punta a punta: sin la casilla, el PDF real dice que no incluye predicciones."""
    at = open_page()
    at.checkbox[0].uncheck().run()
    at.button[0].click().run(timeout=300)
    path = queries.report_path(use_trained["run_id"])
    out = subprocess.run(
        ["pdftotext", str(path), "-"], capture_output=True, text=True, check=True
    ).stdout
    assert " ".join(out.split()).count("sin incluir las predicciones") == 1
