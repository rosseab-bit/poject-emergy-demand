"""Tests del reporte PDF: estructura, contenido y generación sin abrir Streamlit."""

import dataclasses
import json
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
from support import main_file_only

from src import (
    config,
    explain,
    queries,
    recommend,
    report,
    report_charts,
    report_data,
    report_layout,
    report_sections,
)


def extract_text(pdf: bytes) -> str:
    """Texto de un PDF generado con ``page_compression=0``, con espacios normalizados."""
    pieces = []
    for raw in re.findall(rb"\(((?:\\.|[^\\)])*)\) Tj", pdf):
        text = re.sub(rb"\\([0-7]{3})", lambda m: bytes([int(m.group(1), 8)]), raw)
        text = text.replace(b"\\(", b"(").replace(b"\\)", b")").replace(b"\\\\", b"\\")
        piece = text.decode("cp1252")
        if re.fullmatch(r"Corrida [0-9a-f]{8}|Página \d+", piece):
            continue  # pie de página: puede caer en medio de una oración
        pieces.append(piece)
    return re.sub(r"\s+", " ", " ".join(pieces))


def contains(text: str, phrase: str) -> bool:
    """Si ``phrase`` está en ``text``, sin importar cómo se partió en líneas.

    ReportLab corta las líneas en guiones y en signos como ``<`` y ``>``; al leer el
    texto del PDF esos cortes quedan como espacios de más. Se compara sin espacios.
    """
    squash = lambda value: re.sub(r"\s+", "", value)
    return squash(phrase) in squash(text)


def page_count(pdf: bytes) -> int:
    """Cantidad de páginas de un PDF."""
    return len(re.findall(rb"/Type /Page\b", pdf))


@pytest.fixture(scope="module")
def charts(trained_report_data) -> dict[str, bytes]:
    """Gráficos de la corrida de prueba (se dibujan una sola vez)."""
    return report_charts.render_charts(trained_report_data)


@pytest.fixture(scope="module")
def pdf(trained_report_data, charts) -> bytes:
    """PDF con el texto sin comprimir, para poder leerlo en los tests."""
    return report.build_pdf(trained_report_data, charts, page_compression=0)


@pytest.fixture(scope="module")
def text(pdf: bytes) -> str:
    """Texto del PDF de prueba."""
    return extract_text(pdf)


def sample_predictions() -> list[dict]:
    """Dos predicciones guardadas, con la estructura real."""
    entry = {
        "probabilidad": 0.598,
        "clase": 1,
        "f1_holdout": 0.44,
        "f1_siempre_alta": 0.4,
        "supera_referencia": False,
    }
    other = {**entry, "probabilidad": 0.31, "clase": 0}
    return [
        {
            "id": 1,
            "created_at": "2026-11-12T18:00:00+00:00",
            "fecha": "2026-09-01",
            "temperatura": 12.0,
            "modelos": {"logistic_regression": entry, "random_forest": other},
        },
        {
            "id": 2,
            "created_at": "2026-11-12T18:05:00+00:00",
            "fecha": "2026-09-01",
            "temperatura": 21.5,
            "modelos": {"logistic_regression": other},
        },
    ]


def test_pdf_is_a_valid_multipage_a4_document(pdf: bytes) -> None:
    """Empieza y termina como un PDF, tiene varias páginas y pesa poco."""
    assert pdf.startswith(b"%PDF-")
    assert pdf.rstrip().endswith(b"%%EOF")
    assert 5 <= page_count(pdf) <= 12
    assert b"/MediaBox [ 0 0 595.2756 841.8898 ]" in pdf
    assert len(pdf) < 1_500_000


def test_metadata_names_the_report_and_the_run(pdf: bytes, trained_report_data) -> None:
    """El título del documento lleva el nombre del proyecto y la corrida."""
    assert b"/Title" in pdf
    assert (
        trained_report_data.run_id.encode()
        in pdf[pdf.index(b"/Title") : pdf.index(b"/Title") + 200]
    )


def test_same_data_gives_the_same_bytes(trained_report_data, charts) -> None:
    """El PDF es reproducible: mismos datos, mismos bytes."""
    first = report.build_pdf(trained_report_data, charts)
    second = report.build_pdf(trained_report_data, charts)
    assert first == second


def test_sections_appear_in_the_planned_order(text: str) -> None:
    """Las seis secciones están, en orden."""
    titles = [
        "1. Datos y ETL",
        "2. Resultados por modelo y escenario",
        "3. Comparación",
        "4. Modelo recomendado y por qué",
        "5. Predicciones realizadas",
        "6. Límites",
    ]
    # La primera aparición de cada título es la del índice de la portada; usamos la última.
    positions = [text.rindex(title) for title in titles]
    assert positions == sorted(positions)
    assert text.index("Contenido") < text.index("Clasificación de demanda") + 2000


def test_cover_has_the_reproducibility_data(text: str, trained_report_data) -> None:
    """La portada trae corrida, fecha, semilla, hash completo, versiones y validación."""
    data = trained_report_data
    assert f"Reporte de la corrida {data.run_id}" in text
    assert data.generated_at in text
    assert f"Semilla {data.config['seed']}" in text
    assert data.config["dataset_sha256"] in text
    assert f"scikit-learn {data.config['versions']['scikit-learn']}" in text
    assert "hold-out de los últimos 48 meses" in text
    assert "TimeSeriesSplit de 3 cortes" in text
    commit = data.config["git_commit"]
    assert commit is None or commit in text


def test_etl_section_reports_the_real_numbers(text: str) -> None:
    """Filas, correcciones, target y supuestos salen de la base."""
    assert "308 meses; 259 tienen todas las variables" in text
    assert "2: 2002-03, 2021-08" in text
    assert "cuantil 75 %" in text and "36 meses anteriores" in text
    assert "Proporción de altas" in text and "25,4 %" in text
    assert "Buenos Aires / AMBA" in text
    assert "nivel mensual" in text
    assert "base de 18 °C" in text


def test_every_conclusion_matches_the_rule_and_the_text(
    text: str, trained_report_data
) -> None:
    """El titular de cada escenario es el que redactan las reglas, con su etiqueta."""
    for scenario, conclusion in trained_report_data.conclusions.items():
        label, _status = report_layout.CATEGORY_STATUS[conclusion.verdict.category]
        assert label in text
        assert contains(text, report_layout.pdf_text(conclusion.explanation.headline))
        for reason in conclusion.explanation.reasons:
            assert contains(text, report_layout.pdf_text(reason)), (scenario, reason)


def test_comparison_table_says_whether_each_model_beats_the_reference(
    text: str, trained_report_data
) -> None:
    """La tabla de comparación trae F1, desvío y si supera a «siempre alta»."""
    assert "F1 validación (prom. ± desv.)" in text
    assert "¿Supera la referencia?" in text
    for conclusion in trained_report_data.conclusions.values():
        for score in conclusion.scores:
            assert f"{explain.fmt(score.cv_mean)} ± {explain.fmt(score.cv_std)}" in text
            assert isinstance(recommend.beats_reference(score), bool)
    assert "Escenario A contra B" in text and "Diferencia (B - A)" in text


def test_results_section_has_params_and_holdout_metrics(
    text: str, trained_report_data
) -> None:
    """Hiperparámetros sin prefijo, F1 por corte y métricas del hold-out por modelo."""
    assert "Validación cruzada (solo con datos de entrenamiento)" in text
    assert "Hold-out (la clase positiva es «alta»)" in text
    for doc in trained_report_data.docs:
        assert report_layout.format_params(doc["params"]) in text
        assert ", ".join(explain.fmt(s, 2) for s in doc["cv"]["scores"]) in text
        assert explain.fmt(doc["holdout"]["accuracy"]) in text
    assert "clf__" not in text


def test_limits_state_the_real_caveats(text: str, trained_report_data) -> None:
    """Los límites mencionan el hold-out chico, la calibración y la zona de temperatura."""
    section = text[text.rindex("6. Límites") :]
    assert "48 meses y solo 12 son de demanda alta" in section
    assert "no están calibradas" in section
    assert "Buenos Aires / AMBA" in section
    assert "única semilla (7)" in section
    assert "escenario B usa la potencia máxima" in section
    main = trained_report_data.conclusions["A"]
    sentence = "ningún modelo supera con claridad a la regla «siempre alta»"
    assert (sentence in section) == (not main.verdict.recommend_for_use)


def test_limits_add_the_reference_warning_only_when_it_applies(
    trained_report_data,
) -> None:
    """La advertencia sobre «siempre alta» aparece solo si el A no la supera."""
    warning = "En el escenario A ningún modelo supera con claridad"
    base = trained_report_data
    assert any(warning in item for item in report_sections.limits(base)) != (
        base.conclusions["A"].verdict.recommend_for_use
    )
    forced = dataclasses.replace(
        base,
        conclusions={
            **base.conclusions,
            "A": dataclasses.replace(
                base.conclusions["A"],
                verdict=dataclasses.replace(
                    base.conclusions["A"].verdict, recommend_for_use=True
                ),
            ),
        },
    )
    assert not any(warning in item for item in report_sections.limits(forced))


def test_predictions_section_without_predictions(trained_report_data, charts) -> None:
    """Pedidas pero sin ninguna guardada: lo dice."""
    data = dataclasses.replace(
        trained_report_data, predictions=[], predictions_included=True
    )
    out = extract_text(report.build_pdf(data, charts, page_compression=0))
    assert "No hay predicciones guardadas para esta corrida." in out
    assert "sin incluir las predicciones" not in out


def test_predictions_section_lists_them_with_probability_and_class(
    trained_report_data, charts
) -> None:
    """Con predicciones: una fila cada una, con probabilidad y clase por modelo."""
    data = dataclasses.replace(
        trained_report_data, predictions=sample_predictions(), predictions_included=True
    )
    section_text = extract_text(report.build_pdf(data, charts, page_compression=0))
    section = section_text[section_text.rindex("5. Predicciones realizadas") :]
    assert "2026-09" in section
    assert "12,0" in section and "21,5" in section
    assert "0,60 (alta)" in section and "0,31 (normal)" in section
    assert "Las probabilidades no están calibradas" in section
    assert "No hay predicciones guardadas" not in section


def test_predictions_can_be_left_out(trained_report_data, charts) -> None:
    """Si no se piden, el reporte lo aclara en vez de decir que no hay."""
    data = dataclasses.replace(
        trained_report_data, predictions=[], predictions_included=False
    )
    out = extract_text(report.build_pdf(data, charts, page_compression=0))
    assert "se generó sin incluir las predicciones" in out
    assert "No hay predicciones guardadas" not in out


def test_dynamic_text_is_escaped_not_interpreted(trained_report_data, charts) -> None:
    """Un aviso con <, > o & se muestra tal cual y no rompe el documento."""
    data = dataclasses.replace(
        trained_report_data, notes=["Aviso <b>raro</b> & 'comillas'"]
    )
    out = extract_text(report.build_pdf(data, charts, page_compression=0))
    assert contains(out, "Aviso <b>raro</b> & 'comillas'")


def test_unsupported_symbols_are_replaced_so_helvetica_can_print_them() -> None:
    """El signo menos tipográfico y otros símbolos se reemplazan por ASCII."""
    assert report_layout.pdf_text("−0,5") == "-0,5"
    assert report_layout.pdf_text("a → b") == "a -> b"
    assert report_layout.pdf_text("«alta» ± 0,1 °C") == "«alta» ± 0,1 °C"
    assert report_layout.esc("a < b & c") == "a &lt; b &amp; c"


def test_all_report_text_is_encodable_in_the_pdf_font(trained_report_data) -> None:
    """Todo texto redactado por las reglas se puede imprimir con Helvetica (cp1252)."""
    texts = list(report_sections.limits(trained_report_data))
    for conclusion in trained_report_data.conclusions.values():
        texts += [conclusion.explanation.headline, *conclusion.explanation.reasons]
    assert len(texts) > 10  # hay texto que revisar
    unencodable = []
    for text in texts:
        try:
            report_layout.pdf_text(text).encode("cp1252")
        except UnicodeEncodeError:
            unencodable.append(text)
    assert unencodable == []


def test_report_with_a_single_scenario_still_builds(trained_report_data) -> None:
    """Con solo el escenario A no hay comparación A contra B y todo lo demás sale."""
    base = trained_report_data
    only_a = dataclasses.replace(
        base,
        table=base.table[base.table["scenario"] == "A"].reset_index(drop=True),
        docs=[d for d in base.docs if d["scenario"] == "A"],
        conclusions={"A": base.conclusions["A"]},
    )
    pdf = report.build_pdf(
        only_a, report_charts.render_charts(only_a), page_compression=0
    )
    out = extract_text(pdf)
    assert pdf.startswith(b"%PDF-") and page_count(pdf) >= 4
    assert "Escenario A contra B" not in out
    assert "Escenario B · diagnóstico" not in out
    assert "6. Límites" in out


def test_every_category_has_a_written_label_and_a_known_color() -> None:
    """El estado nunca depende solo del color: cada categoría tiene etiqueta escrita."""
    assert set(report_layout.CATEGORY_STATUS) == set(recommend.CATEGORIES)
    for label, status in report_layout.CATEGORY_STATUS.values():
        assert label and status in report_layout.STATUS


def test_report_path_and_exists(tmp_path: Path) -> None:
    """El PDF se guarda como reporte_<run_id>.pdf en la carpeta de reportes."""
    assert report.report_path("abc", tmp_path) == tmp_path / "reporte_abc.pdf"
    assert report.report_path("abc") == config.REPORTS_DIR / "reporte_abc.pdf"
    assert report.report_exists("abc", tmp_path) is False


def test_save_report_writes_and_overwrites(tmp_path: Path) -> None:
    """Guardar crea la carpeta y pisa el archivo anterior."""
    calls = []

    def fake(run_id, include_predictions, **paths):
        calls.append((run_id, include_predictions, paths))
        return b"%PDF-fake-" + str(len(calls)).encode()

    target = tmp_path / "salida" / "reportes"
    paths = {"db_path": tmp_path / "structured.db"}
    path = report.save_report("r1", False, target, fake, **paths)
    assert path == target / "reporte_r1.pdf" and path.read_bytes() == b"%PDF-fake-1"
    report.save_report("r1", True, target, fake, **paths)
    assert path.read_bytes() == b"%PDF-fake-2"
    assert calls[0] == ("r1", False, paths)
    assert calls[1][1] is True
    assert report.report_exists("r1", target) is True


def test_save_report_logs_the_event_under_the_run(tmp_path: Path) -> None:
    """Generar el reporte deja inicio, el evento con el archivo y fin, atados al run_id."""
    db_path = tmp_path / "structured.db"

    def fake(run_id, include_predictions, **paths):
        return b"%PDF-" + b"x" * 3000

    report.save_report("r9", False, tmp_path, fake, db_path=db_path)
    logs = queries.get_logs("r9", db_path=db_path)
    messages = logs["message"].tolist()
    assert messages[0] == "Inicio: _generate_and_write"
    assert "Reporte generado: reporte_r9.pdf (2 KB, predicciones omitidas)" in messages
    assert messages[-1].startswith("Fin: _generate_and_write")
    assert queries.get_logs(db_path=db_path)["run_id"].isna().sum() == 0


def test_save_report_leaves_its_log_event_in_the_main_database_file(
    tmp_path: Path,
) -> None:
    """El evento del reporte ya está en el .db principal (sin depender del -wal)."""
    db_path = tmp_path / "structured.db"

    def fake(run_id, include_predictions, **paths):
        return b"%PDF-" + b"x" * 100

    report.save_report("r5", True, tmp_path, fake, db_path=db_path)
    copy = main_file_only(db_path, tmp_path / "copia")
    conn = sqlite3.connect(copy)
    try:
        messages = [r[0] for r in conn.execute("SELECT message FROM logs")]
    finally:
        conn.close()
    assert any(m.startswith("Reporte generado: reporte_r5.pdf") for m in messages)


def test_failed_report_is_logged_as_error_and_leaves_no_file(tmp_path: Path) -> None:
    """Si falla la generación, queda el error en logs y no se escribe ningún archivo."""
    db_path = tmp_path / "structured.db"

    def boom(run_id, include_predictions, **paths):
        raise RuntimeError("no se pudo armar")

    with pytest.raises(RuntimeError, match="no se pudo armar"):
        report.save_report("r9", True, tmp_path, boom, db_path=db_path)
    errors = queries.get_logs("r9", levels=["ERROR"], db_path=db_path)
    assert errors["message"].str.contains("no se pudo armar").any()
    assert not report.report_exists("r9", tmp_path)


def test_report_naming_is_defined_once_in_queries(tmp_path: Path) -> None:
    """report y queries comparten el nombre del archivo; has_report mira el disco."""
    assert queries.report_path("abc", tmp_path) == report.report_path("abc", tmp_path)
    assert queries.has_report("abc", tmp_path) is False
    (tmp_path / "reporte_abc.pdf").write_bytes(b"%PDF-")
    assert queries.has_report("abc", tmp_path) is True
    assert report.report_exists("abc", tmp_path) is True


def test_report_step_follows_the_pdf_on_disk(tmp_path: Path) -> None:
    """El paso Reporte de la corrida está hecho solo si existe su PDF."""
    paths = {"db_path": tmp_path / "s.db", "unstructured_path": tmp_path / "u.db"}
    assert queries.run_steps("r1", reports_dir=tmp_path, **paths)["Reporte"] is False
    (tmp_path / "reporte_r1.pdf").write_bytes(b"%PDF-")
    assert queries.run_steps("r1", reports_dir=tmp_path, **paths)["Reporte"] is True
    assert queries.run_steps("otra", reports_dir=tmp_path, **paths)["Reporte"] is False


def test_generate_report_reads_only_the_databases(
    trained_run: dict, tmp_path: Path
) -> None:
    """generate_report y save_report arman el PDF desde las bases, sin Streamlit."""
    paths = {k: trained_run[k] for k in ("db_path", "unstructured_path", "models_dir")}
    saved = report.save_report(trained_run["run_id"], True, tmp_path, **paths)
    pdf = saved.read_bytes()
    assert pdf.startswith(b"%PDF-") and page_count(pdf) >= 5
    with pytest.raises(ValueError, match="No existe la corrida"):
        report.generate_report("noexiste", **paths)


def test_a_fresh_process_generates_the_same_report(
    trained_run: dict, tmp_path: Path
) -> None:
    """Un proceso nuevo, sin memoria, genera un reporte con la misma conclusión."""
    code = (
        "import sys, json; from pathlib import Path; from src import report;"
        "a = sys.argv; pdf = report.generate_report(a[1], True, Path(a[2]), Path(a[3]),"
        " Path(a[4]), page_compression=0); Path(a[5]).write_bytes(pdf);"
        "print(json.dumps({'n': len(pdf), 'streamlit': 'streamlit' in sys.modules}))"
    )
    out = tmp_path / "hijo.pdf"
    done = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            trained_run["run_id"],
            str(trained_run["db_path"]),
            str(trained_run["unstructured_path"]),
            str(trained_run["models_dir"]),
            str(out),
        ],
        cwd=config.ROOT_DIR,
        capture_output=True,
        text=True,
        check=True,
        timeout=180,
    )
    info = json.loads(done.stdout.strip().splitlines()[-1])
    assert info["streamlit"] is False  # no hace falta abrir la interfaz
    here = report_data.collect_report_data(
        trained_run["run_id"],
        True,
        trained_run["db_path"],
        trained_run["unstructured_path"],
        trained_run["models_dir"],
    )
    child_text = extract_text(out.read_bytes())
    assert (
        report_layout.pdf_text(here.conclusions["A"].explanation.headline) in child_text
    )
    assert trained_run["run_id"] in child_text
