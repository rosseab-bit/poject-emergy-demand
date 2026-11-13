"""Tests de la recolección de datos del reporte (solo desde las bases)."""

import dataclasses
import json
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from support import isolated_logging, isolated_mongo, tiny_grids

from src import (
    config,
    db,
    logger,
    pipeline,
    predict,
    queries,
    recommend,
    report,
    report_charts,
    report_data,
    results,
)

LR = ("logistic_regression",)


@pytest.fixture(scope="module")
def report_run(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Corrida propia: regresión logística en A y B, y dos predicciones."""
    base = tmp_path_factory.mktemp("report")
    paths = {
        "db_path": base / "structured.db",
        "unstructured_path": base / "unstructured.db",
        "models_dir": base / "models",
    }
    with isolated_logging(), isolated_mongo() as client, tiny_grids():
        run_id = pipeline.run_etl(db_path=paths["db_path"], seed=7)
        pipeline.run_training(
            run_id,
            db_path=paths["db_path"],
            unstructured_path=paths["unstructured_path"],
            models_dir=paths["models_dir"],
            model_names=LR,
        )
        predict.predict_next_month(run_id, 12.0, **paths)
        predict.predict_next_month(run_id, 20.0, **paths)
        etl_only = pipeline.run_etl(db_path=paths["db_path"])
    return {"run_id": run_id, "etl_only": etl_only, "mongo": client, **paths}


def collect(run: dict, **kwargs) -> report_data.ReportData:
    """Reúne los datos del reporte de la corrida de prueba."""
    paths = {k: run[k] for k in ("db_path", "unstructured_path", "models_dir")}
    return report_data.collect_report_data(run["run_id"], **paths, **kwargs)


def test_run_and_configuration_come_from_the_databases(report_run: dict) -> None:
    """Corrida, semilla, hash del dataset y parámetros salen de lo guardado."""
    data = collect(report_run)
    assert data.run_id == report_run["run_id"]
    assert data.run["seed"] == 7 and data.run["status"] == "ok"
    assert data.config["seed"] == 7
    assert data.config["dataset_sha256"] == results.file_sha256(config.RAW_CSV_PATH)
    assert data.config["holdout_months"] == config.HOLDOUT_MONTHS
    assert data.config["target"]["window_months"] == config.TARGET_WINDOW_MONTHS
    assert "scikit-learn" in data.config["versions"]
    assert data.generated_at.endswith("+00:00")


def test_etl_overview_and_results_tables(report_run: dict) -> None:
    """Trae el resumen del ETL y las métricas de los dos escenarios."""
    data = collect(report_run)
    assert (data.overview["rows"], data.overview["rows_complete"]) == (308, 259)
    assert data.overview["target"]["total"] == 236
    assert data.overview["target"]["positivos"] == 60
    assert data.table[["scenario", "model"]].values.tolist() == [
        ["A", "logistic_regression"],
        ["B", "logistic_regression"],
    ]
    assert len(data.docs) == 2
    assert (data.noise["n_test"], data.noise["positives"]) == (48, 12)


def test_each_scenario_has_a_verdict_a_text_and_the_variables(report_run: dict) -> None:
    """Hay una conclusión por escenario, coherente con la regla y con el texto."""
    data = collect(report_run)
    assert list(data.conclusions) == ["A", "B"]
    for scenario, conclusion in data.conclusions.items():
        expected = recommend.evaluate_scenario(
            recommend.scores_from_table(data.table, scenario), scenario
        )
        assert conclusion.verdict == expected
        assert conclusion.verdict.recommended == "logistic_regression"
        assert conclusion.explanation.headline  # redactado por reglas
        assert "48 meses y solo 12" in " ".join(conclusion.explanation.reasons)
    assert len(data.conclusions["A"].importance) == 14
    assert len(data.conclusions["B"].importance) == 15
    assert set(data.conclusions["A"].importance["tipo"]) == {"coeficiente"}


def test_predictions_are_included_in_order_or_left_out(report_run: dict) -> None:
    """Las predicciones se incluyen, de la más vieja a la más nueva, o se omiten."""
    with_predictions = collect(report_run)
    assert [p["temperatura"] for p in with_predictions.predictions] == [12.0, 20.0]
    ids = [p["id"] for p in with_predictions.predictions]
    assert ids == sorted(ids)
    first = with_predictions.predictions[0]
    assert first["fecha"] == "2026-09-01" and "created_at" in first
    assert "logistic_regression" in first["modelos"]
    assert collect(report_run, include_predictions=False).predictions == []


def test_traceability_notes() -> None:
    """Avisa si la corrida tenía cambios sin commitear o no registró el commit."""
    assert (
        report_data.traceability_notes({"git_commit": "abc", "git_dirty": False}) == []
    )
    dirty = report_data.traceability_notes({"git_commit": "abc", "git_dirty": True})
    assert len(dirty) == 1 and "sin commitear" in dirty[0]
    missing = report_data.traceability_notes({"git_commit": None, "git_dirty": None})
    assert len(missing) == 1 and "commit" in missing[0]


def test_dirty_run_note_reaches_the_report_data(report_run: dict) -> None:
    """Una corrida hecha con cambios sin commitear lo deja anotado en el reporte."""
    data = collect(report_run)
    config_doc = results.load_run_config(
        report_run["run_id"], report_run["unstructured_path"]
    )
    assert (config_doc["git_dirty"] is True) == any(
        "sin commitear" in n for n in data.notes
    )


def test_unknown_untrained_or_missing_etl_runs_are_rejected(
    report_run: dict, tmp_path: Path
) -> None:
    """Sin corrida, sin entrenar o sin ETL no se puede armar el reporte."""
    with pytest.raises(ValueError, match="No existe la corrida"):
        collect(report_run | {"run_id": "noexiste"})
    with pytest.raises(ValueError, match="primero entrenala"):
        collect(report_run | {"run_id": report_run["etl_only"]})
    empty = {
        "db_path": tmp_path / "s.db",
        "unstructured_path": tmp_path / "u.db",
        "models_dir": tmp_path / "m",
    }
    db.init_structured_db(empty["db_path"])
    run_id = db.create_run(db_path=empty["db_path"])
    with pytest.raises(ValueError, match="primero entrenala"):
        collect(empty | {"run_id": run_id})


@pytest.mark.parametrize("damage", ["missing", "tampered"])
def test_unreadable_model_is_reported_as_a_note_not_a_crash(
    report_run: dict, tmp_path: Path, damage: str
) -> None:
    """Si falta o se alteró un .joblib, el reporte sigue y deja un aviso."""
    copy = tmp_path / "models"
    shutil.copytree(report_run["models_dir"], copy)
    victim = next(copy.glob("*_A_logistic_regression.joblib"))
    if damage == "missing":
        victim.unlink()
    else:
        victim.write_bytes(victim.read_bytes() + b"x")
    data = collect(report_run | {"models_dir": copy})
    assert data.conclusions["A"].importance is None
    assert data.conclusions["B"].importance is not None
    assert any("escenario A" in note and "variables" in note for note in data.notes)
    assert "pesan más" not in " ".join(data.conclusions["A"].explanation.reasons)


def test_collecting_adds_no_events_to_logs(report_run: dict) -> None:
    """Armar los datos del reporte no escribe en la colección logs."""
    root = logging.getLogger(logger.ROOT_LOGGER_NAME)
    saved_level, saved_handlers = root.level, list(root.handlers)
    try:
        logger.setup_logging()
        before = len(queries.get_logs())
        collect(report_run)
        logger.flush_logs()  # si hubiera algo pendiente, saldría acá
        assert len(queries.get_logs()) == before
    finally:
        root.setLevel(saved_level)
        root.handlers[:] = saved_handlers


def test_data_comes_only_from_the_databases_even_in_a_fresh_process(
    report_run: dict,
) -> None:
    """Un proceso nuevo, sin memoria, arma los mismos datos solo desde el disco."""
    code = (
        "import sys, json; from pathlib import Path; from src import report_data as r;"
        "a = sys.argv; d = r.collect_report_data(a[1], db_path=Path(a[2]),"
        " unstructured_path=Path(a[3]), models_dir=Path(a[4]));"
        "print(json.dumps({'seed': d.config['seed'], 'sha': d.config['dataset_sha256'],"
        " 'rows': d.overview['rows'], 'n_pred': len(d.predictions),"
        " 'cat': {k: c.verdict.category for k, c in d.conclusions.items()},"
        " 'head': d.conclusions['A'].explanation.headline}))"
    )
    done = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            report_run["run_id"],
            str(report_run["db_path"]),
            str(report_run["unstructured_path"]),
            str(report_run["models_dir"]),
        ],
        cwd=config.ROOT_DIR,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    child = json.loads(done.stdout.strip().splitlines()[-1])
    here = collect(report_run)
    assert child["seed"] == here.config["seed"] == 7
    assert child["sha"] == here.config["dataset_sha256"]
    assert child["rows"] == 308 and child["n_pred"] == 2
    assert child["cat"] == {k: c.verdict.category for k, c in here.conclusions.items()}
    assert child["head"] == here.conclusions["A"].explanation.headline


def test_multi_model_run_uses_the_same_rule(trained_run: dict) -> None:
    """Con dos modelos por escenario (LR y RF) el veredicto sale de la regla."""
    data = report_data.collect_report_data(
        trained_run["run_id"],
        db_path=trained_run["db_path"],
        unstructured_path=trained_run["unstructured_path"],
        models_dir=trained_run["models_dir"],
    )
    for scenario, conclusion in data.conclusions.items():
        scores = recommend.scores_from_table(data.table, scenario)
        assert conclusion.verdict == recommend.evaluate_scenario(scores, scenario)
        assert set(conclusion.verdict.candidates) <= {
            "logistic_regression",
            "random_forest",
        }
        assert conclusion.importance is not None


def test_a_pdf_can_be_built_with_a_single_model_per_scenario(report_run: dict) -> None:
    """Entrenar un solo modelo por escenario (la pantalla lo permite) no rompe el PDF.

    Con una sola columna, el gráfico de matrices de confusión queda más alto que la
    página y ReportLab lanzaba LayoutError.
    """
    paths = {k: report_run[k] for k in ("db_path", "unstructured_path", "models_dir")}
    pdf = report.generate_report(report_run["run_id"], True, **paths)
    assert pdf.startswith(b"%PDF-") and pdf.rstrip().endswith(b"%%EOF")
    assert len(re.findall(rb"/Type /Page\b", pdf)) >= 4


@pytest.mark.parametrize("scenario", ["A", "B"])
def test_a_pdf_can_be_built_with_one_model_in_one_scenario(
    report_run: dict, scenario: str
) -> None:
    """Un solo modelo en un solo escenario (A o B) también arma el PDF completo."""
    base = collect(report_run)
    only = dataclasses.replace(
        base,
        table=base.table[base.table["scenario"] == scenario].reset_index(drop=True),
        docs=[d for d in base.docs if d["scenario"] == scenario],
        conclusions={scenario: base.conclusions[scenario]},
    )
    pdf = report.build_pdf(only, report_charts.render_charts(only))
    assert pdf.startswith(b"%PDF-") and len(re.findall(rb"/Type /Page\b", pdf)) >= 4


def test_report_needs_the_clean_data_too(report_run: dict, tmp_path: Path) -> None:
    """Con resultados guardados pero sin datos limpios (ETL borrado), lo dice."""
    empty_structured = tmp_path / "structured.db"
    db.init_structured_db(empty_structured)
    with db.get_connection(empty_structured) as conn:
        conn.execute(
            "INSERT INTO runs (run_id, started_at, status, seed) VALUES (?, 'ahora', 'ok', 7)",
            (report_run["run_id"],),
        )
    with pytest.raises(ValueError, match="ejecutá el ETL"):
        report_data.collect_report_data(
            report_run["run_id"],
            db_path=empty_structured,
            unstructured_path=report_run["unstructured_path"],
            models_dir=report_run["models_dir"],
        )
