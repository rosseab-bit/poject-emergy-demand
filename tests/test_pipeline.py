"""Tests de src/pipeline.py."""

from pathlib import Path

import pytest

from src import data, db, pipeline, results


@pytest.mark.usefixtures("clean_root_logger")
def test_run_etl_end_to_end(tmp_path: Path) -> None:
    """El ETL deja 308 filas (259 completas) y todo logueado con su run_id."""
    db_path = tmp_path / "structured.db"
    run_id = pipeline.run_etl(db_path=db_path)

    clean = data.load_clean_data(db_path)
    assert len(clean) == 308
    assert data.count_complete_rows(clean) == 259

    run = db.get_run(run_id, db_path)
    assert run["status"] == "ok"
    assert run["finished_at"] is not None

    with db.get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT level, module, message FROM logs WHERE run_id = ?", (run_id,)
        ).fetchall()
    messages = " | ".join(r["message"] for r in rows)
    for expected in (
        "Inicio: load_raw",
        "CSV leído: 308",
        "repetidos: 2",
        "datos_limpios guardada: 308",
        "Fin: build_dataset",
        "Balance de clases: 236 meses con target, 60 positivos (25.4 %)",
        "Fin: run_etl_steps",
    ):
        assert expected in messages
    assert sum(r["level"] == "WARNING" for r in rows) == 2
    assert not any(r["level"] == "ERROR" for r in rows)
    with db.get_connection(db_path) as conn:
        orphans = conn.execute("SELECT COUNT(*) FROM logs WHERE run_id IS NULL")
        assert orphans.fetchone()[0] == 0


@pytest.mark.usefixtures("clean_root_logger")
def test_run_etl_failure_is_logged_and_run_marked_error(tmp_path: Path) -> None:
    """Si el CSV es inválido, la corrida queda en error y el error en logs."""
    db_path = tmp_path / "structured.db"
    bad_csv = tmp_path / "malo.csv"
    bad_csv.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="Faltan columnas"):
        pipeline.run_etl(csv_path=bad_csv, db_path=db_path)

    with db.get_connection(db_path) as conn:
        run = conn.execute("SELECT run_id, status FROM runs").fetchone()
        errors = conn.execute(
            "SELECT message FROM logs WHERE level = 'ERROR' AND run_id = ?",
            (run["run_id"],),
        ).fetchall()
    assert run["status"] == "error"
    assert len(errors) >= 1


@pytest.mark.usefixtures("clean_root_logger")
def test_run_pipeline_trains_saves_and_links_everything(tmp_path: Path) -> None:
    """La corrida completa entrena 6 resultados y los guarda atados al run_id."""
    db_path = tmp_path / "structured.db"
    udb_path = tmp_path / "unstructured.db"
    models_dir = tmp_path / "models"
    run_id = pipeline.run_pipeline(
        db_path=db_path, unstructured_path=udb_path, models_dir=models_dir
    )

    assert db.get_run(run_id, db_path)["status"] == "ok"
    with db.get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT level, message FROM logs WHERE run_id = ?", (run_id,)
        ).fetchall()
        orphans = conn.execute("SELECT COUNT(*) FROM logs WHERE run_id IS NULL")
        assert orphans.fetchone()[0] == 0
        db.attach_unstructured(conn, udb_path)
        joined = conn.execute(
            "SELECT m.scenario, m.model, json_extract(m.doc, '$.holdout.f1') AS f1 "
            "FROM runs r JOIN unstructured.resultados_modelos m "
            "ON m.run_id = r.run_id WHERE r.run_id = ?",
            (run_id,),
        ).fetchall()
        config_rows = conn.execute(
            "SELECT json_extract(doc, '$.seed') AS seed "
            "FROM unstructured.configuracion WHERE run_id = ?",
            (run_id,),
        ).fetchall()

    messages = [r["message"] for r in rows]
    trained = [
        m for m in messages if m.startswith("Resultado ") and "F1 validación" in m
    ]
    assert len(trained) == 6
    assert sum(m.startswith("Resultado guardado:") for m in messages) == 6
    assert not any(r["level"] == "ERROR" for r in rows)

    expected = {
        (s, m)
        for s in ("A", "B")
        for m in ("logistic_regression", "random_forest", "xgboost")
    }
    assert {(r["scenario"], r["model"]) for r in joined} == expected
    assert all(0.0 <= r["f1"] <= 1.0 for r in joined)
    assert [r["seed"] for r in config_rows] == [42]

    assert len(list(models_dir.glob("*.joblib"))) == 6
    for scenario, model in expected:
        doc = results.load_result(run_id, scenario, model, udb_path)
        assert results.load_model(doc, models_dir) is not None
