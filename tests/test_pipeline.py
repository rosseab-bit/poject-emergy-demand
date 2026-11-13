"""Tests de src/pipeline.py."""

from pathlib import Path

import pytest
from support import count_rows, main_file_only, mongo_logs, orphan_logs

from src import config, data, db, pipeline, queries, results, runs, train

# Integración: corre el ETL y el entrenamiento de verdad.
pytestmark = pytest.mark.slow


def test_run_etl_end_to_end(tmp_path: Path) -> None:
    """El ETL deja 308 filas (259 completas) y todo logueado con su run_id."""
    db_path = tmp_path / "structured.db"
    run_id = pipeline.run_etl(db_path=db_path)

    clean = data.load_clean_data(db_path)
    assert len(clean) == 308
    assert data.count_complete_rows(clean) == 259

    run = runs.get_run(run_id)
    assert run["status"] == "ok"
    assert run["finished_at"] is not None

    rows = mongo_logs(run_id)
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
    # El último evento ocurre fuera de todo paso: sale al cerrar la corrida.
    assert rows[-1]["message"] == f"Corrida terminada: run_id={run_id}"
    assert orphan_logs() == 0


def test_run_etl_events_are_differentiated_by_step(tmp_path: Path) -> None:
    """Los eventos del ETL dicen a qué paso pertenecen, con inicio, fin y duración."""
    run_id = pipeline.run_etl(db_path=tmp_path / "structured.db")
    rows = mongo_logs(run_id)
    steps = {r["step"] for r in rows if "step" in r}
    assert {"run_etl_steps", "load_raw", "save_raw_data", "save_clean_data"} <= steps
    inside = [r for r in rows if r["message"].startswith("CSV leído")]
    assert inside[0]["step"] == "load_raw"  # un evento común, etiquetado con su paso
    starts = [r for r in rows if r.get("event") == "inicio"]
    ends = [r for r in rows if r.get("event") == "fin"]
    assert len(starts) == len(ends) > 5
    assert all(isinstance(r["duration_ms"], int) for r in ends)
    assert all("duration_ms" not in r for r in starts)


def test_run_etl_failure_is_logged_and_run_marked_error(tmp_path: Path) -> None:
    """Si el CSV es inválido, la corrida queda en error y el error en logs."""
    db_path = tmp_path / "structured.db"
    bad_csv = tmp_path / "malo.csv"
    bad_csv.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="Faltan columnas"):
        pipeline.run_etl(csv_path=bad_csv, db_path=db_path)

    [run] = queries.list_runs().to_dict("records")
    errors = mongo_logs(run["run_id"], level="ERROR")
    assert run["status"] == "error"
    assert len(errors) >= 1
    assert {e["event"] for e in errors if "event" in e} == {"error"}


# Grilla completa de hiperparámetros: es el test de integración real.
def test_run_pipeline_trains_saves_and_links_everything(tmp_path: Path) -> None:
    """La corrida completa entrena 6 resultados y los guarda atados al run_id."""
    db_path = tmp_path / "structured.db"
    udb_path = tmp_path / "unstructured.db"
    models_dir = tmp_path / "models"
    run_id = pipeline.run_pipeline(
        db_path=db_path, unstructured_path=udb_path, models_dir=models_dir
    )

    assert runs.get_run(run_id)["status"] == "ok"
    rows = mongo_logs(run_id)
    assert orphan_logs() == 0
    with db.get_connection(db_path) as conn:
        db.attach_unstructured(conn, udb_path)
        joined = conn.execute(
            "SELECT m.scenario, m.model, json_extract(m.doc, '$.holdout.f1') AS f1 "
            "FROM unstructured.resultados_modelos m WHERE m.run_id = ?",
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


def test_run_etl_records_the_given_seed(tmp_path: Path) -> None:
    """La semilla elegida queda en la colección runs."""
    db_path = tmp_path / "structured.db"
    run_id = pipeline.run_etl(db_path=db_path, seed=7)
    assert runs.get_run(run_id)["seed"] == 7
    other = pipeline.run_etl(db_path=db_path)
    assert runs.get_run(other)["seed"] == config.SEED


LR = ("logistic_regression",)


@pytest.fixture
def paths(tmp_path: Path) -> dict:
    """Rutas temporales de las dos bases y de los modelos."""
    return {
        "db_path": tmp_path / "structured.db",
        "unstructured_path": tmp_path / "unstructured.db",
        "models_dir": tmp_path / "models",
    }


@pytest.mark.usefixtures("small_grids")
def test_training_continues_the_etl_run(paths: dict) -> None:
    """Entrenar continúa el run_id del ETL: una sola corrida, todo atado a ella."""
    run_id = pipeline.run_etl(db_path=paths["db_path"], seed=7)
    calls = []
    same = pipeline.run_training(
        run_id,
        scenarios=("A",),
        model_names=LR,
        on_progress=lambda d, t, label: calls.append((d, t, label)),
        **paths,
    )
    assert same == run_id
    assert calls == [(1, 1, "A/logistic_regression")]

    assert len(queries.list_runs()) == 1
    messages = [e["message"] for e in mongo_logs(run_id)]
    assert orphan_logs() == 0
    assert any("datos_limpios guardada" in m for m in messages)  # ETL
    assert any("Resultado guardado: A/logistic_regression" in m for m in messages)
    run = runs.get_run(run_id)
    assert (run["status"], run["seed"]) == ("ok", 7)
    doc = results.load_result(
        run_id, "A", "logistic_regression", paths["unstructured_path"]
    )
    assert doc["model"] == "logistic_regression"
    with db.get_connection(paths["unstructured_path"]) as conn:
        seed = conn.execute(
            "SELECT json_extract(doc, '$.seed') FROM configuracion WHERE run_id = ?",
            (run_id,),
        ).fetchone()[0]
    assert seed == 7


@pytest.mark.usefixtures("small_grids")
def test_a_finished_run_leaves_both_database_files_complete(
    paths: dict, tmp_path: Path
) -> None:
    """Al terminar cada paso, copiar solo los .db (sin -wal) trae todo lo escrito."""
    run_id = pipeline.run_etl(db_path=paths["db_path"], seed=7)
    etl_copy = main_file_only(paths["db_path"], tmp_path / "tras_etl")
    assert count_rows(etl_copy, "datos_limpios") == 308
    assert count_rows(etl_copy, "datos_originales") == 308
    assert len(queries.get_logs(run_id)) > 0  # los logs ya están en MongoDB

    pipeline.run_training(
        run_id, scenarios=("A",), model_names=("logistic_regression",), **paths
    )
    structured = main_file_only(paths["db_path"], tmp_path / "tras_entrenar")
    unstructured = main_file_only(
        paths["unstructured_path"], tmp_path / "tras_entrenar"
    )
    assert count_rows(structured, "datos_originales") == 308
    assert count_rows(unstructured, "resultados_modelos") == 1
    assert count_rows(unstructured, "configuracion") == 1
    for db_file in (paths["db_path"], paths["unstructured_path"]):
        wal = Path(f"{db_file}-wal")
        assert not wal.exists() or wal.stat().st_size == 0


@pytest.mark.usefixtures("small_grids")
def test_training_twice_on_same_run_is_rejected(paths: dict) -> None:
    """Una corrida no se entrena dos veces: hay que iniciar otra."""
    run_id = pipeline.run_etl(db_path=paths["db_path"])
    pipeline.run_training(run_id, scenarios=("A",), model_names=LR, **paths)
    with pytest.raises(ValueError, match="ya tiene resultados"):
        pipeline.run_training(run_id, scenarios=("A",), model_names=LR, **paths)


def test_training_rejects_unknown_run_and_missing_etl(paths: dict) -> None:
    """Sin corrida o sin ETL previo se pide ejecutar el ETL, sin tocar nada."""
    with pytest.raises(ValueError, match="No existe la corrida"):
        pipeline.run_training("noexiste", **paths)
    db.init_structured_db(paths["db_path"])
    empty_run = runs.create_run()
    with pytest.raises(ValueError, match="Primero ejecutá el ETL"):
        pipeline.run_training(empty_run, **paths)
    assert runs.get_run(empty_run)["status"] == "running"


def test_failed_training_marks_run_as_error(
    paths: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Si el entrenamiento falla, la corrida queda en error y el error en logs."""
    run_id = pipeline.run_etl(db_path=paths["db_path"])

    def boom(*args, **kwargs):
        raise RuntimeError("falla de prueba")

    monkeypatch.setattr(train, "train_all", boom)
    with pytest.raises(RuntimeError, match="falla de prueba"):
        pipeline.run_training(run_id, scenarios=("A",), model_names=LR, **paths)
    assert runs.get_run(run_id)["status"] == "error"
    assert len(mongo_logs(run_id, level="ERROR")) >= 1
