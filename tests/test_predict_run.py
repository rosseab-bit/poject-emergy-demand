"""Tests de la predicción con los modelos guardados de una corrida."""

import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from support import (
    FRESH_MONGO_PRELUDE,
    dump_mongo,
    isolated_logging,
    isolated_mongo,
    mongo_logs,
    tiny_grids,
)

from src import config, data, logger, mongo, pipeline, predict, queries, results

LR, RF = "logistic_regression", "random_forest"


def input_for(db_path: Path, temperature: float = 15.0) -> predict.PredictionInput:
    """Entrada del mes siguiente armada con los datos limpios de una base."""
    return predict.build_input(data.load_clean_data(db_path), temperature)


@pytest.fixture(scope="module")
def predict_run(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Corrida propia con solo el escenario A (regresión logística y Random Forest)."""
    base = tmp_path_factory.mktemp("predict")
    paths = {
        "db_path": base / "structured.db",
        "models_dir": base / "models",
    }
    with isolated_logging(), isolated_mongo() as client, tiny_grids():
        run_id = pipeline.run_etl(db_path=paths["db_path"], seed=7)
        pipeline.run_training(
            run_id,
            db_path=paths["db_path"],
            models_dir=paths["models_dir"],
            scenarios=("A",),
            model_names=(LR, RF),
        )
        etl_only = pipeline.run_etl(db_path=paths["db_path"])
    return {"run_id": run_id, "etl_only": etl_only, "mongo": client, **paths}


@pytest.fixture
def use_logging(predict_run: dict) -> None:
    """Conecta el logger a la base de la corrida y lo restaura al terminar."""
    root = logging.getLogger(logger.ROOT_LOGGER_NAME)
    saved_level, saved_handlers = root.level, list(root.handlers)
    yield
    root.setLevel(saved_level)
    root.handlers[:] = saved_handlers


def test_predict_models_match_a_manual_prediction(trained_run: dict) -> None:
    """Las probabilidades coinciden con cargar el modelo y predecir a mano."""
    entry = input_for(trained_run["db_path"])
    predictions = predict.predict_models(
        trained_run["run_id"],
        entry,
        trained_run["models_dir"],
    )
    assert list(predictions) == [LR, RF]
    for name, prediction in predictions.items():
        doc = results.load_result(trained_run["run_id"], "A", name)
        estimator = results.load_model(doc, trained_run["models_dir"])
        x = entry.features[doc["features"]]
        assert prediction["probabilidad"] == pytest.approx(
            estimator.predict_proba(x)[0, 1]
        )
        assert prediction["clase"] == int(estimator.predict(x)[0])
        assert prediction["clase"] == int(prediction["probabilidad"] >= 0.5)
        assert 0.0 <= prediction["probabilidad"] <= 1.0
        assert prediction["f1_holdout"] == doc["holdout"]["f1"]
        assert prediction["f1_siempre_alta"] == pytest.approx(0.4)
        assert prediction["supera_referencia"] == (doc["holdout"]["f1"] > 0.4)


def test_predict_models_uses_scenario_a_not_b(trained_run: dict) -> None:
    """Se usan las 14 variables del escenario A, sin la potencia máxima."""
    entry = input_for(trained_run["db_path"])
    assert "potencia_maxima" not in entry.features.columns
    doc = results.load_result(trained_run["run_id"], "A", LR)
    assert doc["scenario"] == "A"
    assert "potencia_maxima" not in doc["features"]


def test_predict_models_rejects_a_run_without_scenario_a(predict_run: dict) -> None:
    """Una corrida sin el escenario A entrenado no permite predecir."""
    entry = input_for(predict_run["db_path"])
    for run_id in (predict_run["etl_only"], "noexiste"):
        with pytest.raises(ValueError, match="escenario A"):
            predict.predict_models(
                run_id,
                entry,
                predict_run["models_dir"],
            )


def test_predict_models_rejects_a_tampered_model(
    trained_run: dict, tmp_path: Path
) -> None:
    """Un .joblib alterado no se carga: falla por hash antes de predecir."""
    copy = tmp_path / "models"
    shutil.copytree(trained_run["models_dir"], copy)
    victim = next(copy.glob("*_A_random_forest.joblib"))
    victim.write_bytes(victim.read_bytes() + b"x")
    entry = input_for(trained_run["db_path"])
    with pytest.raises(ValueError, match="hash"):
        predict.predict_models(trained_run["run_id"], entry, copy)


def test_prediction_doc_is_json_and_complete(trained_run: dict) -> None:
    """El documento trae mes, temperatura, 14 variables, rezagos y los modelos."""
    entry = input_for(trained_run["db_path"], 16.5)
    predictions = predict.predict_models(
        trained_run["run_id"],
        entry,
        trained_run["models_dir"],
    )
    doc = predict.build_prediction_doc(entry, predictions)
    decoded = json.loads(results.to_json(doc))
    assert (decoded["fecha"], decoded["mes"], decoded["anio"]) == (
        "2026-09-01",
        9,
        2026,
    )
    assert decoded["temperatura"] == 16.5
    assert len(decoded["variables"]) == 14
    assert len(decoded["rezagos"]) == 8
    assert set(decoded["modelos"]) == {LR, RF}
    assert decoded["scenario"] == "A"


def test_predict_next_month_saves_and_can_be_read_back(predict_run: dict) -> None:
    """La predicción queda guardada en MongoDB y se recupera con la capa de lectura."""
    doc = predict.predict_next_month(
        predict_run["run_id"], 15.0, **only_paths(predict_run)
    )
    assert doc["fecha"] == "2026-09-01"
    assert len(doc["id"]) == 8 and int(doc["id"], 16) >= 0  # id corto, como el run_id
    stored = queries.latest_prediction(predict_run["run_id"])
    assert stored["id"] == doc["id"]
    assert stored["modelos"][LR]["probabilidad"] == doc["modelos"][LR]["probabilidad"]
    raw = mongo.get_database()[mongo.PREDICTIONS].find_one(
        {"id": doc["id"]}, {"modelos.logistic_regression.probabilidad": 1, "_id": 0}
    )
    assert raw["modelos"][LR]["probabilidad"] == doc["modelos"][LR]["probabilidad"]


def only_paths(run: dict) -> dict:
    """Las tres rutas que recibe predict_next_month."""
    return {k: run[k] for k in ("db_path", "models_dir")}


def test_the_expected_temperature_changes_the_prediction(predict_run: dict) -> None:
    """Dos temperaturas distintas dan probabilidades distintas (la entrada importa)."""
    cold = predict.predict_next_month(
        predict_run["run_id"], 8.0, **only_paths(predict_run)
    )
    hot = predict.predict_next_month(
        predict_run["run_id"], 28.0, **only_paths(predict_run)
    )
    assert cold["id"] != hot["id"]
    assert cold["modelos"][LR]["probabilidad"] != hot["modelos"][LR]["probabilidad"]


@pytest.mark.usefixtures("use_logging")
def test_prediction_events_are_logged_under_the_run(predict_run: dict) -> None:
    """Inicio, guardado y fin quedan en logs con el run_id, sin eventos huérfanos."""
    before = len(mongo_logs())
    doc = predict.predict_next_month(
        predict_run["run_id"], 15.0, **only_paths(predict_run)
    )
    rows = mongo_logs()[before:]  # ya salieron todos: se envían al cerrar la predicción
    messages = [r["message"] for r in rows]
    assert {r["run_id"] for r in rows} == {predict_run["run_id"]}
    assert any(m.startswith(f"Predicción guardada (id {doc['id']})") for m in messages)
    assert messages[0] == "Inicio: _predict_and_save"
    assert messages[-1].startswith("Fin: _predict_and_save")
    assert not any(r["level"] == "ERROR" for r in rows)
    assert {r["step"] for r in rows if "step" in r} >= {"_predict_and_save"}


@pytest.mark.usefixtures("use_logging")
def test_invalid_temperature_saves_nothing_and_logs_the_error(
    predict_run: dict,
) -> None:
    """Una temperatura inválida no guarda predicción y deja el error en logs."""
    before = len(queries.prediction_history(predict_run["run_id"]))
    with pytest.raises(ValueError, match="temperatura"):
        predict.predict_next_month(
            predict_run["run_id"], 99.0, **only_paths(predict_run)
        )
    after = len(queries.prediction_history(predict_run["run_id"]))
    assert after == before
    errors = queries.get_logs(predict_run["run_id"], levels=["ERROR"])
    assert errors["message"].str.contains("temperatura").any()


def test_predict_next_month_rejects_unknown_run_and_untrained_run(
    predict_run: dict,
) -> None:
    """Una corrida inexistente o sin entrenar no permite predecir."""
    with pytest.raises(ValueError, match="No existe la corrida"):
        predict.predict_next_month("noexiste", 15.0, **only_paths(predict_run))
    with pytest.raises(ValueError, match="escenario A"):
        predict.predict_next_month(
            predict_run["etl_only"], 15.0, **only_paths(predict_run)
        )


def test_prediction_sends_its_events_and_saves_only_in_mongodb(
    predict_run: dict,
) -> None:
    """Después de predecir, el documento y los eventos ya están en MongoDB."""
    doc = predict.predict_next_month(
        predict_run["run_id"], 16.0, **only_paths(predict_run)
    )
    assert (
        mongo.get_database()[mongo.PREDICTIONS].count_documents({"id": doc["id"]}) == 1
    )
    messages = [e["message"] for e in mongo_logs(predict_run["run_id"])]
    assert any(m.startswith(f"Predicción guardada (id {doc['id']})") for m in messages)
    assert not list(predict_run["db_path"].parent.glob("unstructured*"))


def test_prediction_works_in_a_fresh_process(predict_run: dict, tmp_path: Path) -> None:
    """Funciona después de reiniciar: un proceso nuevo predice solo desde el disco."""
    code = FRESH_MONGO_PRELUDE + (
        "import json; from src import predict;"
        "a = sys.argv; doc = predict.predict_next_month(a[1], 14.0,"
        " db_path=Path(a[2]), models_dir=Path(a[3]));"
        "from src import queries; saved = queries.latest_prediction(a[1]);"
        "print(json.dumps({'id': doc['id'], 'saved': saved['id'], 'p': doc['modelos']['logistic_regression']['probabilidad']}))"
    )
    dump = dump_mongo(predict_run["mongo"], tmp_path / "mongo.json")
    done = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            predict_run["run_id"],
            str(predict_run["db_path"]),
            str(predict_run["models_dir"]),
            str(dump),
        ],
        cwd=config.ROOT_DIR,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    child = json.loads(done.stdout.strip().splitlines()[-1])
    assert child["saved"] == child["id"]  # el hijo la guardó y la releyó de MongoDB
    # El proceso de prueba, con otra memoria, obtiene la misma probabilidad.
    again = predict.predict_next_month(
        predict_run["run_id"], 14.0, **only_paths(predict_run)
    )
    assert again["modelos"][LR]["probabilidad"] == pytest.approx(child["p"])


def test_prediction_history_is_newest_first_and_filterable(predict_run: dict) -> None:
    """El historial va de la más reciente a la más vieja y se filtra por corrida."""
    predict.predict_next_month(predict_run["run_id"], 12.0, **only_paths(predict_run))
    history = queries.prediction_history(predict_run["run_id"])
    assert history["created_at"].tolist() == sorted(history["created_at"], reverse=True)
    assert history["temperatura"].iloc[0] == 12.0
    assert set(history["run_id"]) == {predict_run["run_id"]}
    assert (history["fecha"] == "2026-09-01").all()
    assert history[f"p_{LR}"].between(0, 1).all()
    assert history["p_xgboost"].isna().all()  # ese modelo no se entrenó en esta corrida
    assert len(queries.prediction_history(None)) >= len(history)
    assert queries.prediction_history("noexiste").empty
    assert queries.latest_prediction(predict_run["etl_only"]) is None
