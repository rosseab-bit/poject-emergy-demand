"""Integración con el cluster real de MongoDB (Atlas).

Se omiten salvo que se pidan con ``pytest -m mongo``, y también si no hay un
``.env`` con credenciales. Escriben solo en una colección temporal de una base
``<MONGODB_DB>_tests``, que se borra al terminar.
"""

import logging
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from conftest import REAL_MONGO
from pymongo.collection import Collection
from pymongo.database import Database
from support import tiny_grids

from src import config, logger, mongo, pipeline, predict, queries, results, runs

pytestmark = pytest.mark.mongo


@pytest.fixture
def real_cluster(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Levanta la barrera de ``conftest`` y usa el ``.env`` real; se omite si falta.

    Es el único lugar de los tests que llega al cluster: los tests marcados
    ``mongo`` y solo con ``pytest -m mongo``.
    """
    monkeypatch.setattr(config, "ENV_PATH", Path(REAL_MONGO["env_path"]))
    for name, value in REAL_MONGO["variables"].items():
        monkeypatch.setenv(name, value)
    try:
        mongo.load_settings()
    except mongo.MongoConfigError as error:
        pytest.skip(f"sin credenciales de MongoDB: {error}")
    mongo.reset_client()  # reemplaza el Mongo simulado por el cliente real
    yield


@pytest.fixture
def temp_collection(real_cluster: None) -> Iterator[Collection]:
    """Una colección temporal en una base de pruebas, borrada al terminar."""
    settings = mongo.load_settings()
    database = mongo.get_client()[f"{settings.db_name}_tests"]
    collection = database[f"tmp_{uuid.uuid4().hex[:12]}"]
    yield collection
    collection.drop()


def test_cluster_answers_ping(real_cluster: None) -> None:
    """El cluster acepta la conexión y las credenciales."""
    mongo.check_connection()


def test_user_can_write_read_and_delete(temp_collection: Collection) -> None:
    """El usuario puede insertar, leer y borrar (permisos de escritura)."""
    inserted = temp_collection.insert_one({"run_id": "prueba", "valor": 1})
    found = temp_collection.find_one({"_id": inserted.inserted_id})
    assert found is not None and found["valor"] == 1
    assert temp_collection.delete_one({"_id": inserted.inserted_id}).deleted_count == 1


def test_logger_batch_roundtrip_on_the_real_cluster(
    temp_collection: Collection,
) -> None:
    """Un lote de eventos viaja en una operación y se lee igual que se guardó."""
    handler = logger.MongoHandler(temp_collection)
    test_logger = logging.getLogger("src.integration_probe")
    test_logger.setLevel(logging.INFO)
    test_logger.addHandler(handler)
    token = logger.current_run_id.set("integracion")
    try:
        for index in range(50):
            test_logger.info("evento %d", index, extra={"step": "paso", "event": "fin"})
        started = time.perf_counter()
        handler.flush()
        elapsed = time.perf_counter() - started
    finally:
        logger.current_run_id.reset(token)
        test_logger.removeHandler(handler)
    docs = list(temp_collection.find({"run_id": "integracion"}).sort("_id", 1))
    assert [d["message"] for d in docs] == [f"evento {i}" for i in range(50)]
    assert {d["step"] for d in docs} == {"paso"}
    assert docs[0]["ts"].year >= 2026  # datetime guardado como fecha, no como texto
    assert elapsed < 5  # 50 eventos en un solo viaje, no 50 viajes


def test_runs_roundtrip_on_the_real_cluster(real_cluster: None) -> None:
    """Crear, cerrar y leer una corrida funciona contra Atlas (base de pruebas)."""
    settings = mongo.load_settings()
    mongo.use_client(mongo.get_client(), f"{settings.db_name}_tests")
    runs_collection = mongo.get_database()[mongo.RUNS]
    try:
        run_id = runs.create_run(seed=123, notes="integración")
        assert runs.get_run(run_id)["status"] == "running"
        runs.finish_run(run_id, runs.STATUS_OK)
        run = runs.get_run(run_id)
        assert (run["status"], run["seed"]) == ("ok", 123)
        assert queries.list_runs()["run_id"].tolist()[0] == run_id
    finally:
        runs_collection.drop()


@pytest.fixture
def tests_database(real_cluster: None) -> Iterator[Database]:
    """La base de pruebas del cluster real, con sus colecciones listas y borradas al salir."""
    settings = mongo.load_settings()
    mongo.use_client(mongo.get_client(), f"{settings.db_name}_tests")
    database = mongo.get_database()
    mongo.init_mongo()
    yield database
    for name in database.list_collection_names():
        database.drop_collection(name)


def test_a_full_training_run_is_saved_and_read_back_on_the_real_cluster(
    tests_database: Database, tmp_path: Path
) -> None:
    """ETL, entrenamiento, predicción y lectura contra Atlas, en la base de pruebas.

    Atlas es más estricto que el Mongo simulado (tipos de numpy, nombres de campos),
    así que esta es la prueba que importa para el guardado de resultados.
    """
    paths = {"db_path": tmp_path / "structured.db"}
    with tiny_grids():
        run_id = pipeline.run_etl(seed=7, **paths)
        pipeline.run_training(
            run_id,
            models_dir=tmp_path / "models",
            scenarios=("A",),
            model_names=("logistic_regression", "random_forest"),
            **paths,
        )
    prediction = predict.predict_next_month(
        run_id, 15.0, models_dir=tmp_path / "models", **paths
    )

    assert queries.run_summary(run_id)["steps"] == {
        "ETL": True,
        "Entrenamiento": True,
        "Predicción": True,
        "Reporte": False,
    }
    table = queries.results_table(run_id)
    assert table["model"].tolist() == ["logistic_regression", "random_forest"]
    assert table["f1"].between(0, 1).all()
    doc = results.load_result(run_id, "A", "random_forest")
    assert len(doc["predictions"]) == 48
    assert results.load_run_config(run_id)["seed"] == 7
    assert queries.latest_prediction(run_id)["id"] == prediction["id"]
    assert queries.prediction_history(run_id)["id"].tolist() == [prediction["id"]]
    events = queries.get_logs(run_id)
    assert (events["event"] == "fin").any() and events["run_id"].eq(run_id).all()
    with pytest.raises(Exception, match="duplicate|E11000"):
        tests_database[mongo.PREDICTIONS].insert_one({"id": prediction["id"]})
