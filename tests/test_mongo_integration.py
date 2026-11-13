"""Integración con el cluster real de MongoDB (Atlas).

Se omiten salvo que se pidan con ``pytest -m mongo``, y también si no hay un
``.env`` con credenciales. Escriben solo en una colección temporal de una base
``<MONGODB_DB>_tests``, que se borra al terminar.
"""

import logging
import time
import uuid
from collections.abc import Iterator

import pytest
from pymongo.collection import Collection

from src import logger, mongo, queries, runs

pytestmark = pytest.mark.mongo


@pytest.fixture
def real_cluster() -> Iterator[None]:
    """Usa la configuración real del ``.env``; se omite si falta."""
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
