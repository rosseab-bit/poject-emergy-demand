"""Integración con el cluster real de MongoDB (Atlas).

Se omiten salvo que se pidan con ``pytest -m mongo``, y también si no hay un
``.env`` con credenciales. Escriben solo en una colección temporal de una base
``<MONGODB_DB>_tests``, que se borra al terminar.
"""

import uuid
from collections.abc import Iterator

import pytest
from pymongo.collection import Collection

from src import mongo

pytestmark = pytest.mark.mongo


@pytest.fixture
def real_cluster() -> Iterator[None]:
    """Usa la configuración real del ``.env``; se omite si falta."""
    try:
        mongo.load_settings()
    except mongo.MongoConfigError as error:
        pytest.skip(f"sin credenciales de MongoDB: {error}")
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
