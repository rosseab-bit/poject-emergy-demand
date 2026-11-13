"""Tests de la preparación de MongoDB al arrancar: colecciones e índices."""

import mongomock
import pytest
from pymongo.errors import DuplicateKeyError, OperationFailure, PyMongoError

from src import mongo


def index_names(database: mongomock.Database, collection: str) -> set[str]:
    """Nombres de los índices de una colección, sin el ``_id_`` automático."""
    return set(database[collection].index_information()) - {"_id_"}


def test_init_mongo_creates_every_collection_with_its_indexes(
    mongo_db: mongomock.Database,
) -> None:
    """Arranca con la base vacía y deja las cinco colecciones con sus índices."""
    assert mongo_db.list_collection_names() == []
    mongo.init_mongo()
    assert set(mongo_db.list_collection_names()) == set(mongo.COLLECTIONS)
    assert set(mongo.COLLECTIONS) == {
        "logs",
        "runs",
        "resultados_modelos",
        "configuracion",
        "predicciones",
    }
    for name, specs in mongo.COLLECTIONS.items():
        assert index_names(mongo_db, name) == {spec.name for spec in specs}


def test_unique_indexes_have_the_expected_keys(mongo_db: mongomock.Database) -> None:
    """Un resultado por corrida, escenario y modelo; una configuración y una corrida."""
    mongo.init_mongo()
    results = mongo_db[mongo.RESULTS].index_information()
    assert results["uq_resultados_run_scenario_model"]["unique"] is True
    assert [k for k, _ in results["uq_resultados_run_scenario_model"]["key"]] == [
        "run_id",
        "scenario",
        "model",
    ]
    assert mongo_db[mongo.RUNS].index_information()["uq_runs_run_id"]["unique"]
    assert mongo_db[mongo.CONFIGURATION].index_information()["uq_configuracion_run_id"][
        "unique"
    ]


def test_unique_index_rejects_a_duplicate_result(mongo_db: mongomock.Database) -> None:
    """Guardar dos veces el mismo resultado de una corrida falla."""
    mongo.init_mongo()
    key = {"run_id": "r1", "scenario": "A", "model": "xgboost"}
    mongo_db[mongo.RESULTS].insert_one(dict(key))
    with pytest.raises(DuplicateKeyError):
        mongo_db[mongo.RESULTS].insert_one(dict(key))
    mongo_db[mongo.RESULTS].insert_one({**key, "scenario": "B"})  # otro escenario


def test_init_mongo_is_idempotent_and_keeps_existing_data(
    mongo_db: mongomock.Database,
) -> None:
    """Sobre una base ya preparada no borra ni modifica nada."""
    mongo.init_mongo()
    mongo_db[mongo.LOGS].insert_one({"run_id": "r1", "message": "hola"})
    mongo.init_mongo()
    mongo.init_mongo()
    assert mongo_db[mongo.LOGS].count_documents({}) == 1
    assert index_names(mongo_db, mongo.LOGS) == {"idx_logs_run_ts"}


def test_init_mongo_adds_only_what_is_missing(mongo_db: mongomock.Database) -> None:
    """Con una colección ya existente y sin índices, solo completa lo que falta."""
    mongo_db.create_collection(mongo.RUNS)
    mongo_db[mongo.RUNS].insert_one({"run_id": "viejo"})
    mongo.init_mongo()
    assert set(mongo_db.list_collection_names()) == set(mongo.COLLECTIONS)
    assert mongo_db[mongo.RUNS].count_documents({}) == 1
    assert "uq_runs_run_id" in index_names(mongo_db, mongo.RUNS)


def test_init_mongo_accepts_an_explicit_database() -> None:
    """Se puede preparar una base distinta de la compartida."""
    other = mongomock.MongoClient()["otra"]
    mongo.init_mongo(other)
    assert set(other.list_collection_names()) == set(mongo.COLLECTIONS)


def test_init_mongo_tolerates_a_collection_created_in_the_meantime(
    mongo_db: mongomock.Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Si otro proceso crea la colección a la vez, sigue sin error."""
    real_create = mongo_db.create_collection

    def racing_create(name: str, **kwargs: object) -> object:
        real_create(name)  # "el otro proceso" la crea primero
        return real_create(name)  # y la creación propia falla: ya existe

    monkeypatch.setattr(mongo_db, "create_collection", racing_create)
    mongo.init_mongo()
    assert set(mongo_db.list_collection_names()) == set(mongo.COLLECTIONS)


def test_init_mongo_reports_an_unreachable_server_without_secrets() -> None:
    """Si el servidor no responde, el error es de conexión y claro."""

    class Down:
        class admin:
            @staticmethod
            def command(name: str) -> None:
                raise PyMongoError("sin red")

    class Database:
        client = Down()
        name = "x"

    with pytest.raises(mongo.MongoUnavailableError, match="No se pudo conectar"):
        mongo.init_mongo(Database())  # type: ignore[arg-type]


def test_init_mongo_reports_missing_permissions(mongo_db: mongomock.Database) -> None:
    """Si el usuario no puede crear colecciones, el mensaje lo dice con claridad."""

    class ReadOnly:
        name = "x"
        client = mongo_db.client

        @staticmethod
        def list_collection_names() -> list[str]:
            raise OperationFailure("not authorized on x to execute command")

    with pytest.raises(mongo.MongoUnavailableError, match="preparar las colecciones"):
        mongo.init_mongo(ReadOnly())  # type: ignore[arg-type]


def test_init_mongo_without_configuration_fails_with_a_config_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Sin cliente instalado ni ``.env`` el error es de configuración, no de red."""
    mongo.reset_client()
    monkeypatch.setattr(mongo.config, "ENV_PATH", tmp_path / ".env")
    for name in (mongo.ENV_URI, mongo.ENV_PASSWORD, mongo.ENV_DB_NAME):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(mongo.MongoConfigError):
        mongo.init_mongo()
