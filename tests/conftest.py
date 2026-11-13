"""Fixtures compartidas y reglas de colección de los tests."""

import os
from pathlib import Path

# BARRERA contra el cluster real. Esto corre antes de importar ``src``: durante los
# tests el ``.env`` real y las variables de conexión quedan fuera de alcance, también
# para los procesos hijos (heredan el entorno). Un test que llegue a MongoDB sin un
# cliente simulado falla con un error de configuración, en vez de escribir en Atlas.
# Los tests marcados ``mongo`` recuperan las credenciales con ``REAL_MONGO``.
REAL_MONGO = {
    "env_path": os.environ.get("TP_ENV_PATH")
    or str(Path(__file__).parent.parent / ".env"),
    "variables": {
        name: os.environ.pop(name)
        for name in ("URI", "MONGODB_PASS", "MONGODB_DB")
        if name in os.environ
    },
}
os.environ["TP_ENV_PATH"] = str(Path(__file__).parent / "no_existe" / ".env")

import logging
from collections.abc import Iterator

import mongomock
import pytest
import streamlit as st
from support import isolated_logging, isolated_mongo, tiny_grids, using_mongo

from src import config, logger, mongo, pipeline

# Fixtures que entrenan modelos reales: los tests que las usan se marcan ``slow``.
HEAVY_FIXTURES = {
    "trained_run",
    "use_trained",
    "trained_report_data",
    "predict_run",
    "report_run",
    "trained",
    "small_grids",
}
# Los archivos que recorren la app completa con AppTest también son ``slow``.
APP_TEST_PREFIX = "test_app"


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Marca como ``slow`` lo que entrena modelos o recorre la app completa.

    Los tests ``mongo`` hablan con el cluster real: se omiten salvo que se pidan
    con ``pytest -m mongo``.
    """
    run_mongo = "mongo" in (config.getoption("markexpr") or "")
    for item in items:
        if "mongo" in item.keywords and not run_mongo:
            item.add_marker(
                pytest.mark.skip(reason="cluster real: usar pytest -m mongo")
            )
        uses_trained_run = HEAVY_FIXTURES & set(getattr(item, "fixturenames", ()))
        is_app_test = item.module.__name__.split(".")[-1].startswith(APP_TEST_PREFIX)
        if uses_trained_run or is_app_test:
            item.add_marker(pytest.mark.slow)


@pytest.fixture(autouse=True)
def clean_root_logger() -> Iterator[None]:
    """Deja el logger del paquete como estaba al terminar CADA test.

    Es automática: no depende de que el test se acuerde de pedirla. Sin esto, un
    ``SQLiteHandler`` que quedara enganchado haría que cada evento de log de los
    tests siguientes se escribiera también en bases temporales ya olvidadas, lo
    que los volvía cientos de veces más lentos y ensuciaba bases ajenas.
    """
    root = logging.getLogger(logger.ROOT_LOGGER_NAME)
    old_level, old_handlers = root.level, list(root.handlers)
    yield
    root.setLevel(old_level)
    root.handlers[:] = old_handlers


# Fixtures de módulo o de sesión que corren el pipeline: cada una trae su propio
# MongoDB simulado en la clave "mongo" de su diccionario.
MONGO_FIXTURES = ("trained_run", "predict_run", "report_run", "trained")


@pytest.fixture(autouse=True)
def mongo_db(request: pytest.FixtureRequest) -> Iterator[mongomock.Database]:
    """Una base MongoDB simulada como cliente compartido de CADA test.

    Es automática para que ningún test llegue por accidente al cluster real
    configurado en el ``.env``. Si el test usa una fixture de módulo o de sesión
    que ya corrió el pipeline, se activa el MongoDB simulado de esa fixture, que
    es donde quedaron sus logs y resultados. Al terminar restaura el cliente
    anterior y vacía el caché de Streamlit (la preparación de MongoDB de la app
    se guarda ahí).
    """
    client = mongomock.MongoClient()
    previous = mongo.use_client(client, "demanda_test")
    for name in MONGO_FIXTURES:
        if name in request.fixturenames:
            client = request.getfixturevalue(name)["mongo"]
            mongo.use_client(client, "demanda_test")
    yield client["demanda_test"]
    mongo.restore_client(previous)
    st.cache_resource.clear()


@pytest.fixture
def small_grids() -> Iterator[None]:
    """Usa una sola combinación de hiperparámetros por modelo (entrena en segundos)."""
    with tiny_grids():
        yield


@pytest.fixture
def tmp_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Apunta la base SQLite y los modelos de ``config`` a una carpeta temporal vacía."""
    monkeypatch.setattr(config, "STRUCTURED_DB_PATH", tmp_path / "structured.db")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    return tmp_path


@pytest.fixture(scope="session")
def trained_run(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Una corrida real entrenada una sola vez: ETL y 2 modelos en A y B.

    Usa regresión logística y Random Forest para cubrir un modelo con
    coeficientes y otro con importancias, sin pagar el costo de XGBoost, y una
    grilla chica de hiperparámetros (aquí no se prueba la búsqueda).
    Devuelve la ruta de la base SQLite, la carpeta de modelos, el MongoDB simulado
    con sus logs y resultados, y el ``run_id``.
    """
    base = tmp_path_factory.mktemp("trained")
    paths = {"db_path": base / "structured.db"}
    with isolated_logging(), isolated_mongo() as client, tiny_grids():
        run_id = pipeline.run_etl(seed=7, db_path=paths["db_path"])
        pipeline.run_training(
            run_id,
            models_dir=base / "models",
            model_names=("logistic_regression", "random_forest"),
            **paths,
        )
    return {"run_id": run_id, "models_dir": base / "models", "mongo": client, **paths}


@pytest.fixture
def use_trained(trained_run: dict, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Apunta la app a la corrida entrenada una sola vez para toda la sesión."""
    monkeypatch.setattr(config, "STRUCTURED_DB_PATH", trained_run["db_path"])
    monkeypatch.setattr(config, "MODELS_DIR", trained_run["models_dir"])
    return trained_run


@pytest.fixture(scope="session")
def trained_report_data(trained_run: dict):
    """Datos del reporte de la corrida entrenada (LR y RF en A y B), sin predicciones."""
    from src import report_data

    with using_mongo(trained_run["mongo"]):
        return report_data.collect_report_data(
            trained_run["run_id"],
            include_predictions=False,
            db_path=trained_run["db_path"],
            models_dir=trained_run["models_dir"],
        )
