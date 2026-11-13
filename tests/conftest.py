"""Fixtures compartidas y reglas de colección de los tests."""

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest
from support import isolated_logging, tiny_grids

from src import config, logger, pipeline

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


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Marca como ``slow`` lo que entrena modelos o recorre la app completa."""
    for item in items:
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


@pytest.fixture
def small_grids() -> Iterator[None]:
    """Usa una sola combinación de hiperparámetros por modelo (entrena en segundos)."""
    with tiny_grids():
        yield


@pytest.fixture
def tmp_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Apunta las bases y los modelos de ``config`` a una carpeta temporal vacía."""
    monkeypatch.setattr(config, "STRUCTURED_DB_PATH", tmp_path / "structured.db")
    monkeypatch.setattr(config, "UNSTRUCTURED_DB_PATH", tmp_path / "unstructured.db")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    return tmp_path


@pytest.fixture(scope="session")
def trained_run(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Una corrida real entrenada una sola vez: ETL y 2 modelos en A y B.

    Usa regresión logística y Random Forest para cubrir un modelo con
    coeficientes y otro con importancias, sin pagar el costo de XGBoost, y una
    grilla chica de hiperparámetros (aquí no se prueba la búsqueda).
    Devuelve las rutas de las bases y de los modelos y el ``run_id``.
    """
    base = tmp_path_factory.mktemp("trained")
    paths = {
        "db_path": base / "structured.db",
        "unstructured_path": base / "unstructured.db",
    }
    with isolated_logging(), tiny_grids():
        run_id = pipeline.run_etl(seed=7, db_path=paths["db_path"])
        pipeline.run_training(
            run_id,
            models_dir=base / "models",
            model_names=("logistic_regression", "random_forest"),
            **paths,
        )
    return {"run_id": run_id, "models_dir": base / "models", **paths}


@pytest.fixture
def use_trained(trained_run: dict, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Apunta la app a la corrida entrenada una sola vez para toda la sesión."""
    monkeypatch.setattr(config, "STRUCTURED_DB_PATH", trained_run["db_path"])
    monkeypatch.setattr(
        config, "UNSTRUCTURED_DB_PATH", trained_run["unstructured_path"]
    )
    monkeypatch.setattr(config, "MODELS_DIR", trained_run["models_dir"])
    return trained_run


@pytest.fixture(scope="session")
def trained_report_data(trained_run: dict):
    """Datos del reporte de la corrida entrenada (LR y RF en A y B), sin predicciones."""
    from src import report_data

    return report_data.collect_report_data(
        trained_run["run_id"],
        include_predictions=False,
        db_path=trained_run["db_path"],
        unstructured_path=trained_run["unstructured_path"],
        models_dir=trained_run["models_dir"],
    )
