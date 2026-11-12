"""Fixtures compartidas."""

import logging
from collections.abc import Iterator

import pytest

from src import logger, pipeline


@pytest.fixture
def clean_root_logger() -> Iterator[None]:
    """Deja el logger raíz del paquete como estaba al terminar el test."""
    root = logging.getLogger(logger.ROOT_LOGGER_NAME)
    old_level, old_handlers = root.level, list(root.handlers)
    yield
    root.setLevel(old_level)
    root.handlers[:] = old_handlers


@pytest.fixture(scope="session")
def trained_run(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Una corrida real entrenada una sola vez: ETL y 2 modelos en A y B.

    Usa regresión logística y Random Forest para cubrir un modelo con
    coeficientes y otro con importancias, sin pagar el costo de XGBoost.
    Devuelve las rutas de las bases y de los modelos y el ``run_id``.
    """
    base = tmp_path_factory.mktemp("trained")
    paths = {
        "db_path": base / "structured.db",
        "unstructured_path": base / "unstructured.db",
    }
    root = logging.getLogger(logger.ROOT_LOGGER_NAME)
    old_level, old_handlers = root.level, list(root.handlers)
    try:
        run_id = pipeline.run_etl(seed=7, db_path=paths["db_path"])
        pipeline.run_training(
            run_id,
            models_dir=base / "models",
            model_names=("logistic_regression", "random_forest"),
            **paths,
        )
    finally:
        root.setLevel(old_level)
        root.handlers[:] = old_handlers
    return {"run_id": run_id, "models_dir": base / "models", **paths}
