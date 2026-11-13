"""Tests de las corridas guardadas en la colección ``runs`` de MongoDB."""

import mongomock
import pytest

from src import config, mongo, queries, runs


def test_create_run_starts_running(mongo_db: mongomock.Database) -> None:
    """create_run guarda la corrida en estado running con la semilla."""
    run_id = runs.create_run(seed=7, notes="prueba")
    run = runs.get_run(run_id)
    assert run is not None
    assert run["status"] == "running"
    assert run["seed"] == 7
    assert run["notes"] == "prueba"
    assert run["finished_at"] is None
    assert run["started_at"].endswith("+00:00")
    assert mongo_db[mongo.RUNS].count_documents({"run_id": run_id}) == 1


def test_create_run_default_seed_from_config() -> None:
    """Sin semilla explícita se usa la de config; la semilla 0 también es válida."""
    assert runs.get_run(runs.create_run())["seed"] == config.SEED
    assert runs.get_run(runs.create_run(seed=0))["seed"] == 0


def test_run_ids_are_short_and_unique() -> None:
    """Los run_id tienen 8 caracteres hex y no se repiten."""
    ids = {runs.new_run_id() for _ in range(100)}
    assert len(ids) == 100
    assert all(len(i) == 8 and int(i, 16) >= 0 for i in ids)


def test_finish_run_sets_status() -> None:
    """finish_run guarda estado final y fecha de cierre."""
    run_id = runs.create_run()
    runs.finish_run(run_id, runs.STATUS_OK)
    run = runs.get_run(run_id)
    assert run["status"] == "ok"
    assert run["finished_at"] is not None
    runs.finish_run(run_id, runs.STATUS_ERROR)  # se puede corregir el estado
    assert runs.get_run(run_id)["status"] == "error"


def test_finish_run_rejects_bad_input() -> None:
    """Estado inválido o run_id inexistente lanzan ValueError."""
    run_id = runs.create_run()
    with pytest.raises(ValueError, match="Estado inválido"):
        runs.finish_run(run_id, "raro")
    with pytest.raises(ValueError, match="inexistente"):
        runs.finish_run("noexiste", "ok")
    assert runs.get_run(run_id)["status"] == "running"  # no se tocó


def test_get_run_missing_returns_none() -> None:
    """get_run devuelve None si no existe, y no expone el _id interno de Mongo."""
    assert runs.get_run("noexiste") is None
    assert "_id" not in runs.get_run(runs.create_run())


def test_run_documents_have_a_unique_run_id(mongo_db: mongomock.Database) -> None:
    """El índice único de ``init_mongo`` impide dos corridas con el mismo run_id."""
    mongo.init_mongo()
    run_id = runs.create_run()
    with pytest.raises(Exception, match="duplicate|E11000"):
        mongo_db[mongo.RUNS].insert_one({"run_id": run_id})


def test_list_runs_newest_first_with_ties_in_insertion_order() -> None:
    """La corrida más reciente primero, incluso si empezaron el mismo segundo."""
    first = runs.create_run(seed=1)
    second = runs.create_run(seed=2)
    third = runs.create_run(seed=3)
    listed = queries.list_runs()
    assert listed["run_id"].tolist() == [third, second, first]
    assert list(listed.columns) == queries.RUN_COLUMNS


def test_list_runs_empty_returns_an_empty_frame_with_columns() -> None:
    """Sin corridas devuelve un DataFrame vacío con las columnas."""
    empty = queries.list_runs()
    assert empty.empty and list(empty.columns) == queries.RUN_COLUMNS
