"""Prueba de integración: corrida completa con logger, run_id y @log_step."""

import logging
from pathlib import Path

import mongomock
from support import mongo_logs

from src import db, logger, mongo, runs


def test_full_run_is_traceable_by_run_id(
    tmp_path: Path, mongo_db: mongomock.Database
) -> None:
    """Una corrida deja sus eventos en MongoDB, atados a su run_id en runs."""
    db_path = tmp_path / "structured.db"
    logger.setup_logging()
    db.init_structured_db(db_path)
    run_id = runs.create_run()
    token = logger.current_run_id.set(run_id)

    def paso() -> None:
        logging.getLogger("src.integracion").info("trabajando")

    paso.__module__ = "src.integracion"  # simula un paso dentro de src/
    decorated = logger.log_step(paso)

    try:
        decorated()
        runs.finish_run(run_id, "ok")
    finally:
        logger.current_run_id.reset(token)
        logger.flush_logs()

    events = mongo_logs(run_id)
    assert [e["message"] for e in events][1] == "trabajando"
    assert len(events) == 3
    assert {e["step"] for e in events} == {"paso"}
    assert runs.get_run(run_id)["status"] == "ok"
    assert mongo_db[mongo.LOGS].count_documents({"run_id": run_id}) == 3
