"""Prueba de integración: corrida completa con logger, run_id y @log_step."""

import logging
from pathlib import Path

from src import db, logger


def test_full_run_is_traceable_by_run_id(tmp_path: Path) -> None:
    """Una corrida deja sus eventos en logs, atados a su run_id en runs."""
    db_path = tmp_path / "structured.db"
    logger.setup_logging(db_path)
    db.init_structured_db(db_path)
    run_id = db.create_run(db_path=db_path)
    token = logger.current_run_id.set(run_id)

    def paso() -> None:
        logging.getLogger("src.integracion").info("trabajando")

    paso.__module__ = "src.integracion"  # simula un paso dentro de src/
    decorated = logger.log_step(paso)

    try:
        decorated()
        db.finish_run(run_id, "ok", db_path)
    finally:
        logger.current_run_id.reset(token)

    with db.get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT l.message, r.status FROM logs l "
            "JOIN runs r ON r.run_id = l.run_id WHERE l.run_id = ? ORDER BY l.id",
            (run_id,),
        ).fetchall()
    assert [r["message"] for r in rows][1] == "trabajando"
    assert len(rows) == 3
    assert {r["status"] for r in rows} == {"ok"}
