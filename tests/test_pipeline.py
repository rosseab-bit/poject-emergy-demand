"""Tests de src/pipeline.py."""

from pathlib import Path

import pytest

from src import data, db, pipeline


@pytest.mark.usefixtures("clean_root_logger")
def test_run_etl_end_to_end(tmp_path: Path) -> None:
    """El ETL deja 308 filas (259 completas) y todo logueado con su run_id."""
    db_path = tmp_path / "structured.db"
    run_id = pipeline.run_etl(db_path=db_path)

    clean = data.load_clean_data(db_path)
    assert len(clean) == 308
    assert data.count_complete_rows(clean) == 259

    run = db.get_run(run_id, db_path)
    assert run["status"] == "ok"
    assert run["finished_at"] is not None

    with db.get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT level, module, message FROM logs WHERE run_id = ?", (run_id,)
        ).fetchall()
    messages = " | ".join(r["message"] for r in rows)
    for expected in (
        "Inicio: load_raw",
        "CSV leído: 308",
        "repetidos: 2",
        "datos_limpios guardada: 308",
        "Fin: build_dataset",
        "Balance de clases: 236 meses con target, 60 positivos (25.4 %)",
        "Fin: run_etl_steps",
    ):
        assert expected in messages
    assert sum(r["level"] == "WARNING" for r in rows) == 2
    assert not any(r["level"] == "ERROR" for r in rows)
    with db.get_connection(db_path) as conn:
        orphans = conn.execute("SELECT COUNT(*) FROM logs WHERE run_id IS NULL")
        assert orphans.fetchone()[0] == 0


@pytest.mark.usefixtures("clean_root_logger")
def test_run_etl_failure_is_logged_and_run_marked_error(tmp_path: Path) -> None:
    """Si el CSV es inválido, la corrida queda en error y el error en logs."""
    db_path = tmp_path / "structured.db"
    bad_csv = tmp_path / "malo.csv"
    bad_csv.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="Faltan columnas"):
        pipeline.run_etl(csv_path=bad_csv, db_path=db_path)

    with db.get_connection(db_path) as conn:
        run = conn.execute("SELECT run_id, status FROM runs").fetchone()
        errors = conn.execute(
            "SELECT message FROM logs WHERE level = 'ERROR' AND run_id = ?",
            (run["run_id"],),
        ).fetchall()
    assert run["status"] == "error"
    assert len(errors) >= 1
