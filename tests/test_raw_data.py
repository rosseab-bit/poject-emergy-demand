"""Tests de ``datos_originales``: el CSV completo guardado sin modificar."""

import csv
import sqlite3
from pathlib import Path

import pytest

from src import config, data, db, pipeline


def csv_rows() -> list[dict[str, str]]:
    """Las filas del CSV original tal como están en el archivo."""
    with config.RAW_CSV_PATH.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def test_raw_columns_match_the_csv_header() -> None:
    """La lista de columnas de la configuración es la cabecera real del CSV."""
    header = config.RAW_CSV_PATH.read_text(encoding="utf-8").splitlines()[0]
    assert tuple(header.split(",")) == config.RAW_COLUMNS
    assert len(config.RAW_COLUMNS) == 30


def test_init_structured_db_creates_the_raw_table_empty(tmp_path: Path) -> None:
    """La tabla existe desde el arranque, con las 30 columnas y sin filas."""
    path = tmp_path / "structured.db"
    db.init_structured_db(path)
    with db.get_connection(path) as conn:
        columns = [
            r["name"] for r in conn.execute("PRAGMA table_info(datos_originales)")
        ]
        count = conn.execute("SELECT COUNT(*) FROM datos_originales").fetchone()[0]
    assert columns == list(config.RAW_COLUMNS)
    assert count == 0


def test_save_raw_data_stores_the_whole_csv_unchanged(tmp_path: Path) -> None:
    """Quedan las 308 filas y las 30 columnas, con los mismos valores del archivo."""
    path = tmp_path / "structured.db"
    assert data.save_raw_data(db_path=path) == 308
    expected = csv_rows()
    with db.get_connection(path) as conn:
        stored = conn.execute("SELECT * FROM datos_originales ORDER BY indice_tiempo")
        rows = [dict(row) for row in stored]
    assert len(rows) == len(expected) == 308
    for row, original in zip(rows, expected, strict=True):
        assert row["indice_tiempo"] == original["indice_tiempo"]
        for column in config.RAW_COLUMNS[1:]:
            text = original[column]
            if text == "":
                assert row[column] is None, column
            else:
                assert row[column] == float(text), column


def test_save_raw_data_does_not_modify_the_csv(tmp_path: Path) -> None:
    """El archivo original queda exactamente igual."""
    before = config.RAW_CSV_PATH.read_bytes()
    data.save_raw_data(db_path=tmp_path / "structured.db")
    assert config.RAW_CSV_PATH.read_bytes() == before


def test_save_raw_data_replaces_the_previous_content(tmp_path: Path) -> None:
    """Correrlo dos veces deja el mismo resultado, sin filas duplicadas."""
    path = tmp_path / "structured.db"
    data.save_raw_data(db_path=path)
    data.save_raw_data(db_path=path)
    with db.get_connection(path) as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM datos_originales").fetchone()[0] == 308
        )


def test_save_raw_data_rejects_a_csv_with_other_columns(tmp_path: Path) -> None:
    """Un CSV con columnas distintas se rechaza y dice cuáles faltan y sobran."""
    bad = tmp_path / "otro.csv"
    bad.write_text("indice_tiempo,demanda_total,columna_rara\n2001-01-01,1,2\n")
    with pytest.raises(ValueError, match="columna_rara") as error:
        data.save_raw_data(bad, tmp_path / "structured.db")
    assert "Faltan" in str(error.value) and "potencia_maxima" in str(error.value)


def test_the_etl_fills_both_datasets(tmp_path: Path) -> None:
    """El ETL deja el original completo y el limpio, y lo registra en logs."""
    path = tmp_path / "structured.db"
    run_id = pipeline.run_etl(db_path=path)
    with db.get_connection(path) as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM datos_originales").fetchone()[0] == 308
        )
        assert conn.execute("SELECT COUNT(*) FROM datos_limpios").fetchone()[0] == 308
        messages = [
            r[0]
            for r in conn.execute(
                "SELECT message FROM logs WHERE run_id = ?", (run_id,)
            )
        ]
    assert any(
        "datos_originales guardada: 308 filas, 30 columnas" in m for m in messages
    )


def test_raw_and_clean_agree_on_the_shared_columns(tmp_path: Path) -> None:
    """Lo limpio parte de lo original: la demanda total es la misma en ambas."""
    path = tmp_path / "structured.db"
    pipeline.run_etl(db_path=path)
    conn = sqlite3.connect(path)
    try:
        diff = conn.execute(
            "SELECT COUNT(*) FROM datos_originales o JOIN datos_limpios l "
            "ON l.fecha = o.indice_tiempo WHERE l.demanda_total != o.demanda_total"
        ).fetchone()[0]
    finally:
        conn.close()
    assert diff == 0
