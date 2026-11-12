"""Tests mínimos de src/config.py."""

from src import config


def test_seed_is_int() -> None:
    """La semilla es un entero."""
    assert isinstance(config.SEED, int)


def test_raw_csv_exists() -> None:
    """La ruta al CSV original apunta a un archivo real."""
    assert config.RAW_CSV_PATH.is_file()


def test_project_dirs_exist() -> None:
    """Las carpetas del proyecto existen."""
    for path in (config.DB_DIR, config.MODELS_DIR, config.REPORTS_DIR):
        assert path.is_dir()
