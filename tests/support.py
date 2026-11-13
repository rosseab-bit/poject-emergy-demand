"""Piezas compartidas por los tests.

Reúne lo que antes estaba repetido en varios archivos: el aislamiento del
logger, las grillas de hiperparámetros chicas para tests que no prueban la
búsqueda, las ayudas para manejar la app con ``AppTest`` y el dataset real.
"""

import logging
import shutil
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src import config, data, features, logger, models

APP_PATH = config.ROOT_DIR / "app.py"

# Una sola combinación por modelo: entrena en segundos. Sirve para los tests que
# necesitan modelos entrenados pero no prueban la búsqueda de hiperparámetros.
TINY_GRIDS = {
    models.LOGISTIC_REGRESSION: {"clf__C": [0.1]},
    models.RANDOM_FOREST: {"clf__max_depth": [3], "clf__min_samples_leaf": [3]},
    models.XGBOOST: {
        "clf__n_estimators": [50],
        "clf__max_depth": [2],
        "clf__learning_rate": [0.1],
    },
}


@contextmanager
def isolated_logging() -> Iterator[None]:
    """Deja el logger del paquete como estaba al salir del bloque.

    Lo usan las fixtures de módulo y de sesión, que corren fuera del alcance de
    la fixture automática de cada test.
    """
    root = logging.getLogger(logger.ROOT_LOGGER_NAME)
    saved_level, saved_handlers = root.level, list(root.handlers)
    try:
        yield
    finally:
        root.setLevel(saved_level)
        root.handlers[:] = saved_handlers


@contextmanager
def tiny_grids() -> Iterator[None]:
    """Reemplaza las grillas de hiperparámetros por una combinación por modelo."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(models, "PARAM_GRIDS", TINY_GRIDS)
        yield


def real_dataset() -> pd.DataFrame:
    """Dataset real con todas las variables derivadas y el target."""
    return features.build_dataset(data.flag_repeated_power(data.load_raw()))


def start_app(timeout: int = 120) -> AppTest:
    """Ejecuta ``app.py`` con AppTest y devuelve la sesión."""
    return AppTest.from_file(str(APP_PATH), default_timeout=timeout).run()


def go_to(at: AppTest, page: str) -> AppTest:
    """Navega a una página de la barra lateral."""
    return at.sidebar.radio[0].set_value(page).run()


def open_page(page: str, timeout: int = 120) -> AppTest:
    """Abre la app directamente en una página."""
    return go_to(start_app(timeout), page)


def sidebar_text(at: AppTest) -> str:
    """Todo el texto markdown de la barra lateral, en un solo string."""
    return "\n".join(m.value for m in at.sidebar.markdown)


def main_file_only(db_path: Path, dest_dir: Path) -> Path:
    """Copia solo el archivo ``.db`` principal (sin su ``-wal``) a otra carpeta.

    Es lo que se entrega o se copia para revisar una base: si el volcado del WAL
    no se hizo, a la copia le faltan los datos recientes.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    return Path(shutil.copy(db_path, dest_dir / db_path.name))


def count_rows(path: Path, table: str) -> int | None:
    """Filas de una tabla en un archivo ``.db``; ``None`` si la tabla no existe."""
    conn = sqlite3.connect(path)
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    except sqlite3.OperationalError:
        return None
    finally:
        conn.close()


def user_tables(db_path: Path) -> dict[str, int]:
    """Tablas propias de una base (sin las internas de SQLite) con su cantidad de filas."""
    conn = sqlite3.connect(db_path)
    try:
        names = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        return {
            n: conn.execute(f"SELECT COUNT(*) FROM {n}").fetchone()[0] for n in names
        }
    finally:
        conn.close()


STRUCTURED_TABLES = {"datos_limpios", "datos_originales", "logs", "runs"}
UNSTRUCTURED_TABLES = {"configuracion", "predicciones", "resultados_modelos"}
