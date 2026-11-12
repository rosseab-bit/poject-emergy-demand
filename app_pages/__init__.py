"""Páginas de la interfaz de Streamlit."""

from collections.abc import Callable

from app_pages import comparacion, datos, entrenamiento, logs, prediccion, reporte
from app_pages.sidebar import PAGE_NAMES

PAGES: dict[str, Callable[[], None]] = {
    "Datos": datos.render,
    "Entrenamiento": entrenamiento.render,
    "Comparación": comparacion.render,
    "Predicción": prediccion.render,
    "Reporte": reporte.render,
    "Logs": logs.render,
}
assert tuple(PAGES) == PAGE_NAMES
