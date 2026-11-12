"""Páginas de la interfaz de Streamlit."""

from collections.abc import Callable

import streamlit as st

from app_pages import datos, entrenamiento
from app_pages.sidebar import PAGE_NAMES


def _placeholder(name: str, detail: str) -> Callable[[], None]:
    """Crea una página provisoria que explica cuándo estará disponible."""

    def render() -> None:
        st.header(name)
        st.info(detail)

    return render


PAGES: dict[str, Callable[[], None]] = {
    "Datos": datos.render,
    "Entrenamiento": entrenamiento.render,
    "Comparación": _placeholder("Comparación", "Página en construcción."),
    "Predicción": _placeholder("Predicción", "Disponible en una etapa posterior."),
    "Reporte": _placeholder("Reporte", "Disponible en una etapa posterior."),
    "Logs": _placeholder("Logs", "Página en construcción."),
}
assert tuple(PAGES) == PAGE_NAMES
