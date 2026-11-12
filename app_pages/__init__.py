"""Páginas de la interfaz de Streamlit."""

from collections.abc import Callable

import streamlit as st

from app_pages import comparacion, datos, entrenamiento, logs
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
    "Comparación": comparacion.render,
    "Predicción": _placeholder("Predicción", "Disponible en una etapa posterior."),
    "Reporte": _placeholder("Reporte", "Disponible en una etapa posterior."),
    "Logs": logs.render,
}
assert tuple(PAGES) == PAGE_NAMES
