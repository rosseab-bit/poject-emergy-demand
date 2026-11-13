"""Interfaz de Streamlit: solo pantalla; la lógica vive en ``src/``.

Ejecutar con ``streamlit run app.py``.
"""

import streamlit as st

from app_pages import PAGES, sidebar, state
from src import db

st.set_page_config(page_title="Demanda eléctrica", page_icon="⚡", layout="wide")
db.init_databases()  # en el primer arranque de un clon crea las bases vacías
st.title("Demanda eléctrica mensual: ¿demanda alta?")
st.caption(
    "Clasificación a nivel mensual de la demanda eléctrica respecto de lo "
    "normal para su época."
)

page = sidebar.render()
state.show_flash()
PAGES[page]()
