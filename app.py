"""Interfaz de Streamlit: solo pantalla; la lógica vive en ``src/``.

Ejecutar con ``streamlit run app.py``.
"""

import streamlit as st

from app_pages import PAGES, sidebar, state
from src import db, mongo


@st.cache_resource(show_spinner="Conectando con MongoDB…")
def init_mongo_once() -> bool:
    """Prepara las colecciones de MongoDB una vez por proceso, no en cada clic."""
    mongo.init_mongo()
    return True


st.set_page_config(page_title="Demanda eléctrica", page_icon="⚡", layout="wide")
db.init_databases()  # en el primer arranque de un clon crea las bases vacías
try:
    init_mongo_once()
except (mongo.MongoConfigError, mongo.MongoUnavailableError) as error:
    st.error(f"No se pudo preparar MongoDB. {error}")
    st.info("Revisá el archivo .env (hay una plantilla en .env.example).")
    st.stop()
st.title("Demanda eléctrica mensual: ¿demanda alta?")
st.caption(
    "Clasificación a nivel mensual de la demanda eléctrica respecto de lo "
    "normal para su época."
)

page = sidebar.render()
state.show_flash()
PAGES[page]()
