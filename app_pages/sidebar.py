"""Barra lateral: navegación, corrida actual, semilla y estado de los pasos."""

import pandas as pd
import streamlit as st

from app_pages import state
from app_pages.tables import STATUS_ICONS
from src import config, queries

PAGE_NAMES = (
    "Datos",
    "Entrenamiento",
    "Comparación",
    "Predicción",
    "Reporte",
    "Logs",
)


def _run_label(runs: pd.DataFrame, run_id: str) -> str:
    """Texto del selector: ``run_id · estado · fecha``."""
    row = runs.set_index("run_id").loc[run_id]
    icon = STATUS_ICONS.get(row["status"], "")
    return f"{run_id} · {icon} {row['status']} · {row['started_at']}"


def _render_run_selector(runs: pd.DataFrame) -> str | None:
    """Muestra el selector de corridas y devuelve el ``run_id`` elegido."""
    if runs.empty:
        st.sidebar.info("Todavía no hay corridas. Ejecutá el ETL en la página Datos.")
        return None
    options = runs["run_id"].tolist()
    if st.session_state.get(state.RUN_KEY) not in options:
        st.session_state[state.RUN_KEY] = options[0]
    return st.sidebar.selectbox(
        "Corrida (abrir una anterior)",
        options,
        format_func=lambda run_id: _run_label(runs, run_id),
        key=state.RUN_KEY,
    )


def _render_summary(run_id: str) -> None:
    """Muestra estado, semilla, hash del dataset y pasos de la corrida."""
    summary = queries.run_summary(run_id)
    if summary is None:
        return
    st.sidebar.markdown(f"**run_id:** `{run_id}`")
    st.sidebar.markdown(f"**Estado:** {summary['status']}")
    st.sidebar.markdown(f"**Semilla de esta corrida:** {summary['seed']}")
    digest = summary["dataset_sha256"]
    st.sidebar.markdown(
        f"**Hash del dataset:** `{digest[:16]}…`"
        if digest
        else "**Hash del dataset:** —"
    )
    st.sidebar.markdown("**Pasos**")
    for step, done in summary["steps"].items():
        st.sidebar.markdown(
            f"{'✅' if done else '⬜'} {step} — {'hecho' if done else 'pendiente'}"
        )


def render() -> str:
    """Dibuja la barra lateral y devuelve el nombre de la página elegida."""
    state.apply_pending_run()
    page = st.sidebar.radio("Página", PAGE_NAMES, key=state.PAGE_KEY)
    st.sidebar.divider()
    st.sidebar.subheader("Corrida actual")
    run_id = _render_run_selector(queries.list_runs())
    if run_id is not None:
        _render_summary(run_id)
    st.sidebar.number_input(
        "Semilla para la próxima corrida",
        min_value=0,
        max_value=2**31 - 1,
        value=config.SEED,
        step=1,
        key=state.SEED_KEY,
        help="Se usa al ejecutar el ETL y crear una corrida nueva. "
        "Por defecto es la de config.py.",
    )
    return page
