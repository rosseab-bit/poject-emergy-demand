"""Página Logs: eventos registrados en ``structured.db`` y lista de corridas."""

import streamlit as st

from app_pages import state
from app_pages.tables import highlight_row, logs_for_display, runs_for_display
from src import queries

SCOPE_CURRENT = "Corrida actual"
SCOPE_ALL = "Todas las corridas"
LEVELS = ["INFO", "WARNING", "ERROR"]


def _render_filters(run_id: str | None) -> tuple[str | None, list[str], list[str]]:
    """Filtros de la página: alcance, niveles y módulos (vacío = todos)."""
    scope = SCOPE_ALL
    if run_id is not None:
        scope = st.radio("Mostrar", [SCOPE_CURRENT, SCOPE_ALL], horizontal=True)
    selected_run = run_id if scope == SCOPE_CURRENT else None
    first, second = st.columns(2)
    levels = first.multiselect("Nivel", LEVELS, help="Vacío = todos los niveles.")
    modules = second.multiselect(
        "Módulo",
        queries.list_log_modules(selected_run),
        help="Vacío = todos los módulos.",
    )
    return selected_run, levels, modules


def _render_events(
    selected_run: str | None, levels: list[str], modules: list[str]
) -> None:
    """Contadores y tabla de eventos, con errores y advertencias resaltados."""
    logs = queries.get_logs(selected_run, levels or None, modules or None)
    if logs.empty:
        st.info("No hay eventos con esos filtros.")
        return
    counts = logs["level"].value_counts()
    total, warnings, errors = st.columns(3)
    total.metric("Eventos", len(logs))
    warnings.metric("Advertencias", int(counts.get("WARNING", 0)))
    errors.metric("Errores", int(counts.get("ERROR", 0)))
    st.dataframe(
        logs_for_display(logs).style.apply(highlight_row, axis=1),
        hide_index=True,
        width="stretch",
        column_config={"Mensaje": st.column_config.TextColumn(width="large")},
    )


def _render_runs() -> None:
    """Lista de corridas con su estado."""
    st.subheader("Corridas")
    runs = queries.list_runs()
    if runs.empty:
        st.info("Todavía no hay corridas.")
        return
    st.dataframe(
        runs_for_display(runs).style.apply(highlight_row, axis=1),
        hide_index=True,
        width="stretch",
    )


def render() -> None:
    """Dibuja la página Logs."""
    st.header("Logs")
    st.markdown(
        "Todo lo que hace el código queda registrado en la tabla `logs` de "
        "`structured.db`, atado al `run_id` de cada corrida."
    )
    selected_run, levels, modules = _render_filters(state.current_run_id())
    _render_events(selected_run, levels, modules)
    _render_runs()
