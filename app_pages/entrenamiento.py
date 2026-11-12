"""Página Entrenamiento: elegir escenarios y modelos, entrenar y ver métricas."""

import pandas as pd
import streamlit as st

from app_pages import state
from app_pages.labels import SCENARIO_LABELS, model_label
from src import models, pipeline, queries


def results_for_display(table: pd.DataFrame) -> pd.DataFrame:
    """Tabla de resultados con columnas y valores listos para mostrar."""
    shown = pd.DataFrame(
        {
            "Escenario": table["scenario"],
            "Modelo": table["model"].map(model_label),
            "F1 validación (promedio)": table["cv_mean"],
            "± desvío": table["cv_std"],
            "Accuracy hold-out": table["accuracy"],
            "Precision": table["precision"],
            "Recall": table["recall"],
            "F1 hold-out": table["f1"],
            "F1 «siempre alta»": table["f1_siempre_alta"],
            "F1 aleatorio": table["f1_aleatorio"],
        }
    )
    return shown


def _render_results(run_id: str) -> None:
    """Tabla resumen de métricas de una corrida ya entrenada."""
    table = queries.results_table(run_id)
    st.dataframe(
        results_for_display(table).style.format(precision=3, na_rep="—"),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Hold-out: los últimos 48 meses, que el entrenamiento no vio. «Siempre "
        "alta» y «aleatorio» son referencias sin información: un modelo solo "
        "aporta si las supera. Más detalle en la página Comparación."
    )


def _train(run_id: str, scenarios: list[str], model_names: list[str]) -> None:
    """Entrena con barra de progreso y recarga la app al terminar."""
    bar = st.progress(0.0, text="Preparando el entrenamiento…")

    def on_progress(done: int, total: int, label: str) -> None:
        bar.progress(done / total, text=f"Entrenado {label} ({done}/{total})")

    try:
        pipeline.run_training(
            run_id,
            scenarios=tuple(scenarios),
            model_names=tuple(model_names),
            on_progress=on_progress,
        )
    except Exception as error:  # noqa: BLE001 - se muestra el error en pantalla
        bar.empty()
        st.error(f"El entrenamiento falló: {error}. El detalle está en la página Logs.")
        return
    state.flash(
        f"Entrenamiento terminado: resultados guardados en la corrida {run_id}."
    )
    st.rerun()


def _render_form(run_id: str, seed: int) -> None:
    """Elección de escenarios y modelos y botón para entrenar."""
    scenarios = st.multiselect(
        "Escenarios",
        list(SCENARIO_LABELS),
        default=list(SCENARIO_LABELS),
        format_func=SCENARIO_LABELS.get,
        help="A es el principal: solo usa lo que se conoce antes de que termine "
        "el mes. B suma la potencia máxima, que ocurre en el mismo mes.",
    )
    model_names = st.multiselect(
        "Modelos",
        list(models.MODEL_NAMES),
        default=list(models.MODEL_NAMES),
        format_func=model_label,
    )
    st.caption(
        f"Semilla de esta corrida: **{seed}** (se fijó al ejecutar el ETL). "
        "Los tres modelos usan los mismos cortes de validación temporal."
    )
    total = len(scenarios) * len(model_names)
    if st.button("Entrenar", type="primary", disabled=total == 0):
        _train(run_id, scenarios, model_names)


def render() -> None:
    """Dibuja la página Entrenamiento."""
    st.header("Entrenamiento")
    run_id = state.current_run_id()
    summary = queries.run_summary(run_id) if run_id else None
    if summary is None or not summary["steps"][queries.STEP_ETL]:
        st.warning("Primero ejecutá el ETL en la página Datos.")
        return
    st.markdown(f"Corrida **`{run_id}`**.")
    if summary["steps"][queries.STEP_TRAINING]:
        st.info(
            "Esta corrida ya está entrenada. Para entrenar con otras opciones "
            "iniciá una corrida nueva ejecutando el ETL en la página Datos."
        )
        _render_results(run_id)
        return
    _render_form(run_id, summary["seed"])
