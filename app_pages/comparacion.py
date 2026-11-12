"""Página Comparación: métricas, matrices, variables y escenarios A contra B."""

import pandas as pd
import streamlit as st

from app_pages import state
from app_pages.labels import SCENARIO_LABELS, model_label
from app_pages.tables import results_for_display
from src import queries

REFERENCE_LABEL = "«siempre alta»"


def f1_chart_spec(reference: float) -> dict:
    """Barras del F1 de hold-out por modelo, un panel por escenario.

    Una sola serie por panel (sin leyenda) y una línea punteada con la
    referencia «siempre alta».
    """
    return {
        "facet": {"column": {"field": "Escenario", "type": "nominal", "title": None}},
        "spec": {
            "height": 220,
            "layer": [
                {
                    "mark": {"type": "bar", "cornerRadiusEnd": 4},
                    "encoding": {
                        "x": {
                            "field": "Modelo",
                            "type": "nominal",
                            "sort": None,
                            "title": None,
                            "axis": {"labelAngle": -30},
                        },
                        "y": {
                            "field": "F1 hold-out",
                            "type": "quantitative",
                            "scale": {"domain": [0, 1]},
                            "title": "F1 en el hold-out",
                        },
                        "tooltip": [
                            {"field": "Modelo"},
                            {"field": "F1 hold-out", "format": ".3f"},
                            {"field": "F1 validación (promedio)", "format": ".3f"},
                        ],
                    },
                },
                {
                    "mark": {"type": "rule", "strokeDash": [5, 4], "strokeWidth": 1.5},
                    "encoding": {"y": {"datum": reference}},
                },
                {
                    "mark": {"type": "text", "align": "left", "dy": -6, "dx": 2},
                    "encoding": {
                        "y": {"datum": reference},
                        "x": {"value": 0},
                        "text": {"value": f"{REFERENCE_LABEL} {reference:.2f}"},
                    },
                },
            ],
        },
    }


def _render_notice(noise: dict) -> None:
    """Aviso honesto sobre el ruido de las métricas; no se declara ganador."""
    st.warning(
        f"**Todavía no se declara un modelo ganador.** El hold-out tiene "
        f"{noise['n_test']} meses y solo {noise['positives']} son de demanda alta; "
        f"el F1 de validación varía hasta ±{noise['max_cv_std']:.2f} entre cortes. "
        f"Una diferencia menor que ese desvío puede ser ruido. Además, marcar "
        f"todos los meses como «alta» da F1 = {noise['always_high_f1']:.2f}: un "
        "modelo solo aporta si supera esa referencia."
    )


def _render_chart(table: pd.DataFrame) -> None:
    """Gráfico de barras del F1 de hold-out con la referencia."""
    shown = results_for_display(table)
    shown["Escenario"] = shown["Escenario"].map(SCENARIO_LABELS)
    reference = float(table["f1_siempre_alta"].iloc[0])
    st.vega_lite_chart(shown, f1_chart_spec(reference), width="stretch")


def _confusion_frame(matrix: list[list[int]]) -> pd.DataFrame:
    """Matriz de confusión con etiquetas de filas (real) y columnas (predicho)."""
    return pd.DataFrame(
        matrix,
        index=["Real: normal", "Real: alta"],
        columns=["Predicho: normal", "Predicho: alta"],
    )


def _render_confusion(docs: list[dict]) -> None:
    """Matrices de confusión del hold-out, una por modelo y escenario."""
    st.subheader("Matrices de confusión (hold-out)")
    st.caption(
        "Cada fila es lo que pasó de verdad; cada columna, lo que predijo el modelo."
    )
    for scenario in SCENARIO_LABELS:
        in_scenario = [d for d in docs if d["scenario"] == scenario]
        if not in_scenario:
            continue
        st.markdown(f"**Escenario {SCENARIO_LABELS[scenario]}**")
        for column, doc in zip(st.columns(len(in_scenario)), in_scenario, strict=True):
            column.markdown(f"*{model_label(doc['model'])}*")
            column.dataframe(_confusion_frame(doc["holdout"]["confusion_matrix"]))


def _render_importance(run_id: str, docs: list[dict]) -> None:
    """Coeficientes o importancias de variables de un modelo a elección."""
    st.subheader("Variables que usa cada modelo")
    options = [(d["scenario"], d["model"]) for d in docs]
    scenario, model = st.selectbox(
        "Modelo y escenario",
        options,
        format_func=lambda option: f"{option[0]} · {model_label(option[1])}",
    )
    table = queries.feature_importance_for(run_id, scenario, model)
    kind = table["tipo"].iloc[0]
    st.bar_chart(
        table.set_index("feature")["importancia"],
        horizontal=True,
        y_label=None,
        x_label=kind.capitalize(),
    )
    if kind == "coeficiente":
        st.caption(
            "Coeficientes sobre variables estandarizadas: positivo empuja hacia "
            "«demanda alta», negativo hacia «normal»."
        )
    else:
        st.caption(
            "Importancias (suman 1): cuánto usa el modelo cada variable; sin signo."
        )


def _render_scenarios(table: pd.DataFrame) -> None:
    """Comparación del escenario B (con potencia máxima) contra el A."""
    comparison = queries.scenario_comparison(table)
    if comparison.empty:
        return
    st.subheader("Escenario A contra B: ¿cuánto aporta la potencia máxima?")
    shown = pd.DataFrame(
        {
            "Modelo": comparison["model"].map(model_label),
            "F1 hold-out A": comparison["f1_A"],
            "F1 hold-out B": comparison["f1_B"],
            "Diferencia (B − A)": comparison["dif_f1"],
            "F1 validación A": comparison["cv_A"],
            "F1 validación B": comparison["cv_B"],
            "Diferencia validación (B − A)": comparison["dif_cv"],
        }
    )
    st.dataframe(shown.style.format(precision=3), hide_index=True, width="stretch")
    st.caption(
        "La potencia máxima ocurre en el mismo mes que la demanda, así que no se "
        "conoce de antemano: B sirve para diagnosticar, no para anticipar."
    )


def render() -> None:
    """Dibuja la página Comparación."""
    st.header("Comparación")
    run_id = state.current_run_id()
    table = queries.results_table(run_id) if run_id else None
    if table is None or table.empty:
        st.warning("Primero entrená los modelos en la página Entrenamiento.")
        return
    docs = queries.result_docs(run_id)
    st.markdown(f"Corrida **`{run_id}`**.")
    _render_notice(queries.noise_summary(table, docs))
    st.subheader("Métricas")
    st.dataframe(
        results_for_display(table).style.format(precision=3, na_rep="—"),
        hide_index=True,
        width="stretch",
    )
    _render_chart(table)
    _render_confusion(docs)
    _render_importance(run_id, docs)
    _render_scenarios(table)
