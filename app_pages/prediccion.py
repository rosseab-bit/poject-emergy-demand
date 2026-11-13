"""Página Predicción: el mes siguiente al último dato, con los modelos guardados."""

import pandas as pd
import streamlit as st

from app_pages import state
from app_pages.labels import MONTH_NAMES, model_label
from app_pages.tables import (
    history_for_display,
    lags_for_display,
    prediction_for_display,
)
from src import data, predict, queries

HISTORY_SCOPE_RUN = "Corrida actual"
HISTORY_SCOPE_ALL = "Todas las corridas"


def _ready(run_id: str | None) -> bool:
    """Verifica que la corrida esté entrenada en el escenario A; avisa si no."""
    summary = queries.run_summary(run_id) if run_id else None
    if summary is None or not summary["steps"][queries.STEP_TRAINING]:
        st.warning("Primero entrená los modelos en la página Entrenamiento.")
        return False
    if not (queries.results_table(run_id)["scenario"] == "A").any():
        st.warning(
            "Esta corrida no entrenó el escenario A (anticipación), que es el único "
            "que sirve para predecir. Iniciá una corrida nueva e incluilo."
        )
        return False
    return True


def _render_intro(month: pd.Timestamp) -> None:
    """Aclaraciones sobre qué se predice."""
    st.markdown(
        f"Predice si la demanda eléctrica de **{MONTH_NAMES[month.month - 1]} de "
        f"{month.year}** (el mes siguiente al último dato disponible) será *alta* "
        "respecto de lo normal para su época. La predicción es a **nivel mensual**: "
        "no dice nada de un día ni de una hora."
    )
    st.caption(
        "Usa el escenario A: solo variables que se conocen antes de que termine el "
        "mes (clima esperado, calendario y demanda de meses anteriores). La "
        "potencia máxima ocurre en el mismo mes, así que no se puede usar."
    )


def _render_form(run_id: str, clean: pd.DataFrame) -> None:
    """Formulario: mes fijo, temperatura esperada, rezagos y botón."""
    month = predict.next_month(clean)
    st.subheader("Datos de entrada")
    month_col, year_col, temp_col = st.columns(3)
    month_col.number_input("Mes", value=int(month.month), disabled=True)
    year_col.number_input("Año", value=int(month.year), disabled=True, format="%d")
    low, high = predict.TEMPERATURE_RANGE
    temperature = temp_col.number_input(
        "Temperatura media esperada (°C)",
        min_value=low,
        max_value=high,
        value=round(predict.default_temperature(clean, month.month), 1),
        step=0.1,
        help="Por defecto, el promedio histórico de ese mes.",
    )
    st.caption(
        "El mes es fijo: solo se conocen los rezagos del mes siguiente al último "
        "dato. Dos meses adelante faltaría la demanda del mes intermedio."
    )
    entry = predict.build_input(clean, temperature)
    st.markdown("**Rezagos que se completan solos desde `datos_limpios`**")
    st.dataframe(
        lags_for_display(entry.lags, month).style.format(precision=1),
        hide_index=True,
        width="stretch",
    )
    if st.button("Predecir", type="primary"):
        _predict(run_id, temperature)


def _predict(run_id: str, temperature: float) -> None:
    """Predice, guarda y vuelve a dibujar la página con el resultado."""
    try:
        with st.spinner("Prediciendo con los tres modelos…"):
            doc = predict.predict_next_month(run_id, temperature)
    except Exception as error:  # noqa: BLE001 - se muestra el error en pantalla
        st.error(f"La predicción falló: {error}. El detalle está en la página Logs.")
        return
    state.flash(f"Predicción guardada (id {doc['id']}) en la corrida {run_id}.")
    st.rerun()


def _render_caveats(run_id: str, doc: dict) -> None:
    """Advertencias honestas sobre cómo leer las probabilidades."""
    table = queries.results_table(run_id)
    noise = queries.noise_summary(table, queries.result_docs(run_id))
    beating = [
        model_label(name)
        for name, m in doc["modelos"].items()
        if m["supera_referencia"]
    ]
    st.warning(
        "**Cómo leer estas probabilidades.** No están calibradas: los modelos se "
        "entrenaron dando más peso a los meses de demanda alta, así que inflan la "
        "probabilidad (en el historial solo uno de cada cuatro meses es alto). Un "
        "0,90 no significa «90 % de chances»."
    )
    if beating:
        st.info(
            f"En el hold-out ({noise['n_test']} meses, solo {noise['positives']} de "
            f"demanda alta) únicamente superaron a la regla «siempre alta» "
            f"(F1 {noise['always_high_f1']:.2f}): {', '.join(beating)}. Con tan pocos "
            "casos esa ventaja puede ser ruido."
        )
    else:
        st.error(
            f"Ningún modelo superó en el hold-out a la regla «siempre alta» (F1 "
            f"{noise['always_high_f1']:.2f}): una predicción «alta» aporta poca "
            "información más allá de lo que ya dice el calendario."
        )


def _render_result(run_id: str) -> None:
    """Último resultado guardado de la corrida, leído de la base."""
    doc = queries.latest_prediction(run_id)
    if doc is None:
        st.info("Todavía no hiciste ninguna predicción con esta corrida.")
        return
    st.subheader("Último resultado")
    st.markdown(
        f"**{doc['fecha'][:7]}** con **{doc['temperatura']:.1f} °C** esperados "
        f"(predicción n.º {doc['id']})."
    )
    st.dataframe(
        prediction_for_display(doc).style.format(precision=3),
        hide_index=True,
        width="stretch",
    )
    _render_caveats(run_id, doc)


def _render_history(run_id: str) -> None:
    """Historial de predicciones guardadas."""
    st.subheader("Historial de predicciones guardadas")
    scope = st.radio(
        "Mostrar",
        [HISTORY_SCOPE_RUN, HISTORY_SCOPE_ALL],
        horizontal=True,
        key="pred_scope",
    )
    selected = run_id if scope == HISTORY_SCOPE_RUN else None
    history = queries.prediction_history(selected)
    if history.empty:
        st.caption("No hay predicciones guardadas.")
        return
    st.dataframe(
        history_for_display(history, with_run=selected is None).style.format(
            precision=3
        ),
        hide_index=True,
        width="stretch",
    )


def render() -> None:
    """Dibuja la página Predicción."""
    st.header("Predicción")
    run_id = state.current_run_id()
    if not _ready(run_id):
        return
    clean = data.load_clean_data()
    _render_intro(predict.next_month(clean))
    _render_form(run_id, clean)
    _render_result(run_id)
    _render_history(run_id)
