"""Página Datos: ejecutar el ETL y ver los datos limpios, el target y gráficos."""

import pandas as pd
import streamlit as st

from app_pages import state
from src import config, data, pipeline, queries


def _render_intro() -> None:
    """Resumen del proyecto y de los supuestos."""
    st.markdown(
        "Se clasifica si la **demanda eléctrica de un mes** es *alta* respecto de "
        "lo normal para su época, a partir del clima y el calendario. "
        "La predicción es a **nivel mensual** (no diaria ni horaria)."
    )
    st.info(
        "**Supuesto:** la temperatura del dataset se asume de **Buenos Aires / "
        "AMBA**. Se infirió comparando el promedio mensual con las normales de "
        "esa ciudad, pero la fuente original no lo confirma.",
        icon="ℹ️",
    )


def _run_etl_button() -> None:
    """Botón que ejecuta el ETL como una corrida nueva y la deja seleccionada."""
    seed = state.current_seed(config.SEED)
    st.caption(
        f"Cada ejecución crea una corrida nueva con la semilla **{seed}** "
        "(se cambia en la barra lateral)."
    )
    if not st.button("Ejecutar ETL", type="primary"):
        return
    try:
        with st.spinner("Leyendo, limpiando y guardando los datos…"):
            run_id = pipeline.run_etl(seed=seed)
    except Exception as error:  # noqa: BLE001 - se muestra el error en pantalla
        st.error(f"El ETL falló: {error}. El detalle está en la página Logs.")
        return
    state.flash(f"ETL ejecutado: se creó la corrida {run_id}.")
    state.select_run(run_id)


def _render_overview(overview: dict) -> None:
    """Cifras del ETL, correcciones y definición del target."""
    st.subheader("Resultado del ETL")
    first, second, third, fourth = st.columns(4)
    first.metric("Meses guardados", overview["rows"])
    second.metric("Meses con todas las variables", overview["rows_complete"])
    third.metric("Meses sin sectores (antes de 2005)", overview["rows_sector_nulls"])
    fourth.metric("Potencia máxima repetida", len(overview["repeated_months"]))
    st.caption(f"Período: {overview['first_month']} a {overview['last_month']}.")

    st.markdown("**Correcciones hechas**")
    repeated = ", ".join(overview["repeated_months"]) or "ninguno"
    st.markdown(
        f"- **Potencia máxima repetida** del mes anterior (error de carga): "
        f"{repeated}. Se puso en nulo y se marcó en `potencia_repetida`; el "
        "modelo la imputa solo con datos de entrenamiento.\n"
        "- **Sectores nulos** (residencial, comercio e industria, grandes "
        "usuarios) solo antes de 2005, porque la fuente no los tiene. No se "
        "imputan: esos meses se usan solo por su demanda total, como historia."
    )

    target = overview["target"]
    st.subheader("Definición del target")
    st.markdown(
        f"Un mes es **demanda alta** si el cociente entre su demanda y el "
        f"promedio del mismo mes de los **{config.TARGET_YEARS_BACK} años "
        f"anteriores** supera el **cuantil {config.TARGET_QUANTILE:.0%}** de los "
        f"cocientes de los **{config.TARGET_WINDOW_MONTHS} meses anteriores**. "
        "Usa solo el pasado."
    )
    one, two, three, four = st.columns(4)
    one.metric("Meses con target", target["total"])
    two.metric("Demanda alta", target["positivos"])
    three.metric("Demanda normal", target["negativos"])
    four.metric("Proporción de altas", f"{target['proporcion_positivos']:.1%}")
    st.caption(f"El primer mes con target es {target['first_month']}.")


def _render_charts(clean: pd.DataFrame) -> None:
    """Serie de demanda, estacionalidad y temperatura contra demanda."""
    st.subheader("Gráficos")
    st.markdown("**Demanda total mensual**")
    st.line_chart(
        clean.set_index(data.DATE_COLUMN)["demanda_total"], y_label="Demanda total"
    )

    st.markdown("**Estacionalidad:** demanda del mes sobre el promedio de su año")
    season = queries.seasonality_by_month(clean)
    st.bar_chart(season, y_label="Índice medio (1 = promedio anual)")
    st.caption(
        "Solo años con los 12 meses; el índice quita la tendencia de crecimiento."
    )

    st.markdown("**Temperatura contra demanda:** cada punto es un mes")
    indexed = queries.add_demand_index(clean).dropna(subset=["indice_demanda"])
    st.scatter_chart(
        indexed,
        x="temperatura_promedio",
        y="indice_demanda",
        x_label="Temperatura media del mes (°C)",
        y_label="Índice de demanda",
    )


def render() -> None:
    """Dibuja la página Datos."""
    st.header("Datos")
    _render_intro()
    _run_etl_button()
    overview = queries.data_overview()
    if overview is None:
        st.warning("Todavía no hay datos. Primero ejecutá el ETL.")
        return
    _render_overview(overview)
    clean = data.load_clean_data()
    _render_charts(clean)
    st.subheader("Tabla datos_limpios")
    st.caption(f"{len(clean)} filas; las variables derivadas se calculan al entrenar.")
    st.dataframe(clean, width="stretch", hide_index=True)
