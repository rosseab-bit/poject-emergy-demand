"""Página Reporte: generar el PDF de la corrida y descargarlo."""

from datetime import datetime

import streamlit as st

from app_pages import state
from src import queries, report

CONTENTS = (
    "Portada: corrida, fecha, semilla, hash del dataset, commit y versiones",
    "Datos y ETL: filas, correcciones, balance de clases y supuestos",
    "Resultados de cada modelo y escenario, con matrices de confusión",
    "Comparación: tabla y gráfico con promedio y desvío de la validación cruzada",
    "Modelo recomendado y por qué (o «sin ganador claro»), con las reglas a la vista",
    "Predicciones realizadas",
    "Límites del trabajo",
)


def _render_contents() -> None:
    """Índice de lo que contiene el reporte."""
    st.markdown("El PDF se arma solo con lo guardado en las bases para esta corrida:")
    st.markdown("\n".join(f"{n}. {item}" for n, item in enumerate(CONTENTS, start=1)))
    st.caption(
        "Dice con las mismas reglas de la página Comparación si hay un modelo "
        "claramente mejor; si la diferencia está dentro del ruido de la validación "
        "cruzada o ningún modelo supera a «siempre alta», no declara un ganador."
    )


def _generate(run_id: str, include_predictions: bool) -> None:
    """Genera y guarda el PDF; vuelve a dibujar la página al terminar."""
    try:
        with st.spinner("Armando el reporte (tablas y gráficos)…"):
            path = report.save_report(run_id, include_predictions)
    except Exception as error:  # noqa: BLE001 - se muestra el error en pantalla
        st.error(
            f"No se pudo generar el reporte: {error}. El detalle está en la página Logs."
        )
        return
    state.flash(f"Reporte generado: {path.name}.")
    st.rerun()


def _render_download(run_id: str) -> None:
    """Datos del archivo generado y botón de descarga."""
    path = report.report_path(run_id)
    generated = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    st.subheader("Reporte generado")
    st.markdown(
        f"`{path.name}` · {path.stat().st_size // 1024} KB · generado el {generated}."
    )
    st.download_button(
        "Descargar PDF",
        data=path.read_bytes(),
        file_name=path.name,
        mime="application/pdf",
        type="primary",
    )
    st.caption(
        "Se guardó en la carpeta reports/. Si después hacés más predicciones o "
        "entrenás otra corrida, generalo de nuevo para que las incluya."
    )


def render() -> None:
    """Dibuja la página Reporte."""
    st.header("Reporte")
    run_id = state.current_run_id()
    summary = queries.run_summary(run_id) if run_id else None
    if summary is None or not summary["steps"][queries.STEP_TRAINING]:
        st.warning("Primero entrená los modelos en la página Entrenamiento.")
        return
    st.markdown(f"Corrida **`{run_id}`**.")
    _render_contents()
    predictions = len(queries.prediction_docs(run_id))
    include = st.checkbox(
        "Incluir las predicciones guardadas",
        value=True,
        help="Agrega a la sección 5 la lista de predicciones hechas con esta corrida.",
    )
    st.caption(f"Predicciones guardadas para esta corrida: {predictions}.")
    if st.button("Generar PDF", type="primary"):
        _generate(run_id, include)
    if report.report_exists(run_id):
        _render_download(run_id)
