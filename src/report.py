"""Reporte PDF de una corrida, armado con ReportLab solo desde lo guardado.

El PDF sale de las bases y del ``run_id`` (ver ``report_data``), nunca de la
memoria de Streamlit. Dice qué se hizo, qué resultó y cuánto se puede confiar
en eso: si no hay un modelo claramente mejor, o ninguno supera a la regla
trivial «siempre alta», el reporte lo afirma sin maquillaje.

El diseño vive en ``report_layout`` y las secciones en ``report_sections``; acá
solo se arma el documento y se guarda.
"""

import io
import logging
from collections.abc import Callable
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import SimpleDocTemplate

from src import db, logger, queries, report_charts, report_data, report_sections
from src.report_data import ReportData
from src.report_layout import INK_SECONDARY, TITLE, make_styles, pdf_text

log = logging.getLogger(__name__)


def _document(
    buffer: io.BytesIO, run_id: str, page_compression: int
) -> SimpleDocTemplate:
    """Documento A4 con márgenes de 2 cm y metadatos fijos (salida reproducible)."""
    return SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title=pdf_text(f"{TITLE}: corrida {run_id}"),
        author="TP Final de Programación Avanzada",
        subject="Reporte de una corrida",
        invariant=1,
        pageCompression=page_compression,
    )


def _footer_drawer(run_id: str) -> Callable[[Canvas, SimpleDocTemplate], None]:
    """Función que dibuja el pie de cada página: la corrida y el número de página."""

    def footer(canvas: Canvas, document: SimpleDocTemplate) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(INK_SECONDARY)
        canvas.drawString(2 * cm, 1.1 * cm, pdf_text(f"Corrida {run_id}"))
        canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"Página {document.page}")
        canvas.restoreState()

    return footer


def build_pdf(
    data: ReportData, charts: dict[str, bytes], page_compression: int = 1
) -> bytes:
    """Arma el PDF completo.

    Args:
        data: datos del reporte (``report_data.collect_report_data``).
        charts: gráficos en PNG (``report_charts.render_charts``).
        page_compression: 0 deja el texto legible dentro del PDF (para tests).

    Returns:
        Los bytes del PDF. Con los mismos datos, siempre los mismos bytes.
    """
    styles = make_styles()
    story = [
        *report_sections.cover(data, styles),
        *report_sections.section_data(data, styles),
        *report_sections.section_results(data, charts, styles),
        *report_sections.section_comparison(data, charts, styles),
        *report_sections.section_recommendation(data, charts, styles),
        *report_sections.section_predictions(data, styles),
        *report_sections.section_limits(data, styles),
    ]
    buffer = io.BytesIO()
    footer = _footer_drawer(data.run_id)
    _document(buffer, data.run_id, page_compression).build(
        story, onFirstPage=footer, onLaterPages=footer
    )
    return buffer.getvalue()


def generate_report(
    run_id: str,
    include_predictions: bool = True,
    db_path: Path | None = None,
    unstructured_path: Path | None = None,
    models_dir: Path | None = None,
    page_compression: int = 1,
) -> bytes:
    """Genera el reporte PDF de una corrida, solo desde lo guardado en las bases.

    Args:
        run_id: corrida entrenada.
        include_predictions: si se incluyen las predicciones guardadas.
        db_path: ruta de ``structured.db``; por defecto la de ``config``.
        unstructured_path: ruta de ``unstructured.db``; por defecto la de
            ``config``.
        models_dir: carpeta de modelos; por defecto ``config.MODELS_DIR``.
        page_compression: 0 deja el texto legible dentro del PDF.

    Returns:
        Los bytes del PDF.

    Raises:
        ValueError: si la corrida no existe o no está entrenada.
    """
    data = report_data.collect_report_data(
        run_id, include_predictions, db_path, unstructured_path, models_dir
    )
    return build_pdf(data, report_charts.render_charts(data), page_compression)


def report_path(run_id: str, reports_dir: Path | None = None) -> Path:
    """Ruta del PDF de una corrida: ``reports/reporte_<run_id>.pdf``."""
    return queries.report_path(run_id, reports_dir)


@logger.log_step
def _generate_and_write(
    run_id: str,
    include_predictions: bool,
    reports_dir: Path | None,
    generate: Callable[..., bytes] | None,
    paths: dict[str, Path | None],
) -> Path:
    """Genera el PDF, lo escribe en disco y deja un evento en el log."""
    pdf = (generate or generate_report)(run_id, include_predictions, **paths)
    path = report_path(run_id, reports_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pdf)
    log.info(
        "Reporte generado: %s (%d KB, predicciones %s)",
        path.name,
        len(pdf) // 1024,
        "incluidas" if include_predictions else "omitidas",
    )
    return path


def save_report(
    run_id: str,
    include_predictions: bool = True,
    reports_dir: Path | None = None,
    generate: Callable[..., bytes] | None = None,
    **paths: Path | None,
) -> Path:
    """Genera el reporte y lo guarda en ``reports/``, pisando uno anterior.

    Los eventos quedan en ``logs`` atados al ``run_id``.

    Args:
        run_id: corrida entrenada.
        include_predictions: si se incluyen las predicciones guardadas.
        reports_dir: carpeta de salida; por defecto ``config.REPORTS_DIR``.
        generate: función generadora (para pruebas); por defecto ``generate_report``.
        **paths: ``db_path``, ``unstructured_path`` y ``models_dir`` opcionales.

    Returns:
        La ruta del PDF guardado.
    """
    db_path = paths.get("db_path")
    db.init_structured_db(db_path)
    logger.setup_logging()
    token = logger.current_run_id.set(run_id)
    try:
        return _generate_and_write(
            run_id, include_predictions, reports_dir, generate, paths
        )
    finally:
        logger.current_run_id.reset(token)
        logger.flush_logs()
        db.checkpoint(db_path)


def report_exists(run_id: str, reports_dir: Path | None = None) -> bool:
    """Indica si ya se generó el PDF de una corrida."""
    return queries.has_report(run_id, reports_dir)
