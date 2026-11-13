"""Reporte PDF de una corrida, armado con ReportLab solo desde lo guardado.

El PDF sale de las bases y del ``run_id`` (ver ``report_data``), nunca de la
memoria de Streamlit. Dice qué se hizo, qué resultó y cuánto se puede confiar
en eso: si no hay un modelo claramente mejor, o ninguno supera a la regla
trivial «siempre alta», el reporte lo afirma sin maquillaje.
"""

import io
import logging
import struct
from collections.abc import Callable
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src import db, logger, queries, recommend, report_charts, report_data
from src.explain import fmt
from src.models import MODEL_NAMES, model_label
from src.report_data import ReportData, ScenarioConclusion

log = logging.getLogger(__name__)

TITLE = "Clasificación de demanda eléctrica mensual"
INK = colors.HexColor("#0b0b0b")
INK_SECONDARY = colors.HexColor("#52514e")
ACCENT = colors.HexColor("#2a78d6")
GRID = colors.HexColor("#c3c2b7")
HEADER_FILL = colors.HexColor("#e8e7e1")
ZEBRA = colors.HexColor("#f6f5f2")
# Colores de estado: siempre van con una etiqueta escrita.
STATUS = {
    "good": colors.HexColor("#0ca30c"),
    "warning": colors.HexColor("#fab219"),
    "critical": colors.HexColor("#d03b3b"),
}
CATEGORY_STATUS = {
    recommend.CLEAR_WINNER_BEATS: ("Recomendado", "good"),
    recommend.TIE_BEATS: ("Recomendación provisoria", "warning"),
    recommend.CLEAR_WINNER_NO_BEAT: ("No recomendado", "warning"),
    recommend.TIE_NO_BEAT: ("Sin modelo recomendable", "critical"),
}
SCENARIO_NAMES = {"A": "A · anticipación", "B": "B · diagnóstico"}
# Helvetica (la tipografía estándar del PDF) solo cubre Latin-1/WinAnsi.
TEXT_REPLACEMENTS = {"−": "-", "→": "->", "≥": ">=", "≤": "<="}


def pdf_text(text: object) -> str:
    """Prepara un texto para el PDF: reemplaza símbolos que Helvetica no tiene."""
    out = str(text)
    for old, new in TEXT_REPLACEMENTS.items():
        out = out.replace(old, new)
    return out


def esc(text: object) -> str:
    """Escapa un texto dinámico para usarlo dentro de un ``Paragraph``."""
    return escape(pdf_text(text))


def _styles() -> dict[str, ParagraphStyle]:
    """Estilos de párrafo del reporte."""
    base = getSampleStyleSheet()["Normal"]
    body = ParagraphStyle(
        "body",
        parent=base,
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=INK,
        alignment=TA_LEFT,
        spaceAfter=4,
    )
    return {
        "body": body,
        "small": ParagraphStyle(
            "small", parent=body, fontSize=8, leading=11, textColor=INK_SECONDARY
        ),
        "cell": ParagraphStyle(
            "cell", parent=body, fontSize=8, leading=10, spaceAfter=0
        ),
        "cell_bold": ParagraphStyle(
            "cell_bold",
            parent=body,
            fontSize=8,
            leading=10,
            spaceAfter=0,
            fontName="Helvetica-Bold",
        ),
        "title": ParagraphStyle(
            "title",
            parent=body,
            fontName="Helvetica-Bold",
            fontSize=24,
            leading=28,
            spaceAfter=6,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            parent=body,
            fontSize=13,
            leading=17,
            textColor=INK_SECONDARY,
            spaceAfter=14,
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=body,
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=19,
            spaceBefore=14,
            spaceAfter=6,
            textColor=INK,
            keepWithNext=1,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=body,
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            spaceBefore=10,
            spaceAfter=4,
            textColor=INK,
            keepWithNext=1,
        ),
        "bullet": ParagraphStyle(
            "bullet", parent=body, leftIndent=12, bulletIndent=2, spaceAfter=3
        ),
        "verdict": ParagraphStyle(
            "verdict", parent=body, fontName="Helvetica-Bold", fontSize=10.5, leading=14
        ),
    }


def png_aspect(png: bytes) -> float:
    """Alto dividido por ancho de un PNG, leído de su encabezado."""
    width, height = struct.unpack(">II", png[16:24])
    return height / width


def _image(png: bytes, width: float) -> Image:
    """Imagen PNG escalada a un ancho, conservando su proporción."""
    return Image(io.BytesIO(png), width=width, height=width * png_aspect(png))


def _table(
    rows: list[list[object]],
    widths: list[float],
    styles: dict,
    header: bool = True,
    align_right_from: int | None = None,
) -> Table:
    """Tabla con encabezado, filas alternadas y celdas que ajustan el texto."""
    cells = []
    for r, row in enumerate(rows):
        style = styles["cell_bold"] if header and r == 0 else styles["cell"]
        cells.append(
            [c if not isinstance(c, str) else Paragraph(esc(c), style) for c in row]
        )
    table = Table(cells, colWidths=widths, repeatRows=1 if header else 0)
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, GRID),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if header:
        commands.append(("BACKGROUND", (0, 0), (-1, 0), HEADER_FILL))
    for r in range(1 if header else 0, len(rows)):
        if (r % 2 == 0) == header:
            commands.append(("BACKGROUND", (0, r), (-1, r), ZEBRA))
    table.setStyle(TableStyle(commands))
    return table


def _bullets(texts: list[str], styles: dict) -> list[Paragraph]:
    """Lista con viñetas."""
    return [Paragraph(esc(t), styles["bullet"], bulletText="•") for t in texts]


def format_params(params: dict[str, object]) -> str:
    """Hiperparámetros sin el prefijo ``clf__``, por ejemplo ``C=0.01``."""
    return ", ".join(f"{k.removeprefix('clf__')}={v}" for k, v in params.items())


def _pct(value: float) -> str:
    """Proporción como porcentaje con coma."""
    return f"{fmt(100 * value, 1)} %"


# ----------------------------------------------------------------- secciones


def _cover(data: ReportData, styles: dict) -> list:
    """Portada: identificación, reproducibilidad y la conclusión en una línea."""
    cfg, run = data.config, data.run
    versions = ", ".join(f"{k} {v}" for k, v in cfg["versions"].items())
    commit = cfg.get("git_commit") or "no registrado"
    if cfg.get("git_dirty"):
        commit += " (con cambios sin commitear)"
    rows = [
        ["Corrida (run_id)", data.run_id],
        ["Fecha del reporte (UTC)", data.generated_at],
        ["Estado de la corrida", f"{run['status']} (inicio {run['started_at']})"],
        ["Semilla", str(cfg["seed"])],
        ["Hash del dataset (SHA-256)", cfg["dataset_sha256"]],
        ["Commit de Git", commit],
        ["Versiones", versions],
        [
            "Validación",
            (
                f"hold-out de los últimos {cfg['holdout_months']} meses y "
                f"TimeSeriesSplit de {cfg['cv_splits']} cortes"
            ),
        ],
    ]
    flow = [
        Spacer(1, 1.5 * cm),
        Paragraph(esc(TITLE), styles["title"]),
        Paragraph(esc(f"Reporte de la corrida {data.run_id}"), styles["subtitle"]),
        _table(rows, [4.8 * cm, 12.2 * cm], styles, header=False),
        Spacer(1, 10),
    ]
    if data.notes:
        flow.append(Paragraph("Avisos", styles["h2"]))
        flow.extend(_bullets(data.notes, styles))
    main = data.conclusions.get("A") or next(iter(data.conclusions.values()))
    flow.append(Paragraph("Conclusión principal", styles["h2"]))
    flow.append(_verdict_box(main, styles))
    flow.append(Paragraph("Contenido", styles["h2"]))
    flow.extend(
        _bullets(
            [
                "1. Datos y ETL",
                "2. Resultados por modelo y escenario",
                "3. Comparación",
                "4. Modelo recomendado y por qué",
                "5. Predicciones realizadas",
                "6. Límites",
            ],
            styles,
        )
    )
    flow.append(PageBreak())
    return flow


def _verdict_box(conclusion: ScenarioConclusion, styles: dict) -> Table:
    """Recuadro con la conclusión de un escenario y su etiqueta de estado."""
    label, status = CATEGORY_STATUS[conclusion.verdict.category]
    scenario = SCENARIO_NAMES.get(conclusion.scenario, conclusion.scenario)
    text = Paragraph(
        f"<b>{esc(label)}</b> (escenario {esc(scenario)}). {esc(conclusion.explanation.headline)}",
        styles["body"],
    )
    box = Table([["", text]], colWidths=[0.35 * cm, 16.6 * cm])
    box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, 0), STATUS[status]),
                ("BACKGROUND", (1, 0), (1, 0), ZEBRA),
                ("VALIGN", (1, 0), (1, 0), "MIDDLE"),
                ("TOPPADDING", (1, 0), (1, 0), 6),
                ("BOTTOMPADDING", (1, 0), (1, 0), 6),
                ("LEFTPADDING", (1, 0), (1, 0), 8),
            ]
        )
    )
    return box


def _section_data(data: ReportData, styles: dict) -> list:
    """1. Datos y ETL: filas, correcciones, target y supuestos."""
    o, cfg, t = data.overview, data.config, data.overview["target"]
    target_cfg = cfg["target"]
    repeated = ", ".join(o["repeated_months"]) or "ninguno"
    flow = [Paragraph("1. Datos y ETL", styles["h1"])]
    flow.append(
        Paragraph(
            esc(
                f"Se usa el dataset mensual nacional de demanda eléctrica, de {o['first_month']} a "
                f"{o['last_month']}. La tabla datos_limpios tiene {o['rows']} meses; "
                f"{o['rows_complete']} tienen todas las variables."
            ),
            styles["body"],
        )
    )
    rows = [
        ["Concepto", "Valor"],
        ["Meses guardados", str(o["rows"])],
        ["Meses con todas las variables", str(o["rows_complete"])],
        ["Meses sin sectores (anteriores a 2005)", str(o["rows_sector_nulls"])],
        [
            "Potencia máxima repetida (puesta en nulo)",
            f"{len(o['repeated_months'])}: {repeated}",
        ],
    ]
    flow += [_table(rows, [9 * cm, 8 * cm], styles), Spacer(1, 6)]
    flow.append(Paragraph("Correcciones", styles["h2"]))
    flow += _bullets(
        [
            (
                "La potencia máxima repetida del mes anterior se tomó como error de carga: se puso en "
                "nulo y se marcó; el modelo la imputa solo con datos de entrenamiento."
            ),
            (
                "Los sectores (residencial, comercio e industria, grandes usuarios) no existen antes de "
                "2005: no se imputan; esos meses se usan solo por su demanda total, como historia."
            ),
        ],
        styles,
    )
    flow.append(Paragraph("Definición del target", styles["h2"]))
    flow.append(
        Paragraph(
            esc(
                f"Un mes es de demanda alta si el cociente entre su demanda y el promedio del mismo mes "
                f"de los {target_cfg['years_back']} años anteriores supera el cuantil "
                f"{fmt(100 * target_cfg['quantile'], 0)} % de los cocientes de los "
                f"{target_cfg['window_months']} meses anteriores. Usa solo el pasado."
            ),
            styles["body"],
        )
    )
    rows = [
        ["Meses con target", "Demanda alta", "Demanda normal", "Proporción de altas"],
        [
            str(t["total"]),
            str(t["positivos"]),
            str(t["negativos"]),
            _pct(t["proporcion_positivos"]),
        ],
    ]
    flow += [
        _table(rows, [4.25 * cm] * 4, styles),
        Paragraph(
            esc(f"El primer mes con target es {t['first_month']}."), styles["small"]
        ),
    ]
    flow.append(Paragraph("Supuestos", styles["h2"]))
    flow += _bullets(
        [
            (
                "La temperatura del dataset se asume de Buenos Aires / AMBA (inferido comparando con las "
                "normales de esa ciudad; la fuente original no lo confirma)."
            ),
            "La predicción es a nivel mensual: no dice nada de un día ni de una hora.",
            (
                f"Los grados-día de frío y calor se calculan con base de {fmt(cfg['base_temp_c'], 0)} °C y "
                "la temperatura media mensual (aproximación: no hay temperaturas diarias)."
            ),
        ],
        styles,
    )
    flow.append(Paragraph("Variables de cada escenario", styles["h2"]))
    for scenario, names in cfg["scenarios"].items():
        if scenario not in data.conclusions:
            continue  # un escenario que no se entrenó en esta corrida no se describe
        flow.append(
            Paragraph(
                esc(
                    f"Escenario {SCENARIO_NAMES.get(scenario, scenario)}: "
                    f"{len(names)} variables: {', '.join(names)}."
                ),
                styles["small"],
            )
        )
    return flow


def _model_rows(data: ReportData, scenario: str) -> list:
    """Documentos de resultados de un escenario, de modelo simple a complejo."""
    order = {name: i for i, name in enumerate(MODEL_NAMES)}
    docs = [d for d in data.docs if d["scenario"] == scenario]
    return sorted(docs, key=lambda d: order.get(d["model"], len(order)))


def _section_results(data: ReportData, charts: dict[str, bytes], styles: dict) -> list:
    """2. Resultados por modelo y escenario, con matrices de confusión."""
    flow = [
        Paragraph("2. Resultados por modelo y escenario", styles["h1"]),
        Paragraph(
            esc(
                f"Los tres modelos usan la misma semilla y los mismos cortes. Los hiperparámetros se "
                f"eligen con TimeSeriesSplit maximizando el F1 de la clase «alta»; el hold-out "
                f"({int(data.noise['n_test'])} meses, {int(data.noise['positives'])} de demanda "
                f"alta) se evalúa una sola vez."
            ),
            styles["body"],
        ),
    ]
    for scenario in data.conclusions:
        docs = _model_rows(data, scenario)
        block: list = []
        block.append(
            Paragraph(
                esc(f"Escenario {SCENARIO_NAMES.get(scenario, scenario)}"), styles["h2"]
            )
        )
        rows = [["Modelo", "Hiperparámetros", "F1 por corte", "Promedio ± desvío"]]
        for d in docs:
            scores = ", ".join(fmt(s, 2) for s in d["cv"]["scores"])
            rows.append(
                [
                    model_label(d["model"]),
                    format_params(d["params"]),
                    scores,
                    f"{fmt(d['cv']['mean'])} ± {fmt(d['cv']['std'])}",
                ]
            )
        block += [
            Paragraph(
                "Validación cruzada (solo con datos de entrenamiento)", styles["small"]
            ),
            _table(rows, [3.2 * cm, 6 * cm, 3.7 * cm, 4.1 * cm], styles),
            Spacer(1, 4),
        ]
        rows = [
            [
                "Modelo",
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
                "F1 «siempre alta»",
                "F1 aleatorio",
            ]
        ]
        for d in docs:
            h, b = d["holdout"], d["baselines"]
            rows.append(
                [
                    model_label(d["model"]),
                    fmt(h["accuracy"]),
                    fmt(h["precision"]),
                    fmt(h["recall"]),
                    fmt(h["f1"]),
                    fmt(b["siempre_alta"]["f1"]),
                    fmt(b["aleatorio_proporcional"]["f1"]),
                ]
            )
        block += [
            Paragraph("Hold-out (la clase positiva es «alta»)", styles["small"]),
            _table(
                rows,
                [3.2 * cm, 2.1 * cm, 2.1 * cm, 2.1 * cm, 2.1 * cm, 3 * cm, 2.4 * cm],
                styles,
            ),
            Spacer(1, 4),
        ]
        flow.append(KeepTogether(block))
    flow.append(Paragraph("Matrices de confusión del hold-out", styles["h2"]))
    flow.append(_image(charts["confusion"], 16 * cm))
    return flow


def _section_comparison(
    data: ReportData, charts: dict[str, bytes], styles: dict
) -> list:
    """3. Comparación entre modelos y entre escenarios."""
    flow = [
        Paragraph("3. Comparación", styles["h1"]),
        Paragraph(
            esc(
                "La referencia «siempre alta» marca todos los meses como demanda alta. Un modelo solo "
                "aporta algo si la supera por más que su desvío de validación."
            ),
            styles["body"],
        ),
    ]
    rows = [
        [
            "Escenario",
            "Modelo",
            "F1 validación (prom. ± desv.)",
            "F1 hold-out",
            "F1 «siempre alta»",
            "¿Supera la referencia?",
        ]
    ]
    for scenario, conclusion in data.conclusions.items():
        for score in sorted(
            conclusion.scores, key=lambda s: recommend.complexity_rank(s.model)
        ):
            rows.append(
                [
                    scenario,
                    model_label(score.model),
                    f"{fmt(score.cv_mean)} ± {fmt(score.cv_std)}",
                    fmt(score.f1),
                    fmt(score.reference_f1),
                    "Sí" if recommend.beats_reference(score) else "No",
                ]
            )
    flow += [
        _table(
            rows, [2 * cm, 3.4 * cm, 4.2 * cm, 2.3 * cm, 2.6 * cm, 2.5 * cm], styles
        ),
        Spacer(1, 6),
        _image(charts["f1"], 16.5 * cm),
    ]
    comparison = queries.scenario_comparison(data.table)
    if not comparison.empty:
        flow.append(
            Paragraph(
                "Escenario A contra B: ¿cuánto aporta la potencia máxima?", styles["h2"]
            )
        )
        rows = [
            [
                "Modelo",
                "F1 hold-out A",
                "F1 hold-out B",
                "Diferencia (B - A)",
                "F1 validación A",
                "F1 validación B",
            ]
        ]
        for r in comparison.itertuples():
            rows.append(
                [
                    model_label(r.model),
                    fmt(r.f1_A),
                    fmt(r.f1_B),
                    fmt(r.dif_f1),
                    fmt(r.cv_A),
                    fmt(r.cv_B),
                ]
            )
        flow += [
            _table(
                rows, [3.4 * cm, 2.7 * cm, 2.7 * cm, 3 * cm, 2.6 * cm, 2.6 * cm], styles
            ),
            Paragraph(
                esc(
                    "La potencia máxima ocurre en el mismo mes que la demanda y no se "
                    "conoce de antemano: el escenario B sirve para diagnosticar, no "
                    "para anticipar."
                ),
                styles["small"],
            ),
        ]
    return flow


def _section_recommendation(
    data: ReportData, charts: dict[str, bytes], styles: dict
) -> list:
    """4. Modelo recomendado y por qué, con reglas explícitas."""
    flow = [
        Paragraph("4. Modelo recomendado y por qué", styles["h1"]),
        Paragraph(
            esc(
                "Regla: el mejor es el de mayor F1 en el hold-out. Otro modelo empata con él si la "
                "diferencia es menor que el mayor de los dos desvíos de validación cruzada; con varios "
                "empatados se desempata por simplicidad (regresión logística, Random Forest, XGBoost). "
                "El elegido solo se recomienda si supera a «siempre alta» por más que su desvío. La "
                "recomendación para predecir sale del escenario A."
            ),
            styles["body"],
        ),
    ]
    for scenario, conclusion in data.conclusions.items():
        block = [
            Paragraph(
                esc(f"Escenario {SCENARIO_NAMES.get(scenario, scenario)}"), styles["h2"]
            ),
            _verdict_box(conclusion, styles),
            Spacer(1, 5),
            *_bullets(list(conclusion.explanation.reasons), styles),
        ]
        flow.append(KeepTogether(block[:3]))
        flow += block[3:]
        key = f"importance_{scenario}"
        if key in charts:
            flow.append(_image(charts[key], 15 * cm))
    return flow


def _section_predictions(data: ReportData, styles: dict) -> list:
    """5. Predicciones realizadas con los modelos de la corrida."""
    flow = [Paragraph("5. Predicciones realizadas", styles["h1"])]
    if not data.predictions_included:
        flow.append(
            Paragraph(
                esc("Este reporte se generó sin incluir las predicciones."),
                styles["body"],
            )
        )
        return flow
    if not data.predictions:
        flow.append(
            Paragraph(
                esc("No hay predicciones guardadas para esta corrida."), styles["body"]
            )
        )
        return flow
    names = [m for m in MODEL_NAMES if any(m in p["modelos"] for p in data.predictions)]
    rows = [
        ["N.º", "Guardada (UTC)", "Mes", "Temp. (°C)", *[model_label(m) for m in names]]
    ]
    for p in data.predictions:
        cells = []
        for name in names:
            entry = p["modelos"].get(name)
            cells.append(
                "-"
                if entry is None
                else f"{fmt(entry['probabilidad'], 2)} ({'alta' if entry['clase'] == 1 else 'normal'})"
            )
        rows.append(
            [
                str(p["id"]),
                str(p["created_at"]),
                p["fecha"][:7],
                fmt(p["temperatura"], 1),
                *cells,
            ]
        )
    widths = [1.1 * cm, 4.2 * cm, 1.8 * cm, 1.9 * cm] + [
        (17 - 9) * cm / len(names)
    ] * len(names)
    flow += [
        Paragraph(
            esc(
                "Cada predicción es para el mes siguiente al último dato disponible, con el escenario A y "
                "los rezagos completados desde datos_limpios. Se muestra la probabilidad de demanda alta "
                "y la clase (umbral 0,5)."
            ),
            styles["body"],
        ),
        _table(rows, widths, styles),
        Paragraph(
            esc(
                "Las probabilidades no están calibradas: los modelos se entrenaron dando más "
                "peso a los meses de demanda alta y la inflan. Un 0,90 no significa 90 % de "
                "chances."
            ),
            styles["small"],
        ),
    ]
    return flow


def limits(data: ReportData) -> list[str]:
    """Límites del trabajo, armados con las cifras de la corrida."""
    cfg, noise = data.config, data.noise
    target = data.overview["target"]
    items = [
        (
            f"El hold-out tiene {int(noise['n_test'])} meses y solo {int(noise['positives'])} son de "
            "demanda alta: un acierto de más o de menos cambia mucho el F1, y las diferencias entre "
            "modelos pueden ser ruido."
        ),
        (
            "Hay un único corte temporal y una única semilla "
            f"({cfg['seed']}): con otros valores los resultados pueden variar."
        ),
        (
            "Las probabilidades no están calibradas (los modelos se entrenaron con pesos que "
            "favorecen la clase «alta»): no se pueden leer como chances reales."
        ),
        "La temperatura se asume de Buenos Aires / AMBA, sin confirmación de la fuente.",
        (
            "La predicción es a nivel mensual y solo del mes siguiente al último dato; no sirve para "
            "días ni horas."
        ),
        (
            "El escenario B usa la potencia máxima, que no se conoce de antemano: no sirve para "
            "anticipar."
        ),
        (
            f"El target depende de una ventana móvil de {cfg['target']['window_months']} meses: el primer "
            f"mes con target es {target['first_month']} y hay {target['total']} meses con target."
        ),
    ]
    main = data.conclusions.get("A")
    if main is not None and not main.verdict.recommend_for_use:
        items.append(
            "En el escenario A ningún modelo supera con claridad a la regla «siempre alta»: una "
            "predicción de demanda alta no debe tomarse como evidencia por sí sola."
        )
    return items


def _section_limits(data: ReportData, styles: dict) -> list:
    """6. Límites."""
    return [Paragraph("6. Límites", styles["h1"]), *_bullets(limits(data), styles)]


# ---------------------------------------------------------------- documento


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
    styles = _styles()
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title=pdf_text(f"{TITLE}: corrida {data.run_id}"),
        author="TP Final de Programación Avanzada",
        subject="Reporte de una corrida",
        invariant=1,
        pageCompression=page_compression,
    )

    def footer(canvas: Canvas, document: SimpleDocTemplate) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(INK_SECONDARY)
        canvas.drawString(2 * cm, 1.1 * cm, pdf_text(f"Corrida {data.run_id}"))
        canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"Página {document.page}")
        canvas.restoreState()

    story = [
        *_cover(data, styles),
        *_section_data(data, styles),
        *_section_results(data, charts, styles),
        *_section_comparison(data, charts, styles),
        *_section_recommendation(data, charts, styles),
        *_section_predictions(data, styles),
        *_section_limits(data, styles),
    ]
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
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
    logger.setup_logging(db_path)
    token = logger.current_run_id.set(run_id)
    try:
        return _generate_and_write(
            run_id, include_predictions, reports_dir, generate, paths
        )
    finally:
        logger.current_run_id.reset(token)


def report_exists(run_id: str, reports_dir: Path | None = None) -> bool:
    """Indica si ya se generó el PDF de una corrida."""
    return queries.has_report(run_id, reports_dir)
