"""Piezas de diseño del reporte PDF: estilos, tablas, imágenes y textos.

No conocen los datos del reporte; solo saben dibujar. Las secciones
(``report_sections``) las combinan.
"""

import io
import struct
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, Table, TableStyle

from src import recommend
from src.explain import fmt
from src.report_data import ScenarioConclusion

Styles = dict[str, ParagraphStyle]

TITLE = "Clasificación de demanda eléctrica mensual"
INK = colors.HexColor("#0b0b0b")
INK_SECONDARY = colors.HexColor("#52514e")
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


def text_paragraph(text: object, style: ParagraphStyle) -> Paragraph:
    """Párrafo con un texto dinámico ya escapado."""
    return Paragraph(esc(text), style)


def format_params(params: dict[str, object]) -> str:
    """Hiperparámetros sin el prefijo ``clf__``, por ejemplo ``C=0.01``."""
    return ", ".join(f"{k.removeprefix('clf__')}={v}" for k, v in params.items())


def pct(value: float) -> str:
    """Proporción como porcentaje con coma."""
    return f"{fmt(100 * value, 1)} %"


# ------------------------------------------------------------------ estilos


def _body_style() -> ParagraphStyle:
    """Estilo del texto corrido, del que heredan los demás."""
    return ParagraphStyle(
        "body",
        parent=getSampleStyleSheet()["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=INK,
        alignment=TA_LEFT,
        spaceAfter=4,
    )


def _cell_styles(body: ParagraphStyle) -> Styles:
    """Estilos del texto chico y de las celdas de las tablas."""
    return {
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
    }


def _title_styles(body: ParagraphStyle) -> Styles:
    """Estilos del título y el subtítulo de la portada."""
    return {
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
    }


def _heading_styles(body: ParagraphStyle) -> Styles:
    """Estilos de los títulos de sección; se mantienen junto a lo que sigue."""
    return {
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
    }


def make_styles() -> Styles:
    """Todos los estilos de párrafo del reporte."""
    body = _body_style()
    return {
        "body": body,
        **_cell_styles(body),
        **_title_styles(body),
        **_heading_styles(body),
        "bullet": ParagraphStyle(
            "bullet", parent=body, leftIndent=12, bulletIndent=2, spaceAfter=3
        ),
        "verdict": ParagraphStyle(
            "verdict", parent=body, fontName="Helvetica-Bold", fontSize=10.5, leading=14
        ),
    }


# ------------------------------------------------------ tablas e imágenes


def png_aspect(png: bytes) -> float:
    """Alto dividido por ancho de un PNG, leído de su encabezado."""
    width, height = struct.unpack(">II", png[16:24])
    return height / width


def scaled_image(png: bytes, width: float) -> Image:
    """Imagen PNG escalada a un ancho, conservando su proporción."""
    return Image(io.BytesIO(png), width=width, height=width * png_aspect(png))


def _cells(rows: list[list[object]], styles: Styles, header: bool) -> list:
    """Celdas de texto como párrafos, para que ajusten el ancho de su columna."""
    cells = []
    for r, row in enumerate(rows):
        style = styles["cell_bold"] if header and r == 0 else styles["cell"]
        cells.append(
            [text_paragraph(c, style) if isinstance(c, str) else c for c in row]
        )
    return cells


def _table_commands(n_rows: int, header: bool) -> list:
    """Estilo de la tabla: línea fina, encabezado y filas alternadas."""
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, GRID),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if header:
        commands.append(("BACKGROUND", (0, 0), (-1, 0), HEADER_FILL))
    for r in range(1 if header else 0, n_rows):
        if (r % 2 == 0) == header:
            commands.append(("BACKGROUND", (0, r), (-1, r), ZEBRA))
    return commands


def make_table(
    rows: list[list[object]], widths: list[float], styles: Styles, header: bool = True
) -> Table:
    """Tabla con encabezado, filas alternadas y celdas que ajustan el texto."""
    table = Table(
        _cells(rows, styles, header), colWidths=widths, repeatRows=1 if header else 0
    )
    table.setStyle(TableStyle(_table_commands(len(rows), header)))
    return table


def bullet_list(texts: list[str], styles: Styles) -> list[Paragraph]:
    """Lista con viñetas."""
    return [Paragraph(esc(t), styles["bullet"], bulletText="•") for t in texts]


def verdict_box(conclusion: ScenarioConclusion, styles: Styles) -> Table:
    """Recuadro con la conclusión de un escenario y su etiqueta de estado."""
    label, status = CATEGORY_STATUS[conclusion.verdict.category]
    scenario = SCENARIO_NAMES.get(conclusion.scenario, conclusion.scenario)
    text = Paragraph(
        f"<b>{esc(label)}</b> (escenario {esc(scenario)}). "
        f"{esc(conclusion.explanation.headline)}",
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
