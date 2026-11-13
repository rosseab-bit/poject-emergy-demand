"""Gráficos del reporte PDF, dibujados con matplotlib sin ``pyplot``.

Usan la paleta de referencia del proyecto sobre fondo claro (el PDF es para
leer e imprimir): azul para magnitudes, tinta neutra para el promedio de
validación y la referencia, y el par azul/rojo con el signo de los coeficientes.
El significado nunca depende solo del color: hay leyenda, valores escritos y la
dirección de las barras.
"""

import io

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.axis import Axis
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

from src import features, models
from src.report_data import ReportData

BLUE = "#2a78d6"
RED = "#e34948"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
# Rampa secuencial azul (pasos 100 a 700) para las matrices de confusión.
BLUE_RAMP = (
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#256abf",
    "#184f95",
    "#0d366b",
)

SCENARIO_TITLES = {"A": "Escenario A (anticipación)", "B": "Escenario B (diagnóstico)"}
TOP_VARIABLES = 10
DPI = 200


def figure_to_png(figure: Figure, dpi: int = DPI) -> bytes:
    """Renderiza una figura como PNG y devuelve los bytes."""
    buffer = io.BytesIO()
    FigureCanvasAgg(figure)
    figure.savefig(buffer, format="png", dpi=dpi, facecolor=figure.get_facecolor())
    return buffer.getvalue()


def _style_axes(ax: Axes) -> None:
    """Estilo común: sin marco, grilla fina, tinta secundaria."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _decimal(value: float, digits: int = 2) -> str:
    """Número con coma decimal y signo menos tipográfico."""
    return f"{value:.{digits}f}".replace(".", ",").replace("-", "\u2212")


def _comma_axis(axis: Axis, digits: int) -> None:
    """Hace que los números del eje usen coma decimal."""
    axis.set_major_formatter(FuncFormatter(lambda v, _pos: _decimal(v, digits)))


def _wrap_model(name: str) -> str:
    """Nombre del modelo partido en dos líneas para el eje."""
    return models.model_label(name).replace(" ", "\n", 1)


def short_feature_label(name: str) -> str:
    """Nombre de variable sin artículo y con mayúscula inicial."""
    label = features.feature_label(name)
    for article in ("el ", "la ", "los ", "las "):
        if label.startswith(article):
            label = label[len(article) :]
            break
    return label[:1].upper() + label[1:]


def f1_comparison_figure(data: ReportData) -> Figure:
    """F1 en el hold-out por modelo, con el promedio ± desvío de validación.

    Un panel por escenario. Las barras son el F1 del hold-out; el punto con
    barra de error es el promedio y el desvío de la validación cruzada; la
    línea punteada es la regla «siempre alta».
    """
    scenarios = list(data.conclusions)
    figure = Figure(figsize=(3.6 * len(scenarios) + 0.6, 4.2), facecolor=SURFACE)
    axes = figure.subplots(1, len(scenarios), sharey=True, squeeze=False)[0]
    for ax, scenario in zip(axes, scenarios, strict=True):
        rows = data.table[data.table["scenario"] == scenario].reset_index(drop=True)
        x = np.arange(len(rows))
        ax.bar(x, rows["f1"], width=0.55, color=BLUE, zorder=2)
        ax.errorbar(
            x,
            rows["cv_mean"],
            yerr=rows["cv_std"],
            fmt="o",
            color=INK,
            markersize=8,
            capsize=4,
            linewidth=1.5,
            zorder=3,
        )
        ax.axhline(
            float(rows["f1_siempre_alta"].iloc[0]),
            color=INK_SECONDARY,
            linestyle=(0, (5, 4)),
            linewidth=1.5,
            zorder=1,
        )
        tops = np.maximum(rows["f1"], rows["cv_mean"] + rows["cv_std"])
        for position, value, top in zip(x, rows["f1"], tops, strict=True):
            ax.text(
                position,
                min(float(top) + 0.015, 0.95),
                _decimal(value),
                ha="center",
                va="bottom",
                fontsize=8,
                color=INK,
                zorder=4,
                bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.5},
            )
        ax.set_xticks(x, [_wrap_model(m) for m in rows["model"]])
        ax.set_ylim(0, 1)
        ax.set_xlim(-0.6, len(rows) - 0.4)
        ax.set_title(
            SCENARIO_TITLES.get(scenario, f"Escenario {scenario}"),
            fontsize=10,
            color=INK,
            loc="left",
        )
        _style_axes(ax)
        _comma_axis(ax.yaxis, 1)
    axes[0].set_ylabel("F1 de la clase «alta»", fontsize=9, color=INK_SECONDARY)
    handles = [
        Patch(facecolor=BLUE, label="F1 en el hold-out"),
        Line2D(
            [],
            [],
            color=INK,
            marker="o",
            markersize=8,
            linewidth=1.5,
            label="F1 en validación: promedio ± desvío",
        ),
        Line2D(
            [],
            [],
            color=INK_SECONDARY,
            linestyle=(0, (5, 4)),
            linewidth=1.5,
            label="Regla «siempre alta» (referencia)",
        ),
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=8,
        labelcolor=INK_SECONDARY,
    )
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    return figure


def _ramp_color(fraction: float) -> str:
    """Color de la rampa azul para una fracción entre 0 y 1."""
    index = min(int(fraction * len(BLUE_RAMP)), len(BLUE_RAMP) - 1)
    return BLUE_RAMP[index]


def confusion_figure(data: ReportData) -> Figure:
    """Matrices de confusión del hold-out: una fila por escenario, un panel por modelo."""
    scenarios = list(data.conclusions)
    docs = {(d["scenario"], d["model"]): d for d in data.docs}
    columns = max(sum(1 for (s, _m) in docs if s == scenario) for scenario in scenarios)
    figure = Figure(
        figsize=(2.9 * columns + 0.4, 2.9 * len(scenarios) + 0.2), facecolor=SURFACE
    )
    axes = figure.subplots(len(scenarios), columns, squeeze=False)
    n_test = int(data.noise["n_test"])
    for row, scenario in enumerate(scenarios):
        names = [m for m in models.MODEL_NAMES if (scenario, m) in docs]
        for column in range(columns):
            ax = axes[row][column]
            if column >= len(names):
                ax.axis("off")
                continue
            matrix = np.array(
                docs[(scenario, names[column])]["holdout"]["confusion_matrix"]
            )
            image = np.array(
                [
                    [_hex_to_rgb(_ramp_color(v / n_test)) for v in line]
                    for line in matrix.tolist()
                ]
            )
            ax.imshow(image, aspect="equal")
            for (i, j), value in np.ndenumerate(matrix):
                dark = value / n_test >= 0.35
                ax.text(
                    j,
                    i,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=14,
                    fontweight="bold",
                    color="#ffffff" if dark else INK,
                )
            ax.set_xticks([0, 1], ["normal", "alta"])
            ax.set_yticks([0, 1], ["normal", "alta"])
            ax.tick_params(colors=INK_SECONDARY, labelsize=8, length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xlabel("Predicho", fontsize=8, color=INK_SECONDARY)
            if column == 0:
                ax.set_ylabel(
                    f"Escenario {scenario}\nReal", fontsize=8, color=INK_SECONDARY
                )
            ax.set_title(models.model_label(names[column]), fontsize=9, color=INK)
    figure.tight_layout()
    return figure


def _hex_to_rgb(color: str) -> tuple[float, float, float]:
    """Convierte ``#rrggbb`` a una tupla de flotantes entre 0 y 1."""
    return tuple(int(color[i : i + 2], 16) / 255 for i in (1, 3, 5))


def importance_figure(importance: pd.DataFrame, model: str, scenario: str) -> Figure:
    """Variables con más peso en un modelo: coeficientes con signo o importancias.

    Args:
        importance: salida de ``results.feature_importance`` (ordenada por valor
            absoluto, de mayor a menor).
        model: nombre del modelo.
        scenario: ``"A"`` o ``"B"``.
    """
    top = importance.head(TOP_VARIABLES)
    signed = (top["tipo"] == "coeficiente").all()
    figure = Figure(figsize=(8.4, 0.42 * len(top) + 1.7), facecolor=SURFACE)
    ax = figure.subplots()
    y = np.arange(len(top))[::-1]
    values = top["importancia"].to_numpy()
    colors = [RED if v > 0 else BLUE for v in values] if signed else [BLUE] * len(top)
    ax.barh(y, values, height=0.6, color=colors, zorder=2)
    span = float(np.abs(values).max()) or 1.0
    for position, value in zip(y, values, strict=True):
        ax.text(
            value + (0.02 * span if value >= 0 else -0.02 * span),
            position,
            _decimal(value),
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=8,
            color=INK,
        )
    ax.set_yticks(y, [short_feature_label(f) for f in top["feature"]])
    ax.axvline(0, color=AXIS, linewidth=1)
    margin = 0.25 * span
    ax.set_xlim(
        min(values.min(), 0) - (margin if signed else 0), max(values.max(), 0) + margin
    )
    _comma_axis(ax.xaxis, 2)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    _style_axes(ax)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(
        "Coeficiente (variables estandarizadas)" if signed else "Importancia",
        fontsize=9,
        color=INK_SECONDARY,
    )
    figure.suptitle(
        f"{models.model_label(model)} · {SCENARIO_TITLES.get(scenario, scenario)}",
        fontsize=10,
        color=INK,
        x=0.01,
        ha="left",
    )
    if signed:
        figure.legend(
            handles=[
                Patch(
                    facecolor=RED, label="Empuja hacia «alta» (coeficiente positivo)"
                ),
                Patch(
                    facecolor=BLUE, label="Empuja hacia «normal» (coeficiente negativo)"
                ),
            ],
            loc="lower center",
            ncol=2,
            frameon=False,
            fontsize=8,
            labelcolor=INK_SECONDARY,
        )
        figure.tight_layout(rect=(0, 0.07, 1, 0.97))
    else:
        figure.tight_layout(rect=(0, 0, 1, 0.97))
    return figure


def render_charts(data: ReportData) -> dict[str, bytes]:
    """Dibuja todos los gráficos del reporte y los devuelve como PNG.

    Args:
        data: datos del reporte.

    Returns:
        Diccionario nombre a bytes PNG: ``f1``, ``confusion`` y un
        ``importance_<escenario>`` por cada escenario con variables legibles.
    """
    charts = {
        "f1": figure_to_png(f1_comparison_figure(data)),
        "confusion": figure_to_png(confusion_figure(data)),
    }
    for scenario, conclusion in data.conclusions.items():
        if conclusion.importance is not None:
            figure = importance_figure(
                conclusion.importance, conclusion.verdict.recommended, scenario
            )
            charts[f"importance_{scenario}"] = figure_to_png(figure)
    return charts
