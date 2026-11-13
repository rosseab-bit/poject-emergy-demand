"""Tests de los gráficos del reporte (se revisa lo que dibujan, no solo el PNG)."""

import dataclasses
import itertools
import struct
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex

from src import config, report_charts
from src.report_charts import BLUE, INK, INK_SECONDARY, RED


def png_size(png: bytes) -> tuple[int, int]:
    """Ancho y alto de un PNG, leídos del encabezado."""
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", png[16:24])


def bar_heights(ax) -> list[float]:
    """Alturas de las barras de un panel."""
    return [round(p.get_height(), 6) for p in ax.patches]


def test_f1_chart_draws_the_holdout_bars_the_validation_marks_and_the_reference(
    trained_report_data,
) -> None:
    """Barras = F1 del hold-out; punto = promedio de validación; línea = «siempre alta»."""
    data = trained_report_data
    figure = report_charts.f1_comparison_figure(data)
    axes = [a for a in figure.axes]
    assert len(axes) == 2
    for ax, scenario in zip(axes, ["A", "B"], strict=True):
        rows = data.table[data.table["scenario"] == scenario]
        assert bar_heights(ax) == [round(v, 6) for v in rows["f1"]]
        assert {to_hex(p.get_facecolor()) for p in ax.patches} == {BLUE}
        # barra de error: promedio y desvío de la validación cruzada
        container = ax.containers[-1]
        assert list(container.lines[0].get_ydata()) == pytest.approx(
            list(rows["cv_mean"])
        )
        # línea de referencia
        references = [
            line
            for line in ax.lines
            if len(set(line.get_ydata())) == 1
            and line.get_linestyle() != "None"
            and line.get_ydata()[0] == 0.4
        ]
        assert len(references) == 1
        assert tuple(ax.get_ylim()) == (0, 1)
        assert SCENARIO_WORD[scenario] in ax.get_title(loc="left")


SCENARIO_WORD = {"A": "Escenario A (anticipación)", "B": "Escenario B (diagnóstico)"}


def test_f1_chart_has_a_legend_with_its_three_encodings_and_values_written(
    trained_report_data,
) -> None:
    """Hay leyenda de las tres marcas, y cada barra lleva su valor escrito con coma."""
    figure = report_charts.f1_comparison_figure(trained_report_data)
    legend = figure.legends[0]
    assert [t.get_text() for t in legend.get_texts()] == [
        "F1 en el hold-out",
        "F1 en validación: promedio ± desvío",
        "Regla «siempre alta» (referencia)",
    ]
    texts = [t.get_text() for ax in figure.axes for t in ax.texts]
    expected = [f"{v:.2f}".replace(".", ",") for v in trained_report_data.table["f1"]]
    assert sorted(texts) == sorted(expected)


def test_f1_chart_marks_are_neutral_ink_not_extra_hues(trained_report_data) -> None:
    """El promedio de validación usa tinta neutra; la referencia, tinta secundaria."""
    figure = report_charts.f1_comparison_figure(trained_report_data)
    ax = figure.axes[0]
    marks = ax.containers[-1].lines[0]
    assert to_hex(marks.get_color()) == INK
    reference = next(line for line in ax.lines if list(line.get_ydata()) == [0.4, 0.4])
    assert to_hex(reference.get_color()) == INK_SECONDARY


def test_confusion_chart_shows_every_matrix_exactly(trained_report_data) -> None:
    """Cada panel muestra los cuatro números guardados de la matriz de confusión."""
    data = trained_report_data
    figure = report_charts.confusion_figure(data)
    panels = [ax for ax in figure.axes if ax.axison]
    docs = list(data.docs)
    assert len(panels) == len(docs) == 4
    shown = sorted(sorted(int(t.get_text()) for t in ax.texts) for ax in panels)
    stored = sorted(
        sorted(itertools.chain.from_iterable(doc["holdout"]["confusion_matrix"]))
        for doc in docs
    )
    assert shown == stored
    assert all(sum(values) == 48 for values in shown)


def test_confusion_panels_are_labeled_with_real_and_predicted(
    trained_report_data,
) -> None:
    """Los ejes dicen qué es lo real y qué lo predicho."""
    figure = report_charts.confusion_figure(trained_report_data)
    panels = [ax for ax in figure.axes if ax.axison]
    assert all(ax.get_xlabel() == "Predicho" for ax in panels)
    assert [t.get_text() for t in panels[0].get_xticklabels()] == ["normal", "alta"]
    left = [ax for ax in panels if ax.get_ylabel()]
    assert all("Real" in ax.get_ylabel() for ax in left)
    assert {ax.get_title() for ax in panels} == {"Regresión logística", "Random Forest"}


def test_importance_chart_for_coefficients_uses_sign_color_direction_and_legend() -> (
    None
):
    """Coeficientes: rojo hacia «alta» (derecha), azul hacia «normal» (izquierda)."""
    importance = pd.DataFrame(
        {
            "feature": ["anio", "grados_dia_frio", "demanda_total_lag12", "mes_cos"],
            "importancia": [2.3, -1.5, 0.9, -0.2],
            "tipo": "coeficiente",
        }
    )
    figure = report_charts.importance_figure(importance, "logistic_regression", "A")
    ax = figure.axes[0]
    patches = ax.patches
    widths = [p.get_width() for p in patches]
    colors = [to_hex(p.get_facecolor()) for p in patches]
    assert sorted(widths) == sorted([2.3, -1.5, 0.9, -0.2])
    for width, color in zip(widths, colors, strict=True):
        assert color == (RED if width > 0 else BLUE)
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert (
        labels[0] == "Año" and labels[-1] == "Componente coseno del mes"
    )  # mayor arriba
    assert [t.get_text() for t in figure.legends[0].get_texts()] == [
        "Empuja hacia «alta» (coeficiente positivo)",
        "Empuja hacia «normal» (coeficiente negativo)",
    ]
    assert "anticipación" in figure._suptitle.get_text()
    assert ax.texts[0].get_text() == "2,30"


def test_importance_chart_for_tree_importances_is_one_color_without_legend() -> None:
    """Importancias de árboles: una sola tonalidad y sin leyenda de signo."""
    importance = pd.DataFrame(
        {
            "feature": ["demanda_total_lag1", "temperatura_promedio"],
            "importancia": [0.3, 0.2],
            "tipo": "importancia",
        }
    )
    figure = report_charts.importance_figure(importance, "random_forest", "B")
    assert {to_hex(p.get_facecolor()) for p in figure.axes[0].patches} == {BLUE}
    assert figure.legends == []
    assert figure.axes[0].get_xlabel() == "Importancia"


def test_importance_chart_shows_only_the_top_variables() -> None:
    """Muestra a lo sumo las 10 variables con más peso."""
    importance = pd.DataFrame(
        {
            "feature": [f"f{i}" for i in range(15)],
            "importancia": np.linspace(1, 0.1, 15),
            "tipo": "importancia",
        }
    )
    figure = report_charts.importance_figure(importance, "xgboost", "A")
    assert len(figure.axes[0].patches) == report_charts.TOP_VARIABLES


def test_axes_use_decimal_comma(trained_report_data) -> None:
    """Los números de los ejes llevan coma decimal, igual que el texto."""
    figure = report_charts.f1_comparison_figure(trained_report_data)
    figure.canvas.draw()
    ticks = [t.get_text() for t in figure.axes[0].get_yticklabels()]
    assert "0,4" in ticks and not any("." in t for t in ticks)


def test_short_feature_labels_drop_the_article_and_capitalize() -> None:
    """Las variables se muestran sin «el/la» y con mayúscula inicial."""
    assert report_charts.short_feature_label("anio") == "Año"
    assert report_charts.short_feature_label("grados_dia_frio") == "Grados-día de frío"
    assert (
        report_charts.short_feature_label("demanda_total_lag1")
        == "Demanda total del mes anterior"
    )
    assert report_charts.short_feature_label("desconocida") == "Desconocida"


def test_render_charts_returns_valid_pngs(trained_report_data) -> None:
    """Devuelve PNG válidos y de tamaño razonable, uno por gráfico."""
    charts = report_charts.render_charts(trained_report_data)
    assert set(charts) == {"f1", "confusion", "importance_A", "importance_B"}
    for name, png in charts.items():
        width, height = png_size(png)
        assert width >= 1000 and height >= 700, name
        assert len(png) < 600_000, name


def test_render_charts_is_deterministic(trained_report_data) -> None:
    """Dibujar dos veces el mismo gráfico da los mismos bytes."""
    first = report_charts.figure_to_png(
        report_charts.f1_comparison_figure(trained_report_data)
    )
    second = report_charts.figure_to_png(
        report_charts.f1_comparison_figure(trained_report_data)
    )
    assert first == second


def test_render_charts_skips_variables_that_could_not_be_read(
    trained_report_data,
) -> None:
    """Si no se pudo leer un modelo, ese gráfico de variables se omite sin romper el resto."""
    import copy

    data = copy.copy(trained_report_data)
    data.conclusions = dict(data.conclusions)
    broken = copy.copy(data.conclusions["A"])
    broken.importance = None
    data.conclusions["A"] = broken
    assert set(report_charts.render_charts(data)) == {"f1", "confusion", "importance_B"}


def test_module_does_not_load_pyplot() -> None:
    """Usa la API orientada a objetos: no carga pyplot ni su estado global."""
    code = "import sys; import src.report_charts; print('matplotlib.pyplot' in sys.modules)"
    done = subprocess.run(
        [sys.executable, "-c", code],
        cwd=config.ROOT_DIR,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    assert done.stdout.strip() == "False"


def test_confusion_chart_leaves_empty_panels_when_scenarios_have_different_models(
    trained_report_data,
) -> None:
    """Si un escenario tiene menos modelos que otro, sus paneles sobrantes quedan vacíos."""
    base = trained_report_data
    uneven = dataclasses.replace(
        base,
        docs=[
            d
            for d in base.docs
            if (d["scenario"], d["model"]) != ("B", "random_forest")
        ],
    )
    figure = report_charts.confusion_figure(uneven)
    panels = [ax for ax in figure.axes if ax.axison]
    assert len(figure.axes) == 4 and len(panels) == 3
    assert sum(len(ax.texts) for ax in panels) == 12  # 4 conteos por panel
