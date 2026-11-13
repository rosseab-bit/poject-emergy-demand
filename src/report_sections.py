"""Secciones del reporte PDF: portada y las seis partes del informe.

Cada sección devuelve una lista de elementos de ReportLab armados a partir de
``ReportData`` (lo guardado en las bases). El diseño vive en ``report_layout``.
"""

import pandas as pd
from reportlab.lib.units import cm
from reportlab.platypus import KeepTogether, PageBreak, Paragraph, Spacer

from src import queries, recommend
from src.explain import fmt
from src.models import MODEL_NAMES, model_label
from src.report_data import ReportData, ScenarioConclusion
from src.report_layout import (
    SCENARIO_NAMES,
    TITLE,
    Styles,
    bullet_list,
    format_params,
    make_table,
    pct,
    scaled_image,
    text_paragraph,
    verdict_box,
)

COVER_CONTENTS = [
    "1. Datos y ETL",
    "2. Resultados por modelo y escenario",
    "3. Comparación",
    "4. Modelo recomendado y por qué",
    "5. Predicciones realizadas",
    "6. Límites",
]


def _scenario_title(scenario: str) -> str:
    """Nombre de un escenario para los títulos: ``Escenario A · anticipación``."""
    return f"Escenario {SCENARIO_NAMES.get(scenario, scenario)}"


# ------------------------------------------------------------------ portada


def _cover_rows(data: ReportData) -> list[list[str]]:
    """Filas de la tabla de identificación y reproducibilidad de la portada."""
    cfg, run = data.config, data.run
    versions = ", ".join(f"{k} {v}" for k, v in cfg["versions"].items())
    commit = cfg.get("git_commit") or "no registrado"
    if cfg.get("git_dirty"):
        commit += " (con cambios sin commitear)"
    validation = (
        f"hold-out de los últimos {cfg['holdout_months']} meses y "
        f"TimeSeriesSplit de {cfg['cv_splits']} cortes"
    )
    return [
        ["Corrida (run_id)", data.run_id],
        ["Fecha del reporte (UTC)", data.generated_at],
        ["Estado de la corrida", f"{run['status']} (inicio {run['started_at']})"],
        ["Semilla", str(cfg["seed"])],
        ["Hash del dataset (SHA-256)", cfg["dataset_sha256"]],
        ["Commit de Git", commit],
        ["Versiones", versions],
        ["Validación", validation],
    ]


def cover(data: ReportData, styles: Styles) -> list:
    """Portada: identificación, reproducibilidad y la conclusión en una línea."""
    flow = [
        Spacer(1, 1.5 * cm),
        text_paragraph(TITLE, styles["title"]),
        text_paragraph(f"Reporte de la corrida {data.run_id}", styles["subtitle"]),
        make_table(_cover_rows(data), [4.8 * cm, 12.2 * cm], styles, header=False),
        Spacer(1, 10),
    ]
    if data.notes:
        flow.append(Paragraph("Avisos", styles["h2"]))
        flow.extend(bullet_list(data.notes, styles))
    main = data.conclusions.get("A") or next(iter(data.conclusions.values()))
    flow.append(Paragraph("Conclusión principal", styles["h2"]))
    flow.append(verdict_box(main, styles))
    flow.append(Paragraph("Contenido", styles["h2"]))
    flow.extend(bullet_list(COVER_CONTENTS, styles))
    flow.append(PageBreak())
    return flow


# --------------------------------------------------------- 1. Datos y ETL


def _data_intro(overview: dict, styles: Styles) -> list:
    """Párrafo de apertura de la sección de datos."""
    return [
        text_paragraph(
            f"Se usa el dataset mensual nacional de demanda eléctrica, de "
            f"{overview['first_month']} a {overview['last_month']}. La tabla "
            f"datos_limpios tiene {overview['rows']} meses; "
            f"{overview['rows_complete']} tienen todas las variables.",
            styles["body"],
        )
    ]


def _data_counts(overview: dict, styles: Styles) -> list:
    """Tabla con las cifras del ETL."""
    repeated = ", ".join(overview["repeated_months"]) or "ninguno"
    rows = [
        ["Concepto", "Valor"],
        ["Meses guardados", str(overview["rows"])],
        ["Meses con todas las variables", str(overview["rows_complete"])],
        ["Meses sin sectores (anteriores a 2005)", str(overview["rows_sector_nulls"])],
        [
            "Potencia máxima repetida (puesta en nulo)",
            f"{len(overview['repeated_months'])}: {repeated}",
        ],
    ]
    return [make_table(rows, [9 * cm, 8 * cm], styles), Spacer(1, 6)]


def _data_corrections(styles: Styles) -> list:
    """Las correcciones que hizo el ETL."""
    return [
        Paragraph("Correcciones", styles["h2"]),
        *bullet_list(
            [
                (
                    "La potencia máxima repetida del mes anterior se tomó como error de "
                    "carga: se puso en nulo y se marcó; el modelo la imputa solo con "
                    "datos de entrenamiento."
                ),
                (
                    "Los sectores (residencial, comercio e industria, grandes usuarios) "
                    "no existen antes de 2005: no se imputan; esos meses se usan solo "
                    "por su demanda total, como historia."
                ),
            ],
            styles,
        ),
    ]


def _data_target(overview: dict, config: dict, styles: Styles) -> list:
    """Definición del target y su balance de clases."""
    target, rule = overview["target"], config["target"]
    definition = (
        f"Un mes es de demanda alta si el cociente entre su demanda y el promedio "
        f"del mismo mes de los {rule['years_back']} años anteriores supera el "
        f"cuantil {fmt(100 * rule['quantile'], 0)} % de los cocientes de los "
        f"{rule['window_months']} meses anteriores. Usa solo el pasado."
    )
    rows = [
        ["Meses con target", "Demanda alta", "Demanda normal", "Proporción de altas"],
        [
            str(target["total"]),
            str(target["positivos"]),
            str(target["negativos"]),
            pct(target["proporcion_positivos"]),
        ],
    ]
    return [
        Paragraph("Definición del target", styles["h2"]),
        text_paragraph(definition, styles["body"]),
        make_table(rows, [4.25 * cm] * 4, styles),
        text_paragraph(
            f"El primer mes con target es {target['first_month']}.", styles["small"]
        ),
    ]


def _data_assumptions(config: dict, styles: Styles) -> list:
    """Supuestos del trabajo."""
    return [
        Paragraph("Supuestos", styles["h2"]),
        *bullet_list(
            [
                (
                    "La temperatura del dataset se asume de Buenos Aires / AMBA "
                    "(inferido comparando con las normales de esa ciudad; la fuente "
                    "original no lo confirma)."
                ),
                "La predicción es a nivel mensual: no dice nada de un día ni de una hora.",
                (
                    f"Los grados-día de frío y calor se calculan con base de "
                    f"{fmt(config['base_temp_c'], 0)} °C y la temperatura media mensual "
                    "(aproximación: no hay temperaturas diarias)."
                ),
            ],
            styles,
        ),
    ]


def _data_variables(data: ReportData, styles: Styles) -> list:
    """Variables de cada escenario entrenado (los no entrenados no se describen)."""
    flow = [Paragraph("Variables de cada escenario", styles["h2"])]
    for scenario, names in data.config["scenarios"].items():
        if scenario not in data.conclusions:
            continue
        flow.append(
            text_paragraph(
                f"{_scenario_title(scenario)}: {len(names)} variables: "
                f"{', '.join(names)}.",
                styles["small"],
            )
        )
    return flow


def section_data(data: ReportData, styles: Styles) -> list:
    """1. Datos y ETL: filas, correcciones, target y supuestos."""
    return [
        Paragraph("1. Datos y ETL", styles["h1"]),
        *_data_intro(data.overview, styles),
        *_data_counts(data.overview, styles),
        *_data_corrections(styles),
        *_data_target(data.overview, data.config, styles),
        *_data_assumptions(data.config, styles),
        *_data_variables(data, styles),
    ]


# ----------------------------------------------------------- 2. Resultados


def _model_docs(data: ReportData, scenario: str) -> list:
    """Documentos de resultados de un escenario, de modelo simple a complejo."""
    order = {name: i for i, name in enumerate(MODEL_NAMES)}
    docs = [d for d in data.docs if d["scenario"] == scenario]
    return sorted(docs, key=lambda d: order.get(d["model"], len(order)))


def _validation_rows(docs: list) -> list[list[str]]:
    """Filas de la tabla de validación cruzada de un escenario."""
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
    return rows


def _holdout_rows(docs: list) -> list[list[str]]:
    """Filas de la tabla de métricas del hold-out de un escenario."""
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
    return rows


def _results_block(data: ReportData, scenario: str, styles: Styles) -> KeepTogether:
    """Las dos tablas de un escenario, juntas para que no se partan."""
    docs = _model_docs(data, scenario)
    widths_cv = [3.2 * cm, 6 * cm, 3.7 * cm, 4.1 * cm]
    widths_holdout = [
        3.2 * cm,
        2.1 * cm,
        2.1 * cm,
        2.1 * cm,
        2.1 * cm,
        3 * cm,
        2.4 * cm,
    ]
    return KeepTogether(
        [
            text_paragraph(_scenario_title(scenario), styles["h2"]),
            Paragraph(
                "Validación cruzada (solo con datos de entrenamiento)", styles["small"]
            ),
            make_table(_validation_rows(docs), widths_cv, styles),
            Spacer(1, 4),
            Paragraph("Hold-out (la clase positiva es «alta»)", styles["small"]),
            make_table(_holdout_rows(docs), widths_holdout, styles),
            Spacer(1, 4),
        ]
    )


def section_results(data: ReportData, charts: dict[str, bytes], styles: Styles) -> list:
    """2. Resultados por modelo y escenario, con matrices de confusión."""
    noise = data.noise
    intro = (
        "Los tres modelos usan la misma semilla y los mismos cortes. Los "
        "hiperparámetros se eligen con TimeSeriesSplit maximizando el F1 de la clase "
        f"«alta»; el hold-out ({int(noise['n_test'])} meses, "
        f"{int(noise['positives'])} de demanda alta) se evalúa una sola vez."
    )
    flow = [
        Paragraph("2. Resultados por modelo y escenario", styles["h1"]),
        text_paragraph(intro, styles["body"]),
    ]
    flow.extend(_results_block(data, s, styles) for s in data.conclusions)
    flow.append(Paragraph("Matrices de confusión del hold-out", styles["h2"]))
    flow.append(scaled_image(charts["confusion"], 16 * cm))
    return flow


# ----------------------------------------------------------- 3. Comparación


def _comparison_rows(data: ReportData) -> list[list[str]]:
    """Filas de la tabla modelo por escenario con la referencia."""
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
    return rows


def _scenario_comparison_block(comparison: pd.DataFrame, styles: Styles) -> list:
    """Tabla de A contra B: cuánto aporta la potencia máxima."""
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
    note = (
        "La potencia máxima ocurre en el mismo mes que la demanda y no se conoce de "
        "antemano: el escenario B sirve para diagnosticar, no para anticipar."
    )
    widths = [3.4 * cm, 2.7 * cm, 2.7 * cm, 3 * cm, 2.6 * cm, 2.6 * cm]
    return [
        Paragraph(
            "Escenario A contra B: ¿cuánto aporta la potencia máxima?", styles["h2"]
        ),
        make_table(rows, widths, styles),
        text_paragraph(note, styles["small"]),
    ]


def section_comparison(
    data: ReportData, charts: dict[str, bytes], styles: Styles
) -> list:
    """3. Comparación entre modelos y entre escenarios."""
    intro = (
        "La referencia «siempre alta» marca todos los meses como demanda alta. Un "
        "modelo solo aporta algo si la supera por más que su desvío de validación."
    )
    widths = [2 * cm, 3.4 * cm, 4.2 * cm, 2.3 * cm, 2.6 * cm, 2.5 * cm]
    flow = [
        Paragraph("3. Comparación", styles["h1"]),
        text_paragraph(intro, styles["body"]),
        make_table(_comparison_rows(data), widths, styles),
        Spacer(1, 6),
        scaled_image(charts["f1"], 16.5 * cm),
    ]
    comparison = queries.scenario_comparison(data.table)
    if not comparison.empty:
        flow += _scenario_comparison_block(comparison, styles)
    return flow


# ------------------------------------------------------ 4. Recomendación


RECOMMENDATION_RULE = (
    "Regla: el mejor es el de mayor F1 en el hold-out. Otro modelo empata con él si "
    "la diferencia es menor que el mayor de los dos desvíos de validación cruzada; "
    "con varios empatados se desempata por simplicidad (regresión logística, Random "
    "Forest, XGBoost). El elegido solo se recomienda si supera a «siempre alta» por "
    "más que su desvío. La recomendación para predecir sale del escenario A."
)


def _scenario_conclusion(
    scenario: str,
    conclusion: ScenarioConclusion,
    charts: dict[str, bytes],
    styles: Styles,
) -> list:
    """Conclusión de un escenario: recuadro, razones y variables del recomendado."""
    head = [
        text_paragraph(_scenario_title(scenario), styles["h2"]),
        verdict_box(conclusion, styles),
        Spacer(1, 5),
    ]
    flow = [
        KeepTogether(head),
        *bullet_list(list(conclusion.explanation.reasons), styles),
    ]
    key = f"importance_{scenario}"
    if key in charts:
        flow.append(scaled_image(charts[key], 15 * cm))
    return flow


def section_recommendation(
    data: ReportData, charts: dict[str, bytes], styles: Styles
) -> list:
    """4. Modelo recomendado y por qué, con reglas explícitas."""
    flow = [
        Paragraph("4. Modelo recomendado y por qué", styles["h1"]),
        text_paragraph(RECOMMENDATION_RULE, styles["body"]),
    ]
    for scenario, conclusion in data.conclusions.items():
        flow += _scenario_conclusion(scenario, conclusion, charts, styles)
    return flow


# ------------------------------------------------------- 5. Predicciones


def _prediction_row(prediction: dict, names: list[str]) -> list[str]:
    """Fila de una predicción: identificación y una celda por modelo."""
    cells = []
    for name in names:
        entry = prediction["modelos"].get(name)
        if entry is None:
            cells.append("-")
            continue
        label = "alta" if entry["clase"] == 1 else "normal"
        cells.append(f"{fmt(entry['probabilidad'], 2)} ({label})")
    return [
        str(prediction["id"]),
        str(prediction["created_at"]),
        prediction["fecha"][:7],
        fmt(prediction["temperatura"], 1),
        *cells,
    ]


def _predictions_body(data: ReportData, styles: Styles) -> list:
    """Tabla de predicciones guardadas con sus aclaraciones."""
    names = [m for m in MODEL_NAMES if any(m in p["modelos"] for p in data.predictions)]
    rows = [
        ["N.º", "Guardada (UTC)", "Mes", "Temp. (°C)", *[model_label(m) for m in names]]
    ]
    rows += [_prediction_row(p, names) for p in data.predictions]
    widths = [1.1 * cm, 4.2 * cm, 1.8 * cm, 1.9 * cm] + [
        (17 - 9) * cm / len(names)
    ] * len(names)
    intro = (
        "Cada predicción es para el mes siguiente al último dato disponible, con el "
        "escenario A y los rezagos completados desde datos_limpios. Se muestra la "
        "probabilidad de demanda alta y la clase (umbral 0,5)."
    )
    caveat = (
        "Las probabilidades no están calibradas: los modelos se entrenaron dando más "
        "peso a los meses de demanda alta y la inflan. Un 0,90 no significa 90 % de "
        "chances."
    )
    return [
        text_paragraph(intro, styles["body"]),
        make_table(rows, widths, styles),
        text_paragraph(caveat, styles["small"]),
    ]


def section_predictions(data: ReportData, styles: Styles) -> list:
    """5. Predicciones realizadas con los modelos de la corrida."""
    flow = [Paragraph("5. Predicciones realizadas", styles["h1"])]
    if not data.predictions_included:
        message = "Este reporte se generó sin incluir las predicciones."
    elif not data.predictions:
        message = "No hay predicciones guardadas para esta corrida."
    else:
        return flow + _predictions_body(data, styles)
    return [*flow, text_paragraph(message, styles["body"])]


# ----------------------------------------------------------- 6. Límites


def _base_limits(data: ReportData) -> list[str]:
    """Límites que valen para cualquier corrida, con sus cifras."""
    cfg, noise = data.config, data.noise
    target = data.overview["target"]
    return [
        (
            f"El hold-out tiene {int(noise['n_test'])} meses y solo "
            f"{int(noise['positives'])} son de demanda alta: un acierto de más o de "
            "menos cambia mucho el F1, y las diferencias entre modelos pueden ser ruido."
        ),
        (
            "Hay un único corte temporal y una única semilla "
            f"({cfg['seed']}): con otros valores los resultados pueden variar."
        ),
        (
            "Las probabilidades no están calibradas (los modelos se entrenaron con "
            "pesos que favorecen la clase «alta»): no se pueden leer como chances "
            "reales."
        ),
        "La temperatura se asume de Buenos Aires / AMBA, sin confirmación de la fuente.",
        (
            "La predicción es a nivel mensual y solo del mes siguiente al último dato; "
            "no sirve para días ni horas."
        ),
        (
            "El escenario B usa la potencia máxima, que no se conoce de antemano: no "
            "sirve para anticipar."
        ),
        (
            f"El target depende de una ventana móvil de {cfg['target']['window_months']} "
            f"meses: el primer mes con target es {target['first_month']} y hay "
            f"{target['total']} meses con target."
        ),
    ]


def limits(data: ReportData) -> list[str]:
    """Límites del trabajo, armados con las cifras de la corrida."""
    items = _base_limits(data)
    main = data.conclusions.get("A")
    if main is not None and not main.verdict.recommend_for_use:
        items.append(
            "En el escenario A ningún modelo supera con claridad a la regla «siempre "
            "alta»: una predicción de demanda alta no debe tomarse como evidencia por "
            "sí sola."
        )
    return items


def section_limits(data: ReportData, styles: Styles) -> list:
    """6. Límites."""
    return [Paragraph("6. Límites", styles["h1"]), *bullet_list(limits(data), styles)]
