"""Datos del reporte PDF, reunidos solo a partir de lo guardado en las bases.

Nada sale de la memoria de Streamlit: dado un ``run_id`` se leen la corrida, su
configuración, el ETL, los resultados, los modelos y las predicciones.
"""

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src import db, explain, queries, recommend, results, runs


@dataclass
class ScenarioConclusion:
    """Conclusión de un escenario: veredicto, texto y variables del recomendado.

    Attributes:
        scenario: ``"A"`` o ``"B"``.
        scores: puntajes de cada modelo en el escenario.
        verdict: resultado de la regla de recomendación.
        explanation: conclusión y razones redactadas.
        importance: coeficientes o importancias del modelo recomendado, o
            ``None`` si no se pudo leer el modelo.
    """

    scenario: str
    scores: list[recommend.ModelScore]
    verdict: recommend.Verdict
    explanation: explain.Explanation
    importance: pd.DataFrame | None


@dataclass
class ReportData:
    """Todo lo que lleva el reporte de una corrida.

    Attributes:
        run_id: identificador de la corrida.
        generated_at: fecha de generación del reporte (UTC, ISO 8601).
        run: fila de ``runs`` (estado, fechas, semilla, notas).
        config: documento de ``configuracion`` (hash, versiones, commit...).
        overview: resumen del ETL y del target (``queries.data_overview``).
        table: métricas por modelo y escenario.
        docs: documentos completos de los resultados.
        noise: tamaño y ruido del hold-out (``queries.noise_summary``).
        conclusions: conclusión por escenario, en orden A, B.
        predictions: predicciones guardadas de la corrida (más vieja primero);
            vacío si no se pidió incluirlas.
        notes: avisos sobre lo que no se pudo leer o limita la trazabilidad.
        predictions_included: si se pidió incluir las predicciones en el reporte.
    """

    run_id: str
    generated_at: str
    run: dict[str, object]
    config: dict[str, object]
    overview: dict[str, object]
    table: pd.DataFrame
    docs: list[dict[str, object]]
    noise: dict[str, float]
    conclusions: dict[str, ScenarioConclusion]
    predictions: list[dict[str, object]]
    notes: list[str] = field(default_factory=list)
    predictions_included: bool = True


def traceability_notes(config: dict[str, object]) -> list[str]:
    """Avisos sobre qué tan bien describe la configuración el código usado.

    Args:
        config: documento de ``configuracion`` de la corrida.

    Returns:
        Lista de avisos; vacía si la corrida es plenamente trazable.
    """
    notes = []
    if config.get("git_dirty"):
        notes.append(
            "La corrida se ejecutó con cambios sin commitear: el hash del commit "
            "no describe del todo el código usado."
        )
    if not config.get("git_commit"):
        notes.append("No se registró el commit de Git de la corrida.")
    return notes


def _conclusion(
    run_id: str,
    scenario: str,
    table: pd.DataFrame,
    noise: dict[str, float],
    notes: list[str],
    unstructured_path: Path | None,
    models_dir: Path | None,
) -> ScenarioConclusion:
    """Veredicto, texto e importancias del modelo recomendado de un escenario."""
    scores = recommend.scores_from_table(table, scenario)
    verdict = recommend.evaluate_scenario(scores, scenario)
    importance = None
    try:
        importance = queries.feature_importance_for(
            run_id, scenario, verdict.recommended, unstructured_path, models_dir
        )
    except (FileNotFoundError, ValueError, LookupError) as error:
        notes.append(
            f"No se pudieron leer las variables del modelo recomendado del "
            f"escenario {scenario} ({verdict.recommended}): {error}"
        )
    explanation = explain.explain_scenario(
        verdict, scores, importance, int(noise["n_test"]), int(noise["positives"])
    )
    return ScenarioConclusion(scenario, scores, verdict, explanation, importance)


def collect_report_data(
    run_id: str,
    include_predictions: bool = True,
    db_path: Path | None = None,
    unstructured_path: Path | None = None,
    models_dir: Path | None = None,
) -> ReportData:
    """Reúne desde las bases todo lo que lleva el reporte de una corrida.

    Args:
        run_id: corrida entrenada de la que se arma el reporte.
        include_predictions: si se incluyen las predicciones guardadas.
        db_path: ruta de ``structured.db``; por defecto la de ``config``.
        unstructured_path: ruta de ``unstructured.db``; por defecto la de
            ``config``.
        models_dir: carpeta de modelos; por defecto ``config.MODELS_DIR``.

    Returns:
        Un ``ReportData``.

    Raises:
        ValueError: si la corrida no existe, no tiene resultados o falta el ETL.
    """
    db.init_structured_db(db_path)
    run = runs.get_run(run_id)
    if run is None:
        raise ValueError(f"No existe la corrida {run_id}")
    table = queries.results_table(run_id, unstructured_path)
    if table.empty:
        raise ValueError(f"La corrida {run_id} no tiene resultados: primero entrenala.")
    overview = queries.data_overview(db_path)
    if overview is None:
        raise ValueError("No hay datos limpios: primero ejecutá el ETL.")
    config_doc = results.load_run_config(run_id, unstructured_path)
    docs = queries.result_docs(run_id, unstructured_path)
    noise = queries.noise_summary(table, docs)
    notes = traceability_notes(config_doc)
    conclusions = {
        scenario: _conclusion(
            run_id, scenario, table, noise, notes, unstructured_path, models_dir
        )
        for scenario in sorted(table["scenario"].unique())
    }
    predictions = (
        queries.prediction_docs(run_id, unstructured_path)
        if include_predictions
        else []
    )
    return ReportData(
        run_id=run_id,
        generated_at=runs.now_iso(),
        run=dict(run),
        config=config_doc,
        overview=overview,
        table=table,
        docs=docs,
        noise=noise,
        conclusions=conclusions,
        predictions=predictions,
        notes=notes,
        predictions_included=include_predictions,
    )
