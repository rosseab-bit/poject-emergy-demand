"""Regla explícita para recomendar un modelo a partir de las métricas.

Con ~236 meses con target y un hold-out de 48 (12 de demanda alta), las
métricas son ruidosas. La regla no declara un ganador si la diferencia entre
modelos está dentro del ruido de la validación cruzada, y exige además que el
modelo recomendado supere a la regla trivial «siempre alta».
"""

import math
from dataclasses import dataclass

import pandas as pd

from src import models

CLEAR_WINNER_BEATS = "ganador_claro_supera"
CLEAR_WINNER_NO_BEAT = "ganador_claro_no_supera"
TIE_BEATS = "sin_ganador_supera"
TIE_NO_BEAT = "sin_ganador_no_supera"
CATEGORIES = (CLEAR_WINNER_BEATS, CLEAR_WINNER_NO_BEAT, TIE_BEATS, TIE_NO_BEAT)


@dataclass(frozen=True)
class ModelScore:
    """Métricas de un modelo en un escenario.

    Attributes:
        model: nombre del modelo.
        f1: F1 de la clase "alta" en el hold-out.
        cv_mean: F1 promedio en la validación cruzada.
        cv_std: desvío del F1 entre los cortes de validación.
        reference_f1: F1 de la regla «siempre alta» en el mismo hold-out.
    """

    model: str
    f1: float
    cv_mean: float
    cv_std: float
    reference_f1: float


@dataclass(frozen=True)
class Comparison:
    """Comparación de un modelo contra el de mayor F1 en el hold-out.

    Attributes:
        model: modelo comparado.
        difference: F1 del mejor menos F1 de este modelo (0 para el mejor).
        noise: mayor de los dos desvíos de validación cruzada.
        within_noise: si la diferencia no se distingue del ruido.
    """

    model: str
    difference: float
    noise: float
    within_noise: bool


@dataclass(frozen=True)
class Verdict:
    """Conclusión de la regla para un escenario.

    Attributes:
        scenario: ``"A"`` o ``"B"``.
        best: modelo de mayor F1 en el hold-out.
        candidates: modelos que no se distinguen del mejor, del más simple al
            más complejo.
        clear_winner: ``True`` si el mejor es el único candidato.
        recommended: modelo elegido: el ganador claro o, si hay empate, el más
            simple entre los candidatos.
        margin: F1 del recomendado menos el de la referencia «siempre alta».
        beats_reference: si el margen supera el desvío del recomendado.
        category: una de ``CATEGORIES``.
        recommend_for_use: si el informe recomienda usar un modelo (el ganador
            claro o el provisorio por simplicidad que supera la referencia).
        cv_best: modelo de mayor F1 promedio en validación cruzada.
        cv_agrees: si ``cv_best`` coincide con el mejor del hold-out.
        comparisons: detalle de cada modelo contra el mejor.
    """

    scenario: str
    best: str
    candidates: tuple[str, ...]
    clear_winner: bool
    recommended: str
    margin: float
    beats_reference: bool
    category: str
    recommend_for_use: bool
    cv_best: str
    cv_agrees: bool
    comparisons: tuple[Comparison, ...]


def complexity_rank(model: str) -> int:
    """Posición de un modelo de más simple a más complejo; los desconocidos, al final."""
    names = models.MODEL_NAMES
    return names.index(model) if model in names else len(names)


def noise_threshold(first: ModelScore, second: ModelScore) -> float:
    """Ruido de una comparación: el mayor de los dos desvíos de validación."""
    return max(first.cv_std, second.cv_std)


def is_within_noise(difference: float, noise: float) -> bool:
    """Una diferencia no se distingue del ruido si es menor que él.

    Dos puntajes idénticos (diferencia 0) siempre empatan, incluso con ruido 0.
    """
    return difference == 0 or difference < noise


def _validate(scores: list[ModelScore]) -> None:
    """Verifica que haya modelos, sin repetir y con métricas numéricas."""
    if not scores:
        raise ValueError("No hay modelos para comparar.")
    names = [s.model for s in scores]
    if len(set(names)) != len(names):
        raise ValueError(f"Modelos repetidos: {names}")
    for score in scores:
        values = (score.f1, score.cv_mean, score.cv_std, score.reference_f1)
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f"Métricas no numéricas en {score.model}: {score}")


def beats_reference(score: ModelScore) -> bool:
    """Un modelo supera a «siempre alta» si el margen es mayor que su propio desvío."""
    return score.f1 - score.reference_f1 > score.cv_std


def _category(clear_winner: bool, beats_reference: bool) -> str:
    """Categoría de la conclusión según las dos preguntas de la regla."""
    if clear_winner:
        return CLEAR_WINNER_BEATS if beats_reference else CLEAR_WINNER_NO_BEAT
    return TIE_BEATS if beats_reference else TIE_NO_BEAT


def evaluate_scenario(scores: list[ModelScore], scenario: str) -> Verdict:
    """Aplica la regla de recomendación a los modelos de un escenario.

    1. El mejor es el de mayor F1 en el hold-out.
    2. Otro modelo empata con el mejor si la diferencia de F1 es menor que el
       mayor de los dos desvíos de validación cruzada.
    3. Con un solo candidato hay ganador claro; con varios, no, y se elige el
       más simple (regresión logística, Random Forest, XGBoost).
    4. El elegido solo se recomienda si supera a «siempre alta» por más que su
       propio desvío.

    Args:
        scores: métricas de cada modelo en el escenario.
        scenario: ``"A"`` o ``"B"``.

    Returns:
        El ``Verdict`` con la conclusión y el detalle de cada comparación.

    Raises:
        ValueError: si no hay modelos, se repiten o alguna métrica no es un
            número finito.
    """
    _validate(scores)
    ordered = sorted(scores, key=lambda s: complexity_rank(s.model))
    best = max(ordered, key=lambda s: (s.f1, -complexity_rank(s.model)))
    comparisons = []
    for score in ordered:
        difference = best.f1 - score.f1
        noise = noise_threshold(best, score)
        within = score.model == best.model or is_within_noise(difference, noise)
        comparisons.append(Comparison(score.model, difference, noise, within))
    candidates = tuple(c.model for c in comparisons if c.within_noise)
    clear_winner = len(candidates) == 1
    chosen = next(s for s in ordered if s.model == candidates[0])
    margin = chosen.f1 - chosen.reference_f1
    beats = beats_reference(chosen)
    cv_best = max(ordered, key=lambda s: (s.cv_mean, -complexity_rank(s.model)))
    return Verdict(
        scenario=scenario,
        best=best.model,
        candidates=candidates,
        clear_winner=clear_winner,
        recommended=chosen.model,
        margin=margin,
        beats_reference=beats,
        category=_category(clear_winner, beats),
        recommend_for_use=beats,
        cv_best=cv_best.model,
        cv_agrees=cv_best.model == best.model,
        comparisons=tuple(comparisons),
    )


def scores_from_table(table: pd.DataFrame, scenario: str) -> list[ModelScore]:
    """Convierte las filas de un escenario de ``queries.results_table``.

    Args:
        table: salida de ``queries.results_table``.
        scenario: ``"A"`` o ``"B"``.

    Returns:
        Un ``ModelScore`` por modelo del escenario.

    Raises:
        ValueError: si el escenario no tiene resultados.
    """
    rows = table[table["scenario"] == scenario]
    if rows.empty:
        raise ValueError(f"No hay resultados del escenario {scenario}.")
    return [
        ModelScore(
            model=row.model,
            f1=float(row.f1),
            cv_mean=float(row.cv_mean),
            cv_std=float(row.cv_std),
            reference_f1=float(row.f1_siempre_alta),
        )
        for row in rows.itertuples()
    ]
