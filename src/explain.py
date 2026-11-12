"""Redacción del «por qué» de la recomendación, con reglas explícitas.

Cada oración sale de una condición sobre las métricas y las variables; no hay
párrafos fijos. Así el texto nunca puede contradecir a ``recommend``.
"""

from dataclasses import dataclass

import pandas as pd

from src import features, recommend
from src.models import model_label
from src.recommend import ModelScore, Verdict

TOP_VARIABLES = 3
# Un coeficiente (sobre variables estandarizadas) menor que esto casi no mueve la
# predicción; si todos lo son, el modelo casi no separa unos meses de otros.
SMALL_COEFFICIENT = 0.25


@dataclass(frozen=True)
class Explanation:
    """Texto de la conclusión de un escenario.

    Attributes:
        headline: la conclusión en una oración.
        reasons: las razones, una oración por regla que se cumple.
    """

    headline: str
    reasons: tuple[str, ...]


def fmt(value: float, digits: int = 3) -> str:
    """Formatea un número con coma decimal, como se escribe en castellano."""
    return f"{value:.{digits}f}".replace(".", ",")


def join_names(names: list[str]) -> str:
    """Une nombres con comas y «y»: ``A``, ``A y B`` o ``A, B y C``."""
    if len(names) <= 1:
        return "".join(names)
    return f"{', '.join(names[:-1])} y {names[-1]}"


def headline(verdict: Verdict) -> str:
    """La conclusión del escenario según su categoría."""
    chosen = model_label(verdict.recommended)
    if verdict.category == recommend.CLEAR_WINNER_BEATS:
        return (
            f"Se recomienda {chosen}: se distingue de los otros modelos por más "
            "que el ruido de la validación y supera a la regla «siempre alta»."
        )
    if verdict.category == recommend.CLEAR_WINNER_NO_BEAT:
        return (
            f"{chosen} se distingue de los otros modelos, pero no supera con "
            "claridad a la regla «siempre alta»: no se recomienda usarlo para decidir."
        )
    candidates = join_names([model_label(m) for m in verdict.candidates])
    if verdict.category == recommend.TIE_BEATS:
        return (
            f"Sin ganador claro entre {candidates}. Se recomienda de forma "
            f"provisoria {chosen}, el más simple, que supera a la regla «siempre alta»."
        )
    return (
        f"Sin ganador claro entre {candidates}, y ningún modelo supera con "
        "claridad a la regla «siempre alta». No hay un modelo recomendable; si "
        f"se usa uno, {chosen} por ser el más simple."
    )


def ranking_reason(verdict: Verdict, scores: dict[str, ModelScore]) -> str:
    """Regla 1: quién tiene el mayor F1 en el hold-out."""
    best = scores[verdict.best]
    return (
        f"En el hold-out, el mayor F1 de la clase «alta» es el de "
        f"{model_label(verdict.best)} ({fmt(best.f1)})."
    )


def noise_reasons(verdict: Verdict, scores: dict[str, ModelScore]) -> list[str]:
    """Regla 2: cada otro modelo, contra el ruido de la validación cruzada."""
    best = scores[verdict.best]
    reasons = []
    for comparison in verdict.comparisons:
        if comparison.model == verdict.best:
            continue
        other = scores[comparison.model]
        name = model_label(comparison.model)
        pair = (
            f"{name} (F1 {fmt(other.f1)}) frente a {model_label(verdict.best)} "
            f"(F1 {fmt(best.f1)}): diferencia de {fmt(comparison.difference)}, "
            f"con un ruido de validación de {fmt(comparison.noise)}"
        )
        if comparison.within_noise:
            reasons.append(
                f"{pair}. La diferencia es menor que el ruido: no se distinguen."
            )
        else:
            reasons.append(
                f"{pair}. La diferencia supera el ruido: {name} queda atrás."
            )
    return reasons


def tiebreak_reason(verdict: Verdict) -> str | None:
    """Regla 3: por qué se elige el modelo recomendado cuando hay empate."""
    if verdict.clear_winner:
        return None
    chosen = model_label(verdict.recommended)
    if verdict.recommended == verdict.best:
        return (
            f"Con varios modelos empatados se desempata por simplicidad; "
            f"{chosen} es a la vez el de mayor F1 y el más simple."
        )
    return (
        f"Con varios modelos empatados se desempata por simplicidad, no por F1: se "
        f"elige {chosen} aunque {model_label(verdict.best)} tenga un F1 algo mayor."
    )


def reference_reason(verdict: Verdict, scores: dict[str, ModelScore]) -> str:
    """Regla 4: el modelo elegido contra la regla trivial «siempre alta»."""
    chosen = scores[verdict.recommended]
    name = model_label(verdict.recommended)
    comparison = (
        f"El F1 de {name} ({fmt(chosen.f1)}) frente al de «siempre alta» "
        f"({fmt(chosen.reference_f1)}) da un margen de {fmt(verdict.margin)}, "
        f"con un desvío de validación de {fmt(chosen.cv_std)}."
    )
    if verdict.beats_reference:
        return f"{comparison} El margen es mayor que el desvío: lo supera con claridad."
    return (
        f"{comparison} El margen no es mayor que el desvío: no se puede afirmar que "
        "supere a marcar todos los meses como «alta»."
    )


def cross_validation_reason(verdict: Verdict, scores: dict[str, ModelScore]) -> str:
    """Regla 5: si la validación cruzada coincide con el ranking del hold-out."""
    cv_best = scores[verdict.cv_best]
    if verdict.cv_agrees:
        return (
            f"La validación cruzada coincide: {model_label(verdict.cv_best)} también "
            f"tiene el mayor F1 medio ({fmt(cv_best.cv_mean)})."
        )
    return (
        f"La validación cruzada favorecería a {model_label(verdict.cv_best)} (F1 "
        f"medio {fmt(cv_best.cv_mean)}), no a {model_label(verdict.best)}: otra señal "
        "de que la diferencia en el hold-out es frágil."
    )


def variables_reason(model: str, importance: pd.DataFrame) -> str | None:
    """Regla 6: las variables con más peso en el modelo recomendado."""
    if importance is None or importance.empty:
        return None
    top = importance.head(TOP_VARIABLES)
    name = model_label(model)
    if (top["tipo"] == "coeficiente").all():
        parts = [
            f"{features.feature_label(row.feature)} "
            f"({'empuja hacia «alta»' if row.importancia > 0 else 'empuja hacia «normal»'}, "
            f"coeficiente {fmt(row.importancia, 2)})"
            for row in top.itertuples()
        ]
        text = (
            f"En {name} pesan más {join_names(parts)}. Los coeficientes son sobre "
            "variables estandarizadas."
        )
        largest = float(importance["importancia"].abs().max())
        if largest < SMALL_COEFFICIENT:
            text += (
                f" Todos son chicos (el mayor, en valor absoluto, es {fmt(largest, 2)}): "
                "el modelo casi no separa unos meses de otros."
            )
        return text
    parts = [
        f"{features.feature_label(row.feature)} ({fmt(row.importancia, 2)})"
        for row in top.itertuples()
    ]
    share = float(top["importancia"].sum())
    return (
        f"{name} usa sobre todo {join_names(parts)}; las {len(top)} variables más "
        f"usadas concentran el {fmt(100 * share, 0)} % de la importancia."
    )


def sample_reason(n_test: int, positives: int) -> str:
    """Regla 7: cuántos casos hay en el hold-out."""
    return (
        f"El hold-out tiene {n_test} meses y solo {positives} son de demanda alta: "
        "un acierto de más o de menos cambia mucho el F1."
    )


def scenario_reason(scenario: str) -> str | None:
    """Regla 8: el escenario B no sirve para anticipar."""
    if scenario == "B":
        return (
            "El escenario B usa la potencia máxima, que ocurre en el mismo mes que la "
            "demanda y no se conoce de antemano: sirve para diagnosticar, no para "
            "anticipar. La recomendación para predecir sale del escenario A."
        )
    return None


def explain_scenario(
    verdict: Verdict,
    scores: list[ModelScore],
    importance: pd.DataFrame | None = None,
    n_test: int | None = None,
    positives: int | None = None,
) -> Explanation:
    """Redacta la conclusión de un escenario y sus razones.

    Args:
        verdict: salida de ``recommend.evaluate_scenario``.
        scores: los puntajes con los que se calculó el veredicto.
        importance: coeficientes o importancias del modelo recomendado (salida
            de ``results.feature_importance``); opcional.
        n_test: meses del hold-out; opcional.
        positives: meses de demanda alta en el hold-out; opcional.

    Returns:
        La conclusión y las razones, una oración por regla que se cumple.
    """
    by_model = {score.model: score for score in scores}
    reasons = [ranking_reason(verdict, by_model), *noise_reasons(verdict, by_model)]
    optional = [
        tiebreak_reason(verdict),
        reference_reason(verdict, by_model),
        cross_validation_reason(verdict, by_model),
        variables_reason(verdict.recommended, importance),
        sample_reason(n_test, positives) if n_test and positives is not None else None,
        scenario_reason(verdict.scenario),
    ]
    reasons.extend(text for text in optional if text)
    return Explanation(headline(verdict), tuple(reasons))
