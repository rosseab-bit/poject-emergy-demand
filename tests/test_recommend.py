"""Tests de la regla de recomendación de modelo."""

import random

import pandas as pd
import pytest

from src import recommend
from src.recommend import ModelScore

LR, RF, XGB = "logistic_regression", "random_forest", "xgboost"


def score(model: str, f1: float, std: float = 0.05, ref: float = 0.4, cv: float = 0.3):
    """Atajo para armar el puntaje de un modelo."""
    return ModelScore(model, f1=f1, cv_mean=cv, cv_std=std, reference_f1=ref)


def test_clear_winner_beating_the_reference() -> None:
    """Un modelo que gana por más que el ruido y supera la referencia se recomienda."""
    verdict = recommend.evaluate_scenario(
        [score(LR, 0.70, 0.02), score(RF, 0.45, 0.02), score(XGB, 0.40, 0.02)], "A"
    )
    assert verdict.clear_winner is True
    assert verdict.candidates == (LR,)
    assert verdict.recommended == LR
    assert verdict.margin == pytest.approx(0.30)
    assert verdict.beats_reference is True
    assert verdict.category == recommend.CLEAR_WINNER_BEATS
    assert verdict.recommend_for_use is True


def test_clear_winner_does_not_have_to_be_the_simplest_model() -> None:
    """Si el mejor es XGBoost por más que el ruido, se recomienda XGBoost."""
    verdict = recommend.evaluate_scenario(
        [score(LR, 0.45, 0.02), score(RF, 0.50, 0.02), score(XGB, 0.80, 0.02)], "A"
    )
    assert (verdict.best, verdict.recommended, verdict.clear_winner) == (XGB, XGB, True)


def test_tie_within_the_noise_has_no_clear_winner_and_picks_the_simplest() -> None:
    """Con diferencias dentro del ruido no hay ganador y se elige el más simple."""
    verdict = recommend.evaluate_scenario(
        [score(XGB, 0.60, 0.10), score(LR, 0.57, 0.10), score(RF, 0.30, 0.05)], "A"
    )
    assert verdict.best == XGB
    assert verdict.candidates == (LR, XGB)  # RF queda atrás: 0,30 de diferencia
    assert verdict.clear_winner is False
    assert verdict.recommended == LR  # el más simple del empate, no el mejor
    assert verdict.category == recommend.TIE_BEATS
    assert verdict.recommend_for_use is True


def test_clear_winner_that_does_not_beat_the_reference_is_not_recommended() -> None:
    """Ganar entre modelos no alcanza si no se supera a «siempre alta»."""
    verdict = recommend.evaluate_scenario(
        [
            score(LR, 0.41, 0.05, ref=0.40),
            score(RF, 0.20, 0.02),
            score(XGB, 0.15, 0.02),
        ],
        "A",
    )
    assert verdict.clear_winner is True
    assert verdict.beats_reference is False
    assert verdict.category == recommend.CLEAR_WINNER_NO_BEAT
    assert verdict.recommend_for_use is False


def test_tie_and_reference_not_beaten() -> None:
    """Empate y sin superar la referencia: no hay modelo recomendable."""
    verdict = recommend.evaluate_scenario(
        [score(LR, 0.42, 0.15), score(RF, 0.40, 0.15), score(XGB, 0.38, 0.15)], "A"
    )
    assert verdict.candidates == (LR, RF, XGB)
    assert verdict.category == recommend.TIE_NO_BEAT
    assert verdict.recommend_for_use is False
    assert verdict.recommended == LR  # el que se usaría igual, por simplicidad


def test_real_scenario_a_results() -> None:
    """Con los números reales del escenario A: los tres empatan y ninguno supera la referencia."""
    verdict = recommend.evaluate_scenario(
        [
            score(LR, 0.444, 0.170, cv=0.187),
            score(RF, 0.407, 0.108, cv=0.216),
            score(XGB, 0.340, 0.169, cv=0.195),
        ],
        "A",
    )
    assert verdict.best == LR
    assert verdict.candidates == (LR, RF, XGB)
    assert verdict.clear_winner is False
    assert verdict.recommended == LR
    assert verdict.margin == pytest.approx(0.044)
    assert verdict.beats_reference is False
    assert verdict.category == recommend.TIE_NO_BEAT
    assert verdict.cv_best == RF and verdict.cv_agrees is False


def test_real_scenario_b_results() -> None:
    """Escenario B real: XGBoost queda atrás; LR y RF empatan; LR supera la referencia."""
    verdict = recommend.evaluate_scenario(
        [
            score(LR, 0.489, 0.072, cv=0.390),
            score(RF, 0.400, 0.162, cv=0.183),
            score(XGB, 0.392, 0.066, cv=0.177),
        ],
        "B",
    )
    assert verdict.candidates == (LR, RF)
    by_model = {c.model: c for c in verdict.comparisons}
    assert by_model[XGB].within_noise is False
    assert by_model[XGB].noise == pytest.approx(0.072)  # el mayor de 0,072 y 0,066
    assert by_model[RF].within_noise is True
    assert verdict.recommended == LR
    assert verdict.beats_reference is True
    assert verdict.category == recommend.TIE_BEATS
    assert verdict.cv_agrees is True


def test_noise_is_the_larger_of_the_two_deviations() -> None:
    """Un mejor muy estable no basta: si el otro es muy ruidoso, empatan."""
    verdict = recommend.evaluate_scenario(
        [score(LR, 0.60, 0.01), score(RF, 0.45, 0.30)], "A"
    )
    assert verdict.candidates == (LR, RF)
    assert (
        recommend.noise_threshold(score(LR, 0.6, 0.01), score(RF, 0.45, 0.30)) == 0.30
    )


def test_difference_equal_to_the_noise_is_not_a_tie() -> None:
    """La regla dice «menor que» el desvío: una diferencia igual al ruido no empata."""
    assert recommend.is_within_noise(0.09, 0.10) is True
    assert recommend.is_within_noise(0.10, 0.10) is False
    assert recommend.is_within_noise(0.11, 0.10) is False


def test_identical_scores_always_tie_even_with_zero_noise() -> None:
    """Dos puntajes idénticos empatan aunque el ruido sea 0."""
    assert recommend.is_within_noise(0.0, 0.0) is True
    verdict = recommend.evaluate_scenario(
        [score(XGB, 0.5, 0.0), score(LR, 0.5, 0.0)], "A"
    )
    assert verdict.best == LR  # con F1 igual, gana el más simple
    assert verdict.candidates == (LR, XGB)


def test_single_model_is_a_clear_winner_by_default() -> None:
    """Con un solo modelo no hay con quién empatar."""
    verdict = recommend.evaluate_scenario([score(RF, 0.7, 0.02)], "A")
    assert (verdict.clear_winner, verdict.recommended, verdict.candidates) == (
        True,
        RF,
        (RF,),
    )


def test_result_does_not_depend_on_the_input_order() -> None:
    """Mezclar los modelos de entrada no cambia la conclusión."""
    scores = [score(LR, 0.55, 0.1), score(RF, 0.50, 0.1), score(XGB, 0.20, 0.1)]
    expected = recommend.evaluate_scenario(scores, "A")
    for seed in range(5):
        shuffled = scores[:]
        random.Random(seed).shuffle(shuffled)
        assert recommend.evaluate_scenario(shuffled, "A") == expected


def test_cv_ranking_is_reported_when_it_disagrees_with_the_holdout() -> None:
    """Se informa si la validación cruzada elegiría a otro modelo."""
    verdict = recommend.evaluate_scenario(
        [score(LR, 0.60, 0.1, cv=0.20), score(RF, 0.55, 0.1, cv=0.35)], "A"
    )
    assert (verdict.best, verdict.cv_best, verdict.cv_agrees) == (LR, RF, False)


def test_every_verdict_has_a_known_category_and_consistent_flags() -> None:
    """La categoría es coherente con las dos preguntas de la regla."""
    for std in (0.01, 0.3):
        for top in (0.41, 0.9):
            verdict = recommend.evaluate_scenario(
                [score(LR, top, std), score(RF, 0.3, std), score(XGB, 0.2, std)], "A"
            )
            assert verdict.category in recommend.CATEGORIES
            assert verdict.recommend_for_use == verdict.beats_reference
            assert verdict.clear_winner == (len(verdict.candidates) == 1)
            assert verdict.recommended in verdict.candidates


def test_invalid_inputs_are_rejected() -> None:
    """Sin modelos, con repetidos o con métricas no numéricas, falla."""
    with pytest.raises(ValueError, match="No hay modelos"):
        recommend.evaluate_scenario([], "A")
    with pytest.raises(ValueError, match="repetidos"):
        recommend.evaluate_scenario([score(LR, 0.5), score(LR, 0.6)], "A")
    with pytest.raises(ValueError, match="no numéricas"):
        recommend.evaluate_scenario([score(LR, float("nan"))], "A")
    with pytest.raises(ValueError, match="no numéricas"):
        recommend.evaluate_scenario([score(LR, 0.5, float("inf"))], "A")


def test_unknown_models_sort_after_the_known_ones() -> None:
    """Un modelo desconocido se considera el más complejo."""
    assert recommend.complexity_rank(LR) < recommend.complexity_rank(RF)
    assert recommend.complexity_rank(RF) < recommend.complexity_rank(XGB)
    assert recommend.complexity_rank("otro") > recommend.complexity_rank(XGB)


def test_scores_from_table_reads_one_scenario() -> None:
    """Convierte las filas de un escenario de la tabla de resultados."""
    table = pd.DataFrame(
        {
            "scenario": ["A", "A", "B"],
            "model": [LR, RF, LR],
            "f1": [0.44, 0.40, 0.49],
            "cv_mean": [0.18, 0.21, 0.39],
            "cv_std": [0.17, 0.10, 0.07],
            "f1_siempre_alta": [0.4, 0.4, 0.4],
        }
    )
    scores = recommend.scores_from_table(table, "A")
    assert [s.model for s in scores] == [LR, RF]
    assert scores[0] == ModelScore(LR, 0.44, 0.18, 0.17, 0.4)
    with pytest.raises(ValueError, match="escenario C"):
        recommend.scores_from_table(table, "C")
