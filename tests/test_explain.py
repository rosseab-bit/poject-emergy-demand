"""Tests de la redacción del «por qué» de la recomendación."""

import itertools
import re

import pandas as pd

from src import explain, features, models, recommend
from src.recommend import ModelScore

LR, RF, XGB = "logistic_regression", "random_forest", "xgboost"

# Números reales de la corrida (escenarios A y B).
SCORES_A = [
    ModelScore(LR, 0.444, 0.187, 0.170, 0.4),
    ModelScore(RF, 0.407, 0.216, 0.108, 0.4),
    ModelScore(XGB, 0.340, 0.195, 0.169, 0.4),
]
SCORES_B = [
    ModelScore(LR, 0.489, 0.390, 0.072, 0.4),
    ModelScore(RF, 0.400, 0.183, 0.162, 0.4),
    ModelScore(XGB, 0.392, 0.177, 0.066, 0.4),
]
LR_IMPORTANCE = pd.DataFrame(
    {
        "feature": ["demanda_total_lag12", "grados_dia_calor", "anio", "mes_cos"],
        "importancia": [0.90, -0.55, 0.30, 0.10],
        "tipo": "coeficiente",
    }
)
RF_IMPORTANCE = pd.DataFrame(
    {
        "feature": ["demanda_total_lag1", "temperatura_promedio", "anio"],
        "importancia": [0.30, 0.20, 0.10],
        "tipo": "importancia",
    }
)


def explain_for(scores: list[ModelScore], scenario: str = "A", **kwargs):
    """Evalúa y redacta un escenario."""
    verdict = recommend.evaluate_scenario(scores, scenario)
    return verdict, explain.explain_scenario(verdict, scores, **kwargs)


def test_numbers_use_a_decimal_comma() -> None:
    """Los números se escriben con coma, como en castellano."""
    assert explain.fmt(0.4444) == "0,444"
    assert explain.fmt(0.4444, 2) == "0,44"
    assert explain.fmt(25.0, 0) == "25"


def test_join_names() -> None:
    """Une con comas y «y»."""
    assert explain.join_names([]) == ""
    assert explain.join_names(["A"]) == "A"
    assert explain.join_names(["A", "B"]) == "A y B"
    assert explain.join_names(["A", "B", "C"]) == "A, B y C"


def test_real_scenario_a_says_no_clear_winner_and_not_recommendable() -> None:
    """Con los números reales del A: sin ganador y sin superar la referencia."""
    verdict, text = explain_for(
        SCORES_A, importance=LR_IMPORTANCE, n_test=48, positives=12
    )
    assert verdict.category == recommend.TIE_NO_BEAT
    assert text.headline.startswith(
        "Sin ganador claro entre Regresión logística, Random Forest y XGBoost"
    )
    assert "No hay un modelo recomendable" in text.headline
    assert "Regresión logística por ser el más simple" in text.headline
    body = "\n".join(text.reasons)
    assert "mayor F1 de la clase «alta» es el de Regresión logística (0,444)" in body
    assert (
        "Random Forest (F1 0,407) frente a Regresión logística (F1 0,444): diferencia de 0,037"
        in body
    )
    assert "con un ruido de validación de 0,170" in body
    assert "no se distinguen" in body
    assert "margen de 0,044" in body
    assert "no se puede afirmar que supere a marcar todos los meses como «alta»" in body
    assert "validación cruzada favorecería a Random Forest" in body
    assert "48 meses y solo 12 son de demanda alta" in body
    assert "escenario B" not in body


def test_real_scenario_b_notes_it_is_for_diagnosis_only() -> None:
    """El B supera la referencia, pero el texto aclara que no sirve para anticipar."""
    verdict, text = explain_for(SCORES_B, scenario="B", importance=LR_IMPORTANCE)
    assert verdict.category == recommend.TIE_BEATS
    assert "Se recomienda de forma provisoria Regresión logística" in text.headline
    body = "\n".join(text.reasons)
    assert "XGBoost (F1 0,392)" in body and "queda atrás" in body
    assert "Random Forest (F1 0,400)" in body and "no se distinguen" in body
    assert "lo supera con claridad" in body
    assert "validación cruzada coincide" in body
    assert "sirve para diagnosticar, no para anticipar" in body


def test_clear_winner_headlines() -> None:
    """Ganador claro que supera la referencia y ganador claro que no."""
    strong = [
        ModelScore(LR, 0.80, 0.5, 0.02, 0.4),
        ModelScore(RF, 0.45, 0.4, 0.02, 0.4),
    ]
    verdict, text = explain_for(strong)
    assert verdict.category == recommend.CLEAR_WINNER_BEATS
    assert text.headline.startswith("Se recomienda Regresión logística")
    assert "Sin ganador" not in text.headline
    assert not any("desempata" in r for r in text.reasons)

    weak = [ModelScore(LR, 0.41, 0.4, 0.05, 0.4), ModelScore(RF, 0.2, 0.2, 0.02, 0.4)]
    verdict, text = explain_for(weak)
    assert verdict.category == recommend.CLEAR_WINNER_NO_BEAT
    assert "no se recomienda usarlo para decidir" in text.headline


def test_tiebreak_sentence_depends_on_whether_the_best_is_the_simplest() -> None:
    """Si el recomendado no es el de mayor F1, se aclara que se elige por simplicidad."""
    best_is_complex = [
        ModelScore(XGB, 0.60, 0.4, 0.10, 0.4),
        ModelScore(LR, 0.57, 0.4, 0.10, 0.4),
    ]
    _, text = explain_for(best_is_complex)
    assert any(
        "no por F1: se elige Regresión logística aunque XGBoost" in r
        for r in text.reasons
    )

    best_is_simple = [
        ModelScore(LR, 0.60, 0.4, 0.10, 0.4),
        ModelScore(RF, 0.57, 0.4, 0.10, 0.4),
    ]
    _, text = explain_for(best_is_simple)
    assert any("a la vez el de mayor F1 y el más simple" in r for r in text.reasons)


def test_variables_sentence_for_coefficients_and_importances() -> None:
    """Regresión logística: coeficientes con signo; árboles: importancias y su peso."""
    text = explain.variables_reason(LR, LR_IMPORTANCE)
    assert (
        "la demanda total del mismo mes del año anterior (empuja hacia «alta», coeficiente 0,90)"
        in text
    )
    assert "los grados-día de calor (empuja hacia «normal», coeficiente -0,55)" in text
    assert (
        "el año (empuja hacia «alta»" in text and "mes_cos" not in text
    )  # solo las 3 primeras
    assert "estandarizadas" in text

    text = explain.variables_reason(RF, RF_IMPORTANCE)
    assert (
        "Random Forest usa sobre todo la demanda total del mes anterior (0,30)" in text
    )
    assert "concentran el 60 % de la importancia" in text
    assert explain.variables_reason(RF, None) is None
    assert explain.variables_reason(RF, RF_IMPORTANCE.iloc[0:0]) is None


def test_optional_sentences_are_skipped_without_their_inputs() -> None:
    """Sin importancias ni tamaño de muestra, esas oraciones no aparecen."""
    _, text = explain_for(SCORES_A)
    body = "\n".join(text.reasons)
    assert "pesan más" not in body and "usa sobre todo" not in body
    assert "meses y solo" not in body


def test_text_is_consistent_with_the_verdict_for_every_combination() -> None:
    """Para muchas combinaciones, el texto nunca contradice a la regla."""
    cases = itertools.product((0.01, 0.30), (0.41, 0.90), (0.0, 0.3))
    for std, top, gap in cases:
        scores = [
            ModelScore(LR, top, 0.3, std, 0.4),
            ModelScore(RF, top - gap, 0.3, std, 0.4),
            ModelScore(XGB, 0.2, 0.3, std, 0.4),
        ]
        verdict, text = explain_for(
            scores, importance=LR_IMPORTANCE, n_test=48, positives=12
        )
        body = "\n".join(text.reasons)
        chosen = models.model_label(verdict.recommended)
        assert chosen in text.headline
        if verdict.beats_reference:
            assert (
                "lo supera con claridad" in body and "no se puede afirmar" not in body
            )
        else:
            assert (
                "no se puede afirmar" in body and "lo supera con claridad" not in body
            )
        if verdict.clear_winner:
            assert "Sin ganador claro" not in text.headline
            assert "desempata" not in body
        else:
            assert "Sin ganador claro" in text.headline
            assert "desempata" in body
        if verdict.category == recommend.TIE_NO_BEAT:
            assert "No hay un modelo recomendable" in text.headline
        else:
            assert "No hay un modelo recomendable" not in text.headline
        # ningún número con punto decimal: todos con coma
        assert not re.search(r"\d\.\d", text.headline + body)


def test_every_variable_and_model_has_a_readable_label() -> None:
    """Todas las variables y modelos tienen nombre legible."""
    for name in features.FEATURES_B:
        assert features.feature_label(name) != name
    assert features.feature_label("otra") == "otra"
    for name in models.MODEL_NAMES:
        assert models.model_label(name) != name
    assert models.model_label("otro") == "otro"


def test_small_coefficients_are_called_out() -> None:
    """Si todos los coeficientes son chicos, el texto lo dice; con uno grande, no."""
    tiny = pd.DataFrame(
        {
            "feature": ["anio", "grados_dia_frio", "mes_cos"],
            "importancia": [-0.12, 0.11, 0.05],
            "tipo": "coeficiente",
        }
    )
    text = explain.variables_reason(LR, tiny)
    assert "Todos son chicos (el mayor, en valor absoluto, es 0,12)" in text
    assert "casi no separa unos meses de otros" in text
    assert "Todos son chicos" not in explain.variables_reason(LR, LR_IMPORTANCE)
    just_below = tiny.assign(importancia=[0.24, 0.1, 0.0])
    just_above = tiny.assign(importancia=[0.25, 0.1, 0.0])
    assert "Todos son chicos" in explain.variables_reason(LR, just_below)
    assert "Todos son chicos" not in explain.variables_reason(LR, just_above)


def test_small_coefficient_note_never_applies_to_tree_importances() -> None:
    """Las importancias de los árboles (suman 1) no llevan esa aclaración."""
    assert "Todos son chicos" not in explain.variables_reason(RF, RF_IMPORTANCE)


def test_variable_labels_do_not_create_nested_parentheses() -> None:
    """Ninguna etiqueta de variable lleva paréntesis (se usan dentro de uno)."""
    for name in features.FEATURES_B:
        assert "(" not in features.feature_label(name)
