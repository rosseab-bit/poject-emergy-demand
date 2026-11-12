"""Validación, búsqueda de hiperparámetros, evaluación y guardado de modelos."""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from src import config, models
from src.logger import log_step

logger = logging.getLogger(__name__)

# F1 de la clase "alta" (1); si no predice positivos vale 0, sin advertencias.
F1_SCORER = make_scorer(f1_score, pos_label=1, zero_division=0)


def split_holdout(
    x: pd.DataFrame, y: pd.Series, holdout: int = config.HOLDOUT_MONTHS
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Separa los últimos ``holdout`` meses como conjunto de prueba final.

    El corte es temporal: el entrenamiento es siempre anterior a la prueba.
    ``x`` e ``y`` deben estar ordenados por fecha, como los devuelve
    ``features.get_model_data``.

    Args:
        x: variables predictoras, ordenadas por fecha.
        y: target, con el mismo orden e índice que ``x``.
        holdout: cantidad de meses finales para la prueba.

    Returns:
        Tupla ``(x_train, x_test, y_train, y_test)``.

    Raises:
        ValueError: si ``x`` e ``y`` no coinciden o ``holdout`` no deja datos
            de entrenamiento.
    """
    if not x.index.equals(y.index):
        raise ValueError("x e y deben tener el mismo índice.")
    if not 0 < holdout < len(x):
        raise ValueError(f"holdout inválido: {holdout} (hay {len(x)} filas).")
    return x.iloc[:-holdout], x.iloc[-holdout:], y.iloc[:-holdout], y.iloc[-holdout:]


@dataclass
class SearchResult:
    """Resultado de la búsqueda de hiperparámetros de un modelo.

    Attributes:
        model: nombre del modelo.
        best_params: mejores hiperparámetros según el F1 de validación.
        cv_scores: F1 del mejor modelo en cada corte de validación.
        cv_mean: promedio de ``cv_scores``.
        cv_std: desvío muestral (``ddof=1``) de ``cv_scores``.
        estimator: pipeline reajustado con todo el entrenamiento.
    """

    model: str
    best_params: dict
    cv_scores: list[float]
    cv_mean: float
    cv_std: float
    estimator: BaseEstimator


def make_cv() -> TimeSeriesSplit:
    """Crea los cortes de validación temporal, iguales para los tres modelos."""
    return TimeSeriesSplit(n_splits=config.CV_SPLITS)


@log_step
def search_hyperparameters(
    name: str, x_train: pd.DataFrame, y_train: pd.Series
) -> SearchResult:
    """Busca los mejores hiperparámetros con ``TimeSeriesSplit`` y F1.

    Los datos deben estar ordenados por fecha. Cada corte entrena con el
    pasado y valida con el período siguiente. Al final reajusta el mejor
    pipeline con todo el entrenamiento.

    Args:
        name: uno de ``models.MODEL_NAMES``.
        x_train: variables de entrenamiento, ordenadas por fecha.
        y_train: target de entrenamiento.

    Returns:
        ``SearchResult`` con los mejores parámetros, el F1 por corte (promedio
        y desvío) y el pipeline reajustado.
    """
    search = GridSearchCV(
        models.build_pipeline(name, y_train),
        models.get_param_grid(name),
        scoring=F1_SCORER,
        cv=make_cv(),
        refit=True,
        n_jobs=1,
    )
    search.fit(x_train, y_train)
    best = search.best_index_
    scores = [
        float(search.cv_results_[f"split{k}_test_score"][best])
        for k in range(config.CV_SPLITS)
    ]
    result = SearchResult(
        model=name,
        best_params=dict(search.best_params_),
        cv_scores=scores,
        cv_mean=float(np.mean(scores)),
        cv_std=float(np.std(scores, ddof=1)),
        estimator=search.best_estimator_,
    )
    logger.info(
        "Búsqueda %s: F1 de validación %.3f ± %.3f, parámetros %s",
        name,
        result.cv_mean,
        result.cv_std,
        result.best_params,
    )
    return result
