"""Validación, búsqueda de hiperparámetros, evaluación y guardado de modelos."""

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from src import config, features, models
from src.data import DATE_COLUMN
from src.logger import log_step

logger = logging.getLogger(__name__)

BASELINE_ALWAYS_HIGH = "siempre_alta"
BASELINE_RANDOM = "aleatorio_proporcional"

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
    name: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int = config.SEED,
) -> SearchResult:
    """Busca los mejores hiperparámetros con ``TimeSeriesSplit`` y F1.

    Los datos deben estar ordenados por fecha. Cada corte entrena con el
    pasado y valida con el período siguiente. Al final reajusta el mejor
    pipeline con todo el entrenamiento.

    Args:
        name: uno de ``models.MODEL_NAMES``.
        x_train: variables de entrenamiento, ordenadas por fecha.
        y_train: target de entrenamiento.
        seed: semilla de los modelos; por defecto ``config.SEED``.

    Returns:
        ``SearchResult`` con los mejores parámetros, el F1 por corte (promedio
        y desvío) y el pipeline reajustado.
    """
    search = GridSearchCV(
        models.build_pipeline(name, y_train, seed),
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


@dataclass
class EvalResult:
    """Métricas de un modelo sobre un conjunto de prueba (clase positiva = alta).

    Attributes:
        accuracy: proporción de aciertos.
        precision: de lo que predijo "alta", cuánto lo era.
        recall: de lo que era "alta", cuánto detectó.
        f1: media armónica de precision y recall.
        confusion_matrix: ``[[tn, fp], [fn, tp]]``.
        y_pred: clase predicha de cada mes (0 o 1).
        proba: probabilidad de "alta" de cada mes.
    """

    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: list[list[int]]
    y_pred: list[int]
    proba: list[float]


@log_step
def evaluate(
    estimator: BaseEstimator, x_test: pd.DataFrame, y_test: pd.Series
) -> EvalResult:
    """Evalúa un pipeline ya ajustado sobre datos que no vio (clase 1 = alta).

    No reajusta nada. Si no predice ningún positivo, precision y F1 valen 0.

    Args:
        estimator: pipeline ajustado con ``predict`` y ``predict_proba``.
        x_test: variables del conjunto de prueba.
        y_test: target real del conjunto de prueba.

    Returns:
        ``EvalResult`` con accuracy, precision, recall, F1, matriz de confusión,
        predicciones y probabilidades.
    """
    y_pred = estimator.predict(x_test)
    proba = estimator.predict_proba(x_test)[:, 1]
    result = EvalResult(
        accuracy=float(accuracy_score(y_test, y_pred)),
        precision=float(precision_score(y_test, y_pred, zero_division=0)),
        recall=float(recall_score(y_test, y_pred, zero_division=0)),
        f1=float(f1_score(y_test, y_pred, zero_division=0)),
        confusion_matrix=confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist(),
        y_pred=[int(p) for p in y_pred],
        proba=[float(p) for p in proba],
    )
    logger.info(
        "Evaluación: accuracy %.3f, precision %.3f, recall %.3f, F1 %.3f, "
        "matriz de confusión %s",
        result.accuracy,
        result.precision,
        result.recall,
        result.f1,
        result.confusion_matrix,
    )
    return result


@dataclass
class ModelRun:
    """Resultado completo del entrenamiento de un modelo en un escenario.

    Attributes:
        scenario: ``"A"`` (anticipación) o ``"B"`` (diagnóstico).
        model: nombre del modelo.
        features: variables predictoras usadas.
        n_train: filas de entrenamiento.
        n_test: filas del hold-out.
        search: resultado de la búsqueda (validación cruzada y pipeline).
        test: métricas en el hold-out.
        baselines: métricas de las líneas de referencia en el mismo hold-out.
        test_dates: mes de cada fila del hold-out (``AAAA-MM-DD``).
        y_test: valor real del target en cada fila del hold-out.
    """

    scenario: str
    model: str
    features: list[str]
    n_train: int
    n_test: int
    search: SearchResult
    test: EvalResult
    baselines: dict[str, EvalResult]
    test_dates: list[str]
    y_test: list[int]


@log_step
def evaluate_baselines(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    seed: int = config.SEED,
) -> dict[str, EvalResult]:
    """Evalúa dos líneas de referencia sin información de las variables.

    - ``siempre_alta``: marca todos los meses como "alta".
    - ``aleatorio_proporcional``: sortea la clase según la proporción de
      positivos del entrenamiento (con la semilla dada).

    Sirven para saber si un modelo aporta algo más que el azar o que la regla
    trivial. Se ajustan solo con el entrenamiento.

    Args:
        x_train: variables de entrenamiento.
        y_train: target de entrenamiento.
        x_test: variables del hold-out.
        y_test: target del hold-out.
        seed: semilla del sorteo; por defecto ``config.SEED``.

    Returns:
        Diccionario nombre de referencia a ``EvalResult``.
    """
    references = {
        BASELINE_ALWAYS_HIGH: DummyClassifier(strategy="constant", constant=1),
        BASELINE_RANDOM: DummyClassifier(strategy="stratified", random_state=seed),
    }
    results = {}
    for name, dummy in references.items():
        dummy.fit(x_train, y_train)
        results[name] = evaluate(dummy, x_test, y_test)
    return results


@log_step
def train_model(
    dataset: pd.DataFrame, name: str, scenario: str, seed: int = config.SEED
) -> ModelRun:
    """Entrena y evalúa un modelo en un escenario.

    Separa el hold-out temporal, busca hiperparámetros solo con el
    entrenamiento y evalúa una única vez en el hold-out.

    Args:
        dataset: salida de ``features.build_dataset``.
        name: uno de ``models.MODEL_NAMES``.
        scenario: ``"A"`` o ``"B"``.
        seed: semilla de los modelos; por defecto ``config.SEED``.

    Returns:
        ``ModelRun`` con la búsqueda y la evaluación.
    """
    x, y = features.get_model_data(dataset, scenario)
    x_train, x_test, y_train, y_test = split_holdout(x, y)
    search = search_hyperparameters(name, x_train, y_train, seed)
    test = evaluate(search.estimator, x_test, y_test)
    baselines = evaluate_baselines(x_train, y_train, x_test, y_test, seed)
    logger.info(
        "Resultado %s/%s: F1 validación %.3f ± %.3f, F1 hold-out %.3f "
        "(siempre alta %.3f, aleatorio %.3f)",
        scenario,
        name,
        search.cv_mean,
        search.cv_std,
        test.f1,
        baselines[BASELINE_ALWAYS_HIGH].f1,
        baselines[BASELINE_RANDOM].f1,
    )
    return ModelRun(
        scenario=scenario,
        model=name,
        features=list(x.columns),
        n_train=len(x_train),
        n_test=len(x_test),
        search=search,
        test=test,
        baselines=baselines,
        test_dates=dataset.loc[x_test.index, DATE_COLUMN]
        .dt.strftime("%Y-%m-%d")
        .tolist(),
        y_test=[int(v) for v in y_test],
    )


@log_step
def train_all(
    dataset: pd.DataFrame,
    scenarios: tuple[str, ...] = ("A", "B"),
    model_names: tuple[str, ...] = models.MODEL_NAMES,
    seed: int = config.SEED,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> list[ModelRun]:
    """Entrena todos los modelos en todos los escenarios, con los mismos cortes.

    Args:
        dataset: salida de ``features.build_dataset``.
        scenarios: escenarios a entrenar.
        model_names: modelos a entrenar.
        seed: semilla de los modelos; por defecto ``config.SEED``.
        on_progress: función opcional que se llama al terminar cada modelo con
            ``(terminados, total, "<escenario>/<modelo>")``; sirve para mostrar
            una barra de progreso.

    Returns:
        Un ``ModelRun`` por combinación escenario y modelo.
    """
    combos = [(scenario, name) for scenario in scenarios for name in model_names]
    runs = []
    for done, (scenario, name) in enumerate(combos, start=1):
        runs.append(train_model(dataset, name, scenario, seed))
        if on_progress is not None:
            on_progress(done, len(combos), f"{scenario}/{name}")
    return runs
