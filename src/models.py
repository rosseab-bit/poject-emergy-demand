"""Pipelines de scikit-learn y grillas de hiperparámetros de los tres modelos.

Todo lo que aprende de los datos (imputación, escalado) vive dentro del
``Pipeline``, así se ajusta solo con los datos de entrenamiento.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src import config

LOGISTIC_REGRESSION = "logistic_regression"
RANDOM_FOREST = "random_forest"
XGBOOST = "xgboost"
# Del más simple al más complejo (se usa para desempatar).
MODEL_NAMES = (LOGISTIC_REGRESSION, RANDOM_FOREST, XGBOOST)

MODEL_LABELS = {
    LOGISTIC_REGRESSION: "Regresión logística",
    RANDOM_FOREST: "Random Forest",
    XGBOOST: "XGBoost",
}


def model_label(name: str) -> str:
    """Nombre legible de un modelo; devuelve el mismo texto si no lo conoce."""
    return MODEL_LABELS.get(name, name)


PARAM_GRIDS: dict[str, dict[str, list]] = {
    LOGISTIC_REGRESSION: {"clf__C": [0.01, 0.1, 1.0, 10.0]},
    RANDOM_FOREST: {
        "clf__max_depth": [3, 5, None],
        "clf__min_samples_leaf": [1, 3, 5],
    },
    XGBOOST: {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [2, 3, 4],
        "clf__learning_rate": [0.05, 0.1],
    },
}


def positive_weight(y_train: pd.Series) -> float:
    """Cociente negativos/positivos del entrenamiento (``scale_pos_weight``).

    Args:
        y_train: target binario (0 o 1) del conjunto de entrenamiento.

    Returns:
        Negativos divididos por positivos.

    Raises:
        ValueError: si no hay positivos.
    """
    positives = int(y_train.sum())
    if positives == 0:
        raise ValueError("El entrenamiento no tiene ejemplos positivos.")
    return (len(y_train) - positives) / positives


def _make_classifier(name: str, y_train: pd.Series | None, seed: int):
    """Crea el clasificador, con la semilla dada y el desbalance tratado."""
    if name == LOGISTIC_REGRESSION:
        return LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=seed
        )
    if name == RANDOM_FOREST:
        return RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        )
    if name == XGBOOST:
        if y_train is None:
            raise ValueError("XGBoost necesita y_train para scale_pos_weight.")
        return XGBClassifier(
            scale_pos_weight=positive_weight(y_train),
            eval_metric="logloss",
            random_state=seed,
            n_jobs=1,
        )
    raise ValueError(f"Modelo inválido: {name!r}; usar uno de {MODEL_NAMES}")


def build_pipeline(
    name: str, y_train: pd.Series | None = None, seed: int = config.SEED
) -> Pipeline:
    """Arma el ``Pipeline`` (imputación, escalado, clasificador) de un modelo.

    El desbalance se trata con ``class_weight='balanced'`` (regresión
    logística y Random Forest) o ``scale_pos_weight`` (XGBoost), calculado con
    el entrenamiento completo.

    Args:
        name: uno de ``MODEL_NAMES``.
        y_train: target de entrenamiento; obligatorio para XGBoost.
        seed: semilla del modelo; por defecto ``config.SEED``.

    Returns:
        Pipeline con los pasos ``imputer``, ``scaler`` y ``clf``.

    Raises:
        ValueError: si el nombre no existe o falta ``y_train`` en XGBoost.
    """
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", _make_classifier(name, y_train, seed)),
        ]
    )


def get_param_grid(name: str) -> dict[str, list]:
    """Devuelve la grilla de hiperparámetros de un modelo.

    Args:
        name: uno de ``MODEL_NAMES``.

    Returns:
        Copia de la grilla, con claves del tipo ``clf__<parámetro>``.

    Raises:
        ValueError: si el nombre no existe.
    """
    if name not in PARAM_GRIDS:
        raise ValueError(f"Modelo inválido: {name!r}; usar uno de {MODEL_NAMES}")
    return {k: list(v) for k, v in PARAM_GRIDS[name].items()}
