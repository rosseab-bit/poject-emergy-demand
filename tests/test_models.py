"""Tests de src/models.py."""

import numpy as np
import pandas as pd
import pytest

from src import config, models


def make_data(n: int = 120, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Datos sintéticos con una señal simple, desbalanceados y con nulos."""
    rng = np.random.default_rng(seed)
    x = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("abcd"))
    y = pd.Series((x["a"] + 0.5 * rng.normal(size=n) > 0.8).astype(int))
    x.loc[5, "b"] = np.nan
    return x, y


@pytest.mark.parametrize("name", models.MODEL_NAMES)
def test_pipeline_fits_predicts_and_handles_nulls(name: str) -> None:
    """Cada pipeline ajusta con nulos y predice solo 0 o 1."""
    x, y = make_data()
    pipe = models.build_pipeline(name, y)
    pipe.fit(x, y)
    pred = pipe.predict(x)
    assert len(pred) == len(x)
    assert set(np.unique(pred)) <= {0, 1}
    assert list(pipe.named_steps) == ["imputer", "scaler", "clf"]


@pytest.mark.parametrize("name", models.MODEL_NAMES)
def test_param_grid_keys_are_valid_pipeline_params(name: str) -> None:
    """Todas las claves de la grilla existen en el pipeline."""
    _, y = make_data()
    pipe = models.build_pipeline(name, y)
    grid = models.get_param_grid(name)
    assert grid
    pipe.set_params(**{key: values[0] for key, values in grid.items()})


def test_param_grid_returns_copy() -> None:
    """Modificar la grilla devuelta no altera la original."""
    grid = models.get_param_grid(models.XGBOOST)
    grid["clf__max_depth"].append(99)
    assert 99 not in models.get_param_grid(models.XGBOOST)["clf__max_depth"]


def test_imbalance_is_handled() -> None:
    """Regresión logística y Random Forest balancean; XGBoost pesa los positivos."""
    _, y = make_data()
    lr = models.build_pipeline(models.LOGISTIC_REGRESSION).named_steps["clf"]
    rf = models.build_pipeline(models.RANDOM_FOREST).named_steps["clf"]
    xgb = models.build_pipeline(models.XGBOOST, y).named_steps["clf"]
    assert lr.class_weight == "balanced"
    assert rf.class_weight == "balanced"
    expected = (len(y) - y.sum()) / y.sum()
    assert xgb.scale_pos_weight == pytest.approx(expected)


def test_positive_weight_rejects_no_positives() -> None:
    """Sin positivos no se puede calcular el peso."""
    with pytest.raises(ValueError, match="positivos"):
        models.positive_weight(pd.Series([0, 0, 0]))


def test_xgboost_requires_y_train() -> None:
    """XGBoost sin y_train lanza ValueError."""
    with pytest.raises(ValueError, match="y_train"):
        models.build_pipeline(models.XGBOOST)


@pytest.mark.parametrize("name", models.MODEL_NAMES)
def test_seed_comes_from_config(name: str) -> None:
    """La semilla de cada modelo es config.SEED."""
    _, y = make_data()
    clf = models.build_pipeline(name, y).named_steps["clf"]
    assert clf.random_state == config.SEED


@pytest.mark.parametrize("name", models.MODEL_NAMES)
def test_same_seed_gives_same_predictions(name: str) -> None:
    """Dos entrenamientos con los mismos datos dan las mismas probabilidades."""
    x, y = make_data()
    first = models.build_pipeline(name, y).fit(x, y).predict_proba(x)
    second = models.build_pipeline(name, y).fit(x, y).predict_proba(x)
    np.testing.assert_array_equal(first, second)


def test_imputer_learns_only_from_training_data() -> None:
    """La mediana imputada sale solo de los datos con que se ajusta el pipeline."""
    x, y = make_data()
    pipe = models.build_pipeline(models.LOGISTIC_REGRESSION).fit(
        x.iloc[:60], y.iloc[:60]
    )
    expected = x.iloc[:60].median().to_numpy()
    np.testing.assert_allclose(pipe.named_steps["imputer"].statistics_, expected)


def test_unknown_model_raises() -> None:
    """Un nombre inexistente lanza ValueError."""
    with pytest.raises(ValueError, match="Modelo inválido"):
        models.build_pipeline("svm")
    with pytest.raises(ValueError, match="Modelo inválido"):
        models.get_param_grid("svm")


@pytest.mark.parametrize("name", models.MODEL_NAMES)
def test_custom_seed_reaches_the_model(name: str) -> None:
    """Una semilla distinta llega al clasificador y por defecto es la de config."""
    _, y = make_data()
    assert models.build_pipeline(name, y, seed=7).named_steps["clf"].random_state == 7
    default = models.build_pipeline(name, y).named_steps["clf"]
    assert default.random_state == config.SEED


def test_different_seeds_give_different_forests() -> None:
    """Random Forest con semillas distintas da probabilidades distintas."""
    x, y = make_data()
    a = models.build_pipeline(models.RANDOM_FOREST, y, seed=1).fit(x, y)
    b = models.build_pipeline(models.RANDOM_FOREST, y, seed=2).fit(x, y)
    assert not np.array_equal(a.predict_proba(x), b.predict_proba(x))
