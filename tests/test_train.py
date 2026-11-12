"""Tests de src/train.py."""

import numpy as np
import pandas as pd
import pytest

from src import config, data, features, models, train


def make_xy(n: int = 100) -> tuple[pd.DataFrame, pd.Series]:
    """Variables y target sintéticos, ordenados en el tiempo."""
    x = pd.DataFrame({"a": range(n)})
    y = pd.Series([i % 4 == 0 for i in range(n)], dtype=int)
    return x, y


def test_split_holdout_sizes_and_order() -> None:
    """Los últimos meses van a prueba y el entrenamiento es anterior."""
    x, y = make_xy()
    x_tr, x_te, y_tr, y_te = train.split_holdout(x, y, holdout=20)
    assert len(x_tr) == len(y_tr) == 80
    assert len(x_te) == len(y_te) == 20
    assert x_tr["a"].max() < x_te["a"].min()
    assert x_te["a"].tolist() == list(range(80, 100))


def test_split_holdout_has_no_overlap_and_keeps_alignment() -> None:
    """Entrenamiento y prueba no comparten filas y X sigue alineado con y."""
    x, y = make_xy()
    x_tr, x_te, y_tr, y_te = train.split_holdout(x, y, holdout=30)
    assert set(x_tr.index).isdisjoint(x_te.index)
    assert x_tr.index.equals(y_tr.index)
    assert x_te.index.equals(y_te.index)


def test_split_holdout_default_comes_from_config() -> None:
    """Sin argumento usa config.HOLDOUT_MONTHS."""
    x, y = make_xy(100)
    _, x_te, _, _ = train.split_holdout(x, y)
    assert len(x_te) == config.HOLDOUT_MONTHS


@pytest.mark.parametrize("holdout", [0, -5, 100, 150])
def test_split_holdout_rejects_bad_size(holdout: int) -> None:
    """Un hold-out vacío o que se come todo el dataset lanza ValueError."""
    x, y = make_xy(100)
    with pytest.raises(ValueError, match="holdout inválido"):
        train.split_holdout(x, y, holdout=holdout)


def test_split_holdout_rejects_misaligned_index() -> None:
    """Si x e y no comparten índice lanza ValueError."""
    x, y = make_xy()
    with pytest.raises(ValueError, match="mismo índice"):
        train.split_holdout(x, y.iloc[::-1].reset_index(drop=True).iloc[:50])


def test_split_holdout_real_data() -> None:
    """Con los datos reales: 188 filas de entrenamiento, 48 de prueba, 12 positivos."""
    ds = features.build_dataset(data.flag_repeated_power(data.load_raw()))
    x, y = features.get_model_data(ds, "A")
    x_tr, x_te, y_tr, y_te = train.split_holdout(x, y)
    assert (len(x_tr), len(x_te)) == (188, 48)
    assert int(y_te.sum()) == 12
    assert int(y_tr.sum()) == 48
    assert ds.loc[x_tr.index, "fecha"].max() < ds.loc[x_te.index, "fecha"].min()


def make_series_data(n: int = 150, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Datos ordenados con señal, para probar la búsqueda."""
    rng = np.random.default_rng(seed)
    x = pd.DataFrame(rng.normal(size=(n, 3)), columns=list("abc"))
    y = pd.Series((x["a"] + 0.5 * rng.normal(size=n) > 0.5).astype(int))
    return x, y


def test_make_cv_is_temporal_and_uses_config() -> None:
    """Hay config.CV_SPLITS cortes y la validación siempre es posterior."""
    x, _ = make_series_data()
    splits = list(train.make_cv().split(x))
    assert len(splits) == config.CV_SPLITS
    for tr, va in splits:
        assert tr.max() < va.min()


@pytest.mark.parametrize("name", models.MODEL_NAMES)
def test_search_hyperparameters_returns_consistent_result(name: str) -> None:
    """Devuelve parámetros de la grilla, F1 por corte, promedio, desvío y modelo."""
    x, y = make_series_data()
    res = train.search_hyperparameters(name, x, y)
    grid = models.get_param_grid(name)
    assert res.model == name
    assert set(res.best_params) == set(grid)
    for key, value in res.best_params.items():
        assert value in grid[key]
    assert len(res.cv_scores) == config.CV_SPLITS
    assert all(0.0 <= s <= 1.0 for s in res.cv_scores)
    assert res.cv_mean == pytest.approx(np.mean(res.cv_scores))
    assert res.cv_std == pytest.approx(np.std(res.cv_scores, ddof=1))
    assert set(np.unique(res.estimator.predict(x))) <= {0, 1}


def test_search_hyperparameters_is_reproducible() -> None:
    """Dos búsquedas iguales dan los mismos resultados."""
    x, y = make_series_data()
    a = train.search_hyperparameters(models.RANDOM_FOREST, x, y)
    b = train.search_hyperparameters(models.RANDOM_FOREST, x, y)
    assert a.best_params == b.best_params
    assert a.cv_scores == b.cv_scores


def test_search_hyperparameters_real_data_uses_only_training_rows() -> None:
    """Con datos reales y escenario A busca solo sobre las 188 filas de train."""
    ds = features.build_dataset(data.flag_repeated_power(data.load_raw()))
    x, y = features.get_model_data(ds, "A")
    x_tr, _, y_tr, _ = train.split_holdout(x, y)
    res = train.search_hyperparameters(models.LOGISTIC_REGRESSION, x_tr, y_tr)
    assert len(res.cv_scores) == config.CV_SPLITS
    assert res.estimator.named_steps["imputer"].statistics_.shape[0] == x_tr.shape[1]
