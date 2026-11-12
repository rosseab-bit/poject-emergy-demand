"""Tests de src/queries.py (lectura para la interfaz)."""

import logging
from pathlib import Path

import pandas as pd
import pytest

from src import config, data, db, features, pipeline, queries, results

LR = ("logistic_regression",)


@pytest.fixture
def paths(tmp_path: Path) -> dict:
    """Rutas temporales de las bases y de los modelos."""
    return {
        "db_path": tmp_path / "structured.db",
        "unstructured_path": tmp_path / "unstructured.db",
    }


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Una corrida real: ETL y entrenamiento de la regresión logística (A y B)."""
    base = tmp_path_factory.mktemp("queries")
    paths = {
        "db_path": base / "structured.db",
        "unstructured_path": base / "unstructured.db",
    }
    run_id = pipeline.run_etl(db_path=paths["db_path"], seed=7)
    pipeline.run_training(run_id, models_dir=base / "models", model_names=LR, **paths)
    return {"run_id": run_id, **paths}


def test_list_runs_empty_database_returns_empty_frame(paths: dict) -> None:
    """Sin corridas devuelve un DataFrame vacío con las columnas."""
    runs = queries.list_runs(paths["db_path"])
    assert runs.empty
    assert list(runs.columns) == queries.RUN_COLUMNS


def test_list_runs_newest_first(paths: dict) -> None:
    """La corrida más reciente aparece primero."""
    db.init_structured_db(paths["db_path"])
    first = db.create_run(db_path=paths["db_path"])
    second = db.create_run(seed=3, db_path=paths["db_path"])
    runs = queries.list_runs(paths["db_path"])
    assert runs["run_id"].tolist() == [second, first]
    assert runs["seed"].tolist() == [3, config.SEED]


def test_steps_after_etl_and_after_training(trained: dict, paths: dict) -> None:
    """El ETL se detecta en logs y el entrenamiento por los resultados."""
    steps = queries.run_steps(trained["run_id"], **paths_of(trained))
    assert steps == {
        "ETL": True,
        "Entrenamiento": True,
        "Predicción": False,
        "Reporte": False,
    }
    only_etl = pipeline.run_etl(db_path=paths["db_path"])
    steps = queries.run_steps(only_etl, **paths)
    assert steps["ETL"] is True
    assert steps["Entrenamiento"] is False


def paths_of(trained: dict) -> dict:
    """Solo las rutas de una corrida entrenada."""
    return {k: trained[k] for k in ("db_path", "unstructured_path")}


def test_run_summary_with_hash_and_steps(trained: dict) -> None:
    """El resumen trae estado, semilla, hash del dataset y pasos."""
    summary = queries.run_summary(trained["run_id"], **paths_of(trained))
    assert summary["status"] == "ok"
    assert summary["seed"] == 7
    assert summary["dataset_sha256"] == results.file_sha256(config.RAW_CSV_PATH)
    assert summary["steps"]["Entrenamiento"] is True


def test_run_summary_unknown_run_and_untrained_run(paths: dict) -> None:
    """Corrida inexistente da None; sin entrenar, el hash es None."""
    assert queries.run_summary("noexiste", **paths) is None
    run_id = pipeline.run_etl(db_path=paths["db_path"])
    summary = queries.run_summary(run_id, **paths)
    assert summary["dataset_sha256"] is None
    assert summary["steps"]["ETL"] is True


def test_results_table_matches_stored_documents(trained: dict) -> None:
    """La tabla tiene una fila por escenario y sus métricas coinciden con el JSON."""
    table = queries.results_table(trained["run_id"], trained["unstructured_path"])
    assert list(table.columns) == queries.RESULT_COLUMNS
    assert table[["scenario", "model"]].values.tolist() == [
        ["A", "logistic_regression"],
        ["B", "logistic_regression"],
    ]
    doc = results.load_result(
        trained["run_id"], "A", "logistic_regression", trained["unstructured_path"]
    )
    row = table.iloc[0]
    assert row["f1"] == doc["holdout"]["f1"]
    assert row["cv_mean"] == doc["cv"]["mean"]
    assert row["f1_siempre_alta"] == pytest.approx(0.4)
    assert (row["n_train"], row["n_test"]) == (188, 48)


def test_results_table_orders_by_scenario_then_simple_to_complex(
    paths: dict,
) -> None:
    """Dentro de cada escenario el orden es el de models.MODEL_NAMES."""
    udb = paths["unstructured_path"]
    db.init_unstructured_db(udb)
    scrambled = [
        ("B", "xgboost"),
        ("A", "xgboost"),
        ("B", "logistic_regression"),
        ("A", "random_forest"),
        ("A", "logistic_regression"),
        ("B", "random_forest"),
    ]
    with db.get_connection(udb) as conn:
        for scenario, model in scrambled:
            conn.execute(
                "INSERT INTO resultados_modelos "
                "(run_id, scenario, model, created_at, doc) VALUES ('r', ?, ?, 't', '{}')",
                (scenario, model),
            )
    table = queries.results_table("r", udb)
    assert table[["scenario", "model"]].values.tolist() == [
        ["A", "logistic_regression"],
        ["A", "random_forest"],
        ["A", "xgboost"],
        ["B", "logistic_regression"],
        ["B", "random_forest"],
        ["B", "xgboost"],
    ]


def test_results_table_empty_for_run_without_results(paths: dict) -> None:
    """Una corrida sin entrenar da una tabla vacía con las columnas."""
    table = queries.results_table("noexiste", paths["unstructured_path"])
    assert table.empty
    assert list(table.columns) == queries.RESULT_COLUMNS


def test_result_docs_include_confusion_matrices(trained: dict) -> None:
    """Los documentos completos traen la matriz de confusión de 2x2."""
    docs = queries.result_docs(trained["run_id"], trained["unstructured_path"])
    assert [d["scenario"] for d in docs] == ["A", "B"]
    for doc in docs:
        matrix = doc["holdout"]["confusion_matrix"]
        assert sum(map(sum, matrix)) == 48


def seed_logs(path: Path) -> None:
    """Carga eventos de prueba en dos corridas."""
    db.init_structured_db(path)
    rows = [
        ("r1", "INFO", "src.data", "a"),
        ("r1", "WARNING", "src.data", "b"),
        ("r1", "ERROR", "src.train", "c"),
        ("r2", "INFO", "src.train", "d"),
    ]
    with db.get_connection(path) as conn:
        for run_id, level, module, message in rows:
            conn.execute(
                "INSERT INTO logs (ts, level, module, message, run_id) "
                "VALUES ('t', ?, ?, ?, ?)",
                (level, module, message, run_id),
            )


def test_get_logs_filters(paths: dict) -> None:
    """Se filtra por corrida, nivel y módulo, en orden cronológico."""
    seed_logs(paths["db_path"])
    path = paths["db_path"]
    assert queries.get_logs(db_path=path)["message"].tolist() == ["a", "b", "c", "d"]
    assert queries.get_logs("r1", db_path=path)["message"].tolist() == ["a", "b", "c"]
    assert queries.get_logs("r1", levels=["ERROR"], db_path=path)[
        "message"
    ].tolist() == ["c"]
    both = queries.get_logs(
        levels=["INFO", "ERROR"], modules=["src.train"], db_path=path
    )
    assert both["message"].tolist() == ["c", "d"]
    assert queries.get_logs("r9", db_path=path).empty


def test_list_log_modules(paths: dict) -> None:
    """Lista los módulos con eventos, de una corrida o de todas."""
    seed_logs(paths["db_path"])
    assert queries.list_log_modules(db_path=paths["db_path"]) == [
        "src.data",
        "src.train",
    ]
    assert queries.list_log_modules("r2", db_path=paths["db_path"]) == ["src.train"]


def test_filters_use_parameters_not_string_building(paths: dict) -> None:
    """Un valor con comillas no rompe la consulta ni inyecta SQL."""
    seed_logs(paths["db_path"])
    hostile = "x') OR ('1'='1"
    assert queries.get_logs(levels=[hostile], db_path=paths["db_path"]).empty
    assert queries.get_logs(hostile, db_path=paths["db_path"]).empty


@pytest.fixture(scope="module")
def etl_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Base con el ETL ya ejecutado (solo datos limpios)."""
    path = tmp_path_factory.mktemp("overview") / "structured.db"
    data.save_clean_data(data.flag_repeated_power(data.load_raw()), path)
    return path


def test_data_overview_before_etl_is_none(paths: dict) -> None:
    """Sin ETL no hay resumen."""
    assert queries.data_overview(paths["db_path"]) is None


def test_data_overview_real_numbers(etl_db: Path) -> None:
    """Con el CSV real: 308 meses, 259 completos, 48 con sectores nulos."""
    overview = queries.data_overview(etl_db)
    assert overview["rows"] == 308
    assert overview["rows_complete"] == 259
    assert overview["rows_sector_nulls"] == 48
    assert overview["repeated_months"] == ["2002-03", "2021-08"]
    assert (overview["first_month"], overview["last_month"]) == ("2001-01", "2026-08")
    target = overview["target"]
    assert (target["total"], target["positivos"], target["negativos"]) == (236, 60, 176)
    assert target["first_month"] == "2007-01"
    assert target["proporcion_positivos"] == pytest.approx(60 / 236)


def test_demand_index_only_for_complete_years() -> None:
    """El índice vale 1 en promedio por año y es nulo en años incompletos."""
    dates = pd.date_range("2020-01-01", periods=14, freq="MS")
    df = pd.DataFrame({"fecha": dates, "demanda_total": [100.0] * 12 + [200.0, 300.0]})
    out = queries.add_demand_index(df)
    assert out["indice_demanda"].iloc[:12].tolist() == [1.0] * 12
    assert out["indice_demanda"].iloc[12:].isna().all()


def test_demand_index_removes_the_trend() -> None:
    """Con crecimiento fuerte entre años, cada año se normaliza por su promedio."""
    dates = pd.date_range("2020-01-01", periods=24, freq="MS")
    base = [90.0, 110.0] * 6
    df = pd.DataFrame({"fecha": dates, "demanda_total": base + [v * 2 for v in base]})
    out = queries.add_demand_index(df)
    assert out["indice_demanda"].iloc[:12].tolist() == pytest.approx(
        out["indice_demanda"].iloc[12:].tolist()
    )


def test_seasonality_by_month_has_twelve_months(etl_db: Path) -> None:
    """La estacionalidad real tiene 12 meses y promedia cerca de 1."""
    season = queries.seasonality_by_month(data.load_clean_data(etl_db))
    assert season.index.tolist() == list(range(1, 13))
    assert season["indice_medio"].mean() == pytest.approx(1.0, abs=0.02)
    assert season["indice_medio"].notna().all()


def test_feature_importance_logistic_regression_has_signed_coefficients(
    trained_run: dict,
) -> None:
    """Regresión logística: un coeficiente por variable, con signo, ordenados."""
    table = queries.feature_importance_for(
        trained_run["run_id"],
        "A",
        "logistic_regression",
        trained_run["unstructured_path"],
        trained_run["models_dir"],
    )
    assert table["feature"].sort_values().tolist() == sorted(features.FEATURES_A)
    assert set(table["tipo"]) == {"coeficiente"}
    assert (table["importancia"] < 0).any() and (table["importancia"] > 0).any()
    magnitudes = table["importancia"].abs().tolist()
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_feature_importance_random_forest_sums_to_one_and_b_has_power(
    trained_run: dict,
) -> None:
    """Random Forest: importancias sin signo que suman 1; B incluye la potencia."""
    table = queries.feature_importance_for(
        trained_run["run_id"],
        "B",
        "random_forest",
        trained_run["unstructured_path"],
        trained_run["models_dir"],
    )
    assert set(table["tipo"]) == {"importancia"}
    assert table["importancia"].sum() == pytest.approx(1.0)
    assert (table["importancia"] >= 0).all()
    assert "potencia_maxima" in table["feature"].tolist()
    assert len(table) == len(features.FEATURES_B)


def test_feature_importance_rejects_models_without_importances() -> None:
    """Un clasificador sin coeficientes ni importancias lanza ValueError."""
    from sklearn.dummy import DummyClassifier
    from sklearn.pipeline import Pipeline

    pipe = Pipeline([("clf", DummyClassifier())])
    with pytest.raises(ValueError, match="no expone"):
        results.feature_importance(pipe, ["a"])


def test_feature_importance_rejects_wrong_number_of_names() -> None:
    """Si la cantidad de nombres no coincide con el modelo falla."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    clf = LogisticRegression().fit(np.array([[0.0, 1.0], [1.0, 0.0]]), [0, 1])
    with pytest.raises(ValueError, match="no coincide"):
        results.feature_importance(Pipeline([("clf", clf)]), ["solo_una"])


def test_scenario_comparison_computes_b_minus_a(trained_run: dict) -> None:
    """La comparación A contra B calcula las diferencias B menos A por modelo."""
    table = queries.results_table(
        trained_run["run_id"], trained_run["unstructured_path"]
    )
    comparison = queries.scenario_comparison(table)
    assert comparison["model"].tolist() == ["logistic_regression", "random_forest"]
    for row in comparison.itertuples():
        a = table[(table["scenario"] == "A") & (table["model"] == row.model)].iloc[0]
        b = table[(table["scenario"] == "B") & (table["model"] == row.model)].iloc[0]
        assert row.f1_A == a["f1"] and row.f1_B == b["f1"]
        assert row.dif_f1 == pytest.approx(b["f1"] - a["f1"])
        assert row.dif_cv == pytest.approx(b["cv_mean"] - a["cv_mean"])


def test_scenario_comparison_needs_both_scenarios(trained_run: dict) -> None:
    """Con un solo escenario no hay comparación."""
    table = queries.results_table(
        trained_run["run_id"], trained_run["unstructured_path"]
    )
    only_a = table[table["scenario"] == "A"]
    assert queries.scenario_comparison(only_a).empty


def test_noise_summary_real_numbers(trained_run: dict) -> None:
    """Resume el hold-out: 48 meses, 12 altos, desvío máximo y referencia 0,40."""
    path = trained_run["unstructured_path"]
    table = queries.results_table(trained_run["run_id"], path)
    docs = queries.result_docs(trained_run["run_id"], path)
    noise = queries.noise_summary(table, docs)
    assert noise["n_test"] == 48
    assert noise["positives"] == 12
    assert noise["max_cv_std"] == pytest.approx(table["cv_std"].max())
    assert noise["always_high_f1"] == pytest.approx(0.4)


def test_noise_summary_without_results_is_none() -> None:
    """Sin resultados no hay resumen."""
    empty = pd.DataFrame(columns=queries.RESULT_COLUMNS)
    assert queries.noise_summary(empty, []) is None


def test_prediction_step_follows_the_saved_predictions(tmp_path: Path) -> None:
    """El paso Predicción está pendiente hasta que se guarda una predicción.

    Usa bases propias: no debe escribir en la corrida compartida de la sesión.
    """
    from src import predict

    paths = {
        "db_path": tmp_path / "structured.db",
        "unstructured_path": tmp_path / "unstructured.db",
    }
    models_dir = tmp_path / "models"
    root = logging.getLogger("src")
    saved_handlers, saved_level = list(root.handlers), root.level
    try:
        run_id = pipeline.run_etl(db_path=paths["db_path"])
        other_id = pipeline.run_etl(db_path=paths["db_path"])
        pipeline.run_training(
            run_id, models_dir=models_dir, scenarios=("A",), model_names=LR, **paths
        )
        assert queries.has_predictions(run_id, paths["unstructured_path"]) is False
        assert queries.run_steps(run_id, **paths)["Predicción"] is False
        predict.predict_next_month(run_id, 15.0, models_dir=models_dir, **paths)
    finally:
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)
    assert queries.has_predictions(run_id, paths["unstructured_path"]) is True
    assert queries.run_steps(run_id, **paths) == {
        "ETL": True,
        "Entrenamiento": True,
        "Predicción": True,
        "Reporte": False,
    }
    # Otra corrida sin predicciones no se ve afectada.
    assert queries.has_predictions(other_id, paths["unstructured_path"]) is False
