"""Tests de src/results.py (datos de reproducibilidad)."""

import hashlib
import json
import sqlite3
import subprocess
from pathlib import Path

import numpy as np
import pytest
import sklearn
from support import real_dataset, tiny_grids

from src import config, data, db, features, models, results, train
from src.train import ModelRun


def test_file_sha256_matches_hashlib(tmp_path: Path) -> None:
    """El hash coincide con el de hashlib y cambia si cambia el contenido."""
    path = tmp_path / "a.csv"
    path.write_bytes(b"a,b\n1,2\n")
    assert results.file_sha256(path) == hashlib.sha256(b"a,b\n1,2\n").hexdigest()
    path.write_bytes(b"a,b\n1,3\n")
    assert results.file_sha256(path) != hashlib.sha256(b"a,b\n1,2\n").hexdigest()


def test_file_sha256_reads_large_files_by_blocks(tmp_path: Path) -> None:
    """Un archivo de varios bloques da el mismo hash que leído de una vez."""
    path = tmp_path / "big.bin"
    content = b"x" * (3 * (1 << 20) + 17)
    path.write_bytes(content)
    assert results.file_sha256(path) == hashlib.sha256(content).hexdigest()


def test_real_dataset_hash_is_stable_hex() -> None:
    """El hash del CSV real son 64 caracteres hexadecimales y no cambia."""
    first = results.file_sha256(config.RAW_CSV_PATH)
    assert len(first) == 64
    assert int(first, 16) >= 0
    assert first == results.file_sha256(config.RAW_CSV_PATH)


def test_library_versions_include_python_and_libraries() -> None:
    """Trae Python y las librerías instaladas con su versión real."""
    versions = results.library_versions()
    assert versions["scikit-learn"] == sklearn.__version__
    assert versions["python"].startswith("3.")
    assert set(versions) == {"python", *results.TRACKED_PACKAGES}


def test_library_versions_marks_missing_package() -> None:
    """Un paquete inexistente se marca como no instalado, sin error."""
    versions = results.library_versions(("paquete-que-no-existe",))
    assert versions["paquete-que-no-existe"] == "no instalado"


def test_git_info_in_this_repo() -> None:
    """En este repositorio devuelve un commit de 40 caracteres y un bool."""
    info = results.git_info()
    assert len(info["commit"]) == 40
    assert isinstance(info["dirty"], bool)


def test_git_info_outside_a_repo_returns_none(tmp_path: Path) -> None:
    """Fuera de un repositorio devuelve None en ambos valores."""
    assert results.git_info(tmp_path) == {"commit": None, "dirty": None}


def test_git_info_without_git_installed_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Si git no está instalado no falla."""

    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", no_git)
    assert results.git_info() == {"commit": None, "dirty": None}


def test_reproducibility_info_has_every_field() -> None:
    """Reúne semilla, hash, versiones y datos de git."""
    info = results.reproducibility_info()
    assert info["seed"] == config.SEED
    assert info["dataset_sha256"] == results.file_sha256(config.RAW_CSV_PATH)
    assert "scikit-learn" in info["versions"]
    assert len(info["git_commit"]) == 40
    assert isinstance(info["git_dirty"], bool)


@pytest.fixture(scope="module")
def model_run() -> ModelRun:
    """Un ModelRun real (regresión logística, escenario A), entrenado una vez."""
    with tiny_grids():
        return train.train_model(real_dataset(), models.LOGISTIC_REGRESSION, "A")


@pytest.fixture
def udb(tmp_path: Path) -> Path:
    """Base no estructurada temporal."""
    path = tmp_path / "unstructured.db"
    db.init_unstructured_db(path)
    return path


def fetch_doc(path: Path, table: str, where: str, args: tuple) -> dict:
    """Lee y decodifica el documento JSON de una fila."""
    with db.get_connection(path) as conn:
        row = conn.execute(f"SELECT doc FROM {table} WHERE {where}", args).fetchone()
    return json.loads(row["doc"])


def test_save_run_config_stores_reproducibility_and_settings(udb: Path) -> None:
    """La Tabla 3 guarda semilla, hash, versiones, commit y parámetros."""
    results.save_run_config("run1", results.build_run_config(), udb)
    with db.get_connection(udb) as conn:
        row = conn.execute(
            "SELECT json_extract(doc, '$.seed') AS seed, "
            "json_extract(doc, '$.holdout_months') AS holdout, "
            "json_extract(doc, '$.versions.scikit-learn') AS sk, "
            "json_extract(doc, '$.dataset_sha256') AS sha "
            "FROM configuracion WHERE run_id = 'run1'"
        ).fetchone()
    assert row["seed"] == config.SEED
    assert row["holdout"] == config.HOLDOUT_MONTHS
    assert row["sk"] == sklearn.__version__
    assert row["sha"] == results.file_sha256(config.RAW_CSV_PATH)
    doc = fetch_doc(udb, "configuracion", "run_id = ?", ("run1",))
    assert doc["scenarios"]["A"] == features.FEATURES_A
    assert doc["scenarios"]["B"] == features.FEATURES_B
    assert doc["target"]["window_months"] == config.TARGET_WINDOW_MONTHS


def test_save_run_config_rejects_duplicate_run(udb: Path) -> None:
    """Una corrida tiene una sola configuración."""
    results.save_run_config("run1", {"seed": 1}, udb)
    with pytest.raises(sqlite3.IntegrityError):
        results.save_run_config("run1", {"seed": 1}, udb)


def test_save_model_run_writes_file_and_row(
    model_run: ModelRun, udb: Path, tmp_path: Path
) -> None:
    """Se guarda el .joblib y una fila cuyo hash coincide con el archivo."""
    models_dir = tmp_path / "models"
    path = results.save_model_run("run1", model_run, models_dir, udb)
    assert path == models_dir / "run1_A_logistic_regression.joblib"
    assert path.is_file()
    doc = fetch_doc(
        udb, "resultados_modelos", "run_id = ? AND scenario = ?", ("run1", "A")
    )
    assert doc["model_file"] == {
        "name": path.name,
        "sha256": results.file_sha256(path),
    }
    assert doc["params"] == model_run.search.best_params
    assert doc["cv"]["scores"] == model_run.search.cv_scores
    assert doc["holdout"]["confusion_matrix"] == model_run.test.confusion_matrix
    assert "y_pred" not in doc["holdout"]
    assert set(doc["baselines"]) == {"siempre_alta", "aleatorio_proporcional"}
    assert doc["features"] == features.FEATURES_A


def test_saved_predictions_are_one_per_holdout_month(
    model_run: ModelRun, udb: Path, tmp_path: Path
) -> None:
    """El JSON trae una predicción por mes del hold-out con sus campos."""
    results.save_model_run("run1", model_run, tmp_path / "m", udb)
    doc = fetch_doc(udb, "resultados_modelos", "run_id = ?", ("run1",))
    preds = doc["predictions"]
    assert len(preds) == 48
    assert set(preds[0]) == {"fecha", "real", "predicho", "probabilidad"}
    assert preds[0]["fecha"] == "2022-09-01"
    assert [p["predicho"] for p in preds] == model_run.test.y_pred
    assert sum(p["real"] for p in preds) == 12


def test_save_model_run_rejects_duplicates_without_touching_files(
    model_run: ModelRun, udb: Path, tmp_path: Path
) -> None:
    """Guardar dos veces el mismo resultado falla y no pisa el archivo."""
    models_dir = tmp_path / "models"
    path = results.save_model_run("run1", model_run, models_dir, udb)
    before = results.file_sha256(path)
    with pytest.raises(ValueError, match="Ya existe"):
        results.save_model_run("run1", model_run, models_dir, udb)
    assert results.file_sha256(path) == before


def test_save_model_run_cleans_up_file_if_db_insert_fails(
    model_run: ModelRun,
    udb: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Si falla el guardado en la base, no queda el .joblib huérfano."""

    def boom(*args, **kwargs):
        raise RuntimeError("falla")

    monkeypatch.setattr(results, "build_result_doc", boom)
    models_dir = tmp_path / "models"
    with pytest.raises(RuntimeError):
        results.save_model_run("run1", model_run, models_dir, udb)
    assert list(models_dir.glob("*.joblib")) == []


def test_to_json_handles_numpy_and_none() -> None:
    """Serializa tipos de numpy y valores nulos de las grillas."""
    text = results.to_json({"a": np.float64(0.5), "b": np.int64(3), "c": None})
    assert json.loads(text) == {"a": 0.5, "b": 3, "c": None}


@pytest.fixture
def saved(model_run: ModelRun, udb: Path, tmp_path: Path) -> tuple[Path, Path]:
    """Guarda el ModelRun en una base y una carpeta temporales."""
    models_dir = tmp_path / "models"
    results.save_model_run("run1", model_run, models_dir, udb)
    return udb, models_dir


def test_load_result_with_json_extract(saved: tuple[Path, Path], model_run) -> None:
    """Un resultado se recupera con json_extract y con load_result."""
    udb_path, _ = saved
    with db.get_connection(udb_path) as conn:
        row = conn.execute(
            "SELECT json_extract(doc, '$.holdout.f1') AS f1, "
            "json_extract(doc, '$.params') AS params "
            "FROM resultados_modelos "
            "WHERE run_id = 'run1' AND scenario = 'A' "
            "AND model = 'logistic_regression'"
        ).fetchone()
    assert row["f1"] == model_run.test.f1
    assert json.loads(row["params"]) == model_run.search.best_params
    doc = results.load_result("run1", "A", "logistic_regression", udb_path)
    assert doc["holdout"]["f1"] == model_run.test.f1


def test_load_result_missing_raises(saved: tuple[Path, Path]) -> None:
    """Pedir un resultado que no existe lanza LookupError."""
    with pytest.raises(LookupError):
        results.load_result("otra", "A", "xgboost", saved[0])


def test_saved_model_reproduces_same_predictions(
    saved: tuple[Path, Path], tmp_path: Path
) -> None:
    """El modelo guardado, con los datos releídos de datos_limpios, repite las
    mismas probabilidades y predicciones guardadas en el JSON."""
    udb_path, models_dir = saved
    sdb = tmp_path / "structured.db"
    data.save_clean_data(data.flag_repeated_power(data.load_raw()), sdb)
    dataset = features.build_dataset(data.load_clean_data(sdb))
    x, y = features.get_model_data(dataset, "A")
    _, x_test, _, _ = train.split_holdout(x, y)

    doc = results.load_result("run1", "A", "logistic_regression", udb_path)
    loaded = results.load_model(doc, models_dir)
    stored = doc["predictions"]
    np.testing.assert_array_equal(
        loaded.predict(x_test), [p["predicho"] for p in stored]
    )
    np.testing.assert_allclose(
        loaded.predict_proba(x_test)[:, 1],
        [p["probabilidad"] for p in stored],
        rtol=1e-12,
    )
    dates = dataset.loc[x_test.index, "fecha"].dt.strftime("%Y-%m-%d").tolist()
    assert dates == [p["fecha"] for p in stored]


def test_load_model_rejects_tampered_file(saved: tuple[Path, Path]) -> None:
    """Si el archivo fue alterado, no se carga."""
    udb_path, models_dir = saved
    doc = results.load_result("run1", "A", "logistic_regression", udb_path)
    path = models_dir / doc["model_file"]["name"]
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ValueError, match="hash"):
        results.load_model(doc, models_dir)


def test_load_model_missing_file_raises(saved: tuple[Path, Path]) -> None:
    """Si falta el archivo lanza FileNotFoundError."""
    udb_path, models_dir = saved
    doc = results.load_result("run1", "A", "logistic_regression", udb_path)
    (models_dir / doc["model_file"]["name"]).unlink()
    with pytest.raises(FileNotFoundError):
        results.load_model(doc, models_dir)


def test_cross_query_with_attach_database(
    saved: tuple[Path, Path], tmp_path: Path
) -> None:
    """Un mismo run_id conecta runs y logs con resultados_modelos vía ATTACH."""
    udb_path, _ = saved
    sdb = tmp_path / "structured.db"
    db.init_structured_db(sdb)
    run_id = db.create_run(db_path=sdb)
    with db.get_connection(sdb) as conn:
        conn.execute("UPDATE runs SET run_id = 'run1' WHERE run_id = ?", (run_id,))
        conn.execute(
            "INSERT INTO logs (ts, level, module, message, run_id) "
            "VALUES ('t', 'INFO', 'src.train', 'Resultado A/logistic_regression', 'run1')"
        )
        db.attach_unstructured(conn, udb_path)
        row = conn.execute(
            "SELECT r.status, r.seed, m.model, "
            "json_extract(m.doc, '$.holdout.f1') AS f1, "
            "(SELECT COUNT(*) FROM logs l WHERE l.run_id = r.run_id) AS n_logs "
            "FROM runs r JOIN unstructured.resultados_modelos m ON m.run_id = r.run_id"
        ).fetchone()
    assert row["status"] == "running"
    assert row["seed"] == config.SEED
    assert row["model"] == "logistic_regression"
    assert 0.0 <= row["f1"] <= 1.0
    assert row["n_logs"] == 1


def test_build_run_config_uses_the_given_seed() -> None:
    """La configuración guarda la semilla recibida; por defecto, la de config."""
    assert results.build_run_config(seed=7)["seed"] == 7
    assert results.build_run_config()["seed"] == config.SEED


def test_has_results(saved: tuple[Path, Path]) -> None:
    """Indica si la corrida tiene resultados guardados."""
    assert results.has_results("run1", saved[0]) is True
    assert results.has_results("otra", saved[0]) is False


def test_to_json_rejects_values_it_cannot_serialize() -> None:
    """Un tipo que no es JSON ni numpy da un TypeError con su nombre."""
    with pytest.raises(TypeError, match="object"):
        results.to_json({"x": object()})


def test_save_model_run_refuses_to_overwrite_an_existing_model_file(
    model_run: ModelRun, udb: Path, tmp_path: Path
) -> None:
    """Si el .joblib ya existe (sin fila en la base), no se pisa y no se guarda nada."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    existing = models_dir / results.model_filename("run1", "A", "logistic_regression")
    existing.write_bytes(b"un modelo anterior")
    with pytest.raises(FileExistsError):
        results.save_model_run("run1", model_run, models_dir, udb)
    assert existing.read_bytes() == b"un modelo anterior"
    assert results.saved_models("run1", "A", udb) == []


def test_load_run_config_of_an_unknown_run_raises(udb: Path) -> None:
    """Pedir la configuración de una corrida sin configuración guardada falla claro."""
    with pytest.raises(LookupError, match="noexiste"):
        results.load_run_config("noexiste", udb)
