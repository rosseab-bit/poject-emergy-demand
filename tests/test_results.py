"""Tests de src/results.py (datos de reproducibilidad)."""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import sklearn
from support import real_dataset, tiny_grids

from src import config, data, features, models, mongo, results, train
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


def stored(collection: str, **query: object) -> dict:
    """El documento guardado en una colección de MongoDB, tal cual (con sus metadatos)."""
    return mongo.get_database()[collection].find_one(query)


def test_save_run_config_stores_reproducibility_and_settings() -> None:
    """La Tabla 3 guarda semilla, hash, versiones, commit y parámetros."""
    results.save_run_config("run1", results.build_run_config())
    doc = stored(mongo.CONFIGURATION, run_id="run1")
    assert doc["seed"] == config.SEED
    assert doc["holdout_months"] == config.HOLDOUT_MONTHS
    assert doc["versions"]["scikit-learn"] == sklearn.__version__
    assert doc["dataset_sha256"] == results.file_sha256(config.RAW_CSV_PATH)
    assert doc["scenarios"]["A"] == features.FEATURES_A
    assert doc["scenarios"]["B"] == features.FEATURES_B
    assert doc["target"]["window_months"] == config.TARGET_WINDOW_MONTHS
    assert doc["created_at"].endswith("+00:00")


def test_save_run_config_rejects_duplicate_run() -> None:
    """Una corrida tiene una sola configuración."""
    results.save_run_config("run1", {"seed": 1})
    with pytest.raises(ValueError, match="Ya hay una configuración"):
        results.save_run_config("run1", {"seed": 1})
    assert mongo.get_database()[mongo.CONFIGURATION].count_documents({}) == 1


def test_load_run_config_returns_the_document_without_storage_fields() -> None:
    """Lo que vuelve es la configuración, sin ``run_id``, ``created_at`` ni ``_id``."""
    results.save_run_config("run1", {"seed": 5, "target": {"quantile": 0.75}})
    assert results.load_run_config("run1") == {"seed": 5, "target": {"quantile": 0.75}}


def test_save_model_run_writes_file_and_document(
    model_run: ModelRun, tmp_path: Path
) -> None:
    """Se guarda el .joblib y un documento cuyo hash coincide con el archivo."""
    models_dir = tmp_path / "models"
    path = results.save_model_run("run1", model_run, models_dir)
    assert path == models_dir / "run1_A_logistic_regression.joblib"
    assert path.is_file()
    doc = stored(mongo.RESULTS, run_id="run1", scenario="A")
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
    assert doc["run_id"] == "run1" and doc["created_at"]


def test_saved_predictions_are_one_per_holdout_month(
    model_run: ModelRun, tmp_path: Path
) -> None:
    """El documento trae una predicción por mes del hold-out con sus campos."""
    results.save_model_run("run1", model_run, tmp_path / "m")
    preds = stored(mongo.RESULTS, run_id="run1")["predictions"]
    assert len(preds) == 48
    assert set(preds[0]) == {"fecha", "real", "predicho", "probabilidad"}
    assert preds[0]["fecha"] == "2022-09-01"
    assert [p["predicho"] for p in preds] == model_run.test.y_pred
    assert sum(p["real"] for p in preds) == 12


def test_saved_documents_use_plain_python_types() -> None:
    """Nada de tipos de numpy en lo guardado: MongoDB no los acepta."""
    results.save_run_config("run1", {"a": np.float64(0.5), "b": np.int64(3), "c": None})
    doc = stored(mongo.CONFIGURATION, run_id="run1")
    assert (doc["a"], doc["b"], doc["c"]) == (0.5, 3, None)
    assert type(doc["b"]) is int and type(doc["a"]) is float


def test_save_model_run_rejects_duplicates_without_touching_files(
    model_run: ModelRun, tmp_path: Path
) -> None:
    """Guardar dos veces el mismo resultado falla y no pisa el archivo."""
    models_dir = tmp_path / "models"
    path = results.save_model_run("run1", model_run, models_dir)
    before = results.file_sha256(path)
    with pytest.raises(ValueError, match="Ya existe"):
        results.save_model_run("run1", model_run, models_dir)
    assert results.file_sha256(path) == before
    assert mongo.get_database()[mongo.RESULTS].count_documents({}) == 1


def test_save_model_run_cleans_up_file_if_the_save_fails(
    model_run: ModelRun, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Si falla el guardado en la base, no queda el .joblib huérfano."""

    def boom(*args, **kwargs):
        raise RuntimeError("falla")

    monkeypatch.setattr(results, "build_result_doc", boom)
    models_dir = tmp_path / "models"
    with pytest.raises(RuntimeError):
        results.save_model_run("run1", model_run, models_dir)
    assert list(models_dir.glob("*.joblib")) == []
    assert mongo.get_database()[mongo.RESULTS].count_documents({}) == 0


def test_to_json_handles_numpy_and_none() -> None:
    """Serializa tipos de numpy y valores nulos de las grillas."""
    text = results.to_json({"a": np.float64(0.5), "b": np.int64(3), "c": None})
    assert json.loads(text) == {"a": 0.5, "b": 3, "c": None}


@pytest.fixture
def saved(model_run: ModelRun, tmp_path: Path) -> Path:
    """Guarda el ModelRun en MongoDB (simulado) y en una carpeta temporal."""
    models_dir = tmp_path / "models"
    results.save_model_run("run1", model_run, models_dir)
    return models_dir


def test_load_result_and_query_by_dotted_field(saved: Path, model_run) -> None:
    """Un resultado se recupera con load_result y con una consulta a un campo anidado."""
    found = mongo.get_database()[mongo.RESULTS].find_one(
        {"run_id": "run1", "scenario": "A", "model": "logistic_regression"},
        {"holdout.f1": 1, "_id": 0},
    )
    assert found["holdout"]["f1"] == model_run.test.f1
    doc = results.load_result("run1", "A", "logistic_regression")
    assert doc["holdout"]["f1"] == model_run.test.f1
    assert not {"_id", "run_id", "created_at"} & set(doc)


def test_load_result_missing_raises(saved: Path) -> None:
    """Pedir un resultado que no existe lanza LookupError."""
    with pytest.raises(LookupError):
        results.load_result("otra", "A", "xgboost")


def test_saved_model_reproduces_same_predictions(saved: Path, tmp_path: Path) -> None:
    """El modelo guardado, con los datos releídos de datos_limpios, repite las
    mismas probabilidades y predicciones guardadas en el documento."""
    models_dir = saved
    sdb = tmp_path / "structured.db"
    data.save_clean_data(data.flag_repeated_power(data.load_raw()), sdb)
    dataset = features.build_dataset(data.load_clean_data(sdb))
    x, y = features.get_model_data(dataset, "A")
    _, x_test, _, _ = train.split_holdout(x, y)

    doc = results.load_result("run1", "A", "logistic_regression")
    loaded = results.load_model(doc, models_dir)
    stored_predictions = doc["predictions"]
    np.testing.assert_array_equal(
        loaded.predict(x_test), [p["predicho"] for p in stored_predictions]
    )
    np.testing.assert_allclose(
        loaded.predict_proba(x_test)[:, 1],
        [p["probabilidad"] for p in stored_predictions],
        rtol=1e-12,
    )
    dates = dataset.loc[x_test.index, "fecha"].dt.strftime("%Y-%m-%d").tolist()
    assert dates == [p["fecha"] for p in stored_predictions]


def test_load_model_rejects_tampered_file(saved: Path) -> None:
    """Si el archivo fue alterado, no se carga."""
    doc = results.load_result("run1", "A", "logistic_regression")
    path = saved / doc["model_file"]["name"]
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ValueError, match="hash"):
        results.load_model(doc, saved)


def test_load_model_missing_file_raises(saved: Path) -> None:
    """Si falta el archivo lanza FileNotFoundError."""
    doc = results.load_result("run1", "A", "logistic_regression")
    (saved / doc["model_file"]["name"]).unlink()
    with pytest.raises(FileNotFoundError):
        results.load_model(doc, saved)


def test_build_run_config_uses_the_given_seed() -> None:
    """La configuración guarda la semilla recibida; por defecto, la de config."""
    assert results.build_run_config(seed=7)["seed"] == 7
    assert results.build_run_config()["seed"] == config.SEED


def test_has_results(saved: Path) -> None:
    """Indica si la corrida tiene resultados guardados."""
    assert results.has_results("run1") is True
    assert results.has_results("otra") is False


def test_to_json_rejects_values_it_cannot_serialize() -> None:
    """Un tipo que no es JSON ni numpy da un TypeError con su nombre."""
    with pytest.raises(TypeError, match="object"):
        results.to_json({"x": object()})


def test_save_model_run_refuses_to_overwrite_an_existing_model_file(
    model_run: ModelRun, tmp_path: Path
) -> None:
    """Si el .joblib ya existe (sin documento), no se pisa y no se guarda nada."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    existing = models_dir / results.model_filename("run1", "A", "logistic_regression")
    existing.write_bytes(b"un modelo anterior")
    with pytest.raises(FileExistsError):
        results.save_model_run("run1", model_run, models_dir)
    assert existing.read_bytes() == b"un modelo anterior"
    assert results.saved_models("run1", "A") == []


def test_saved_models_lists_only_that_run_and_scenario_from_simple_to_complex(
    model_run: ModelRun, tmp_path: Path
) -> None:
    """Los modelos de una corrida y escenario salen en el orden de models.MODEL_NAMES."""
    results.save_model_run("run1", model_run, tmp_path / "m")
    other = mongo.get_database()[mongo.RESULTS]
    other.insert_many(
        [
            {"run_id": "run1", "scenario": "A", "model": "xgboost"},
            {"run_id": "run1", "scenario": "B", "model": "random_forest"},
            {"run_id": "run2", "scenario": "A", "model": "random_forest"},
        ]
    )
    assert results.saved_models("run1", "A") == ["logistic_regression", "xgboost"]
    assert results.saved_models("run1", "B") == ["random_forest"]
    assert results.saved_models("nada", "A") == []


def test_load_run_config_of_an_unknown_run_raises() -> None:
    """Pedir la configuración de una corrida sin configuración guardada falla claro."""
    with pytest.raises(LookupError, match="noexiste"):
        results.load_run_config("noexiste")
