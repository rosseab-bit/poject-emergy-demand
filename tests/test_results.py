"""Tests de src/results.py (datos de reproducibilidad)."""

import hashlib
import subprocess
from pathlib import Path

import pytest
import sklearn

from src import config, results


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
