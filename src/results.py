"""Datos de reproducibilidad y guardado de resultados en ``unstructured.db``."""

import hashlib
import logging
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from src import config
from src.logger import log_step

logger = logging.getLogger(__name__)

TRACKED_PACKAGES = (
    "pandas",
    "numpy",
    "scikit-learn",
    "xgboost",
    "streamlit",
    "reportlab",
    "matplotlib",
    "joblib",
)
GIT_TIMEOUT_SECONDS = 10


def file_sha256(path: Path) -> str:
    """Calcula el SHA-256 de un archivo, leyéndolo por bloques.

    Args:
        path: ruta del archivo.

    Returns:
        El hash en hexadecimal (64 caracteres).
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def library_versions(packages: tuple[str, ...] = TRACKED_PACKAGES) -> dict[str, str]:
    """Devuelve las versiones de Python y de las librerías del proyecto.

    Args:
        packages: nombres de paquetes a consultar.

    Returns:
        Diccionario nombre a versión; ``"no instalado"`` si falta alguna.
    """
    versions = {"python": platform.python_version()}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "no instalado"
    return versions


def _git(args: list[str], repo_dir: Path) -> str | None:
    """Ejecuta un comando de git de solo lectura; ``None`` si no se puede."""
    try:
        done = subprocess.run(
            ["git", *args],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def git_info(repo_dir: Path | None = None) -> dict[str, str | bool | None]:
    """Devuelve el hash del commit actual y si hay cambios sin commitear.

    Usa solo ``git log`` y ``git status`` (lectura). Si no hay git o la
    carpeta no es un repositorio, ambos valores son ``None``.

    Args:
        repo_dir: carpeta del repositorio; por defecto la raíz del proyecto.

    Returns:
        ``{"commit": <hash o None>, "dirty": <bool o None>}``.
    """
    repo = repo_dir if repo_dir is not None else config.ROOT_DIR
    commit = _git(["log", "-1", "--format=%H"], repo)
    status = _git(["status", "--porcelain"], repo)
    if not commit:
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": None if status is None else bool(status)}


@log_step
def reproducibility_info(csv_path: Path | None = None) -> dict[str, object]:
    """Reúne lo necesario para repetir una corrida.

    Args:
        csv_path: CSV original; por defecto ``config.RAW_CSV_PATH``.

    Returns:
        Diccionario con ``seed``, ``dataset_sha256``, ``versions``,
        ``git_commit`` y ``git_dirty``.
    """
    path = csv_path if csv_path is not None else config.RAW_CSV_PATH
    git = git_info()
    info = {
        "seed": config.SEED,
        "dataset_sha256": file_sha256(path),
        "versions": library_versions(),
        "git_commit": git["commit"],
        "git_dirty": git["dirty"],
    }
    logger.info(
        "Reproducibilidad: semilla %s, dataset %s, commit %s",
        info["seed"],
        str(info["dataset_sha256"])[:12],
        str(info["git_commit"])[:12],
    )
    return info
