"""Puerta de calidad: el linter y la configuración de ruff forman parte de los tests."""

import json
import subprocess
import sys
import tomllib
from pathlib import Path

from src import config

RUFF_CONFIG = config.ROOT_DIR / "ruff.toml"


def run_ruff(*args: str) -> subprocess.CompletedProcess:
    """Ejecuta ruff (el de este entorno) desde la raíz del proyecto."""
    return subprocess.run(
        [sys.executable, "-m", "ruff", *args],
        cwd=config.ROOT_DIR,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


def test_the_whole_project_passes_ruff_check() -> None:
    """Todo el código y los tests cumplen el linter con la configuración del repo."""
    done = run_ruff("check", ".")
    assert done.returncode == 0, done.stdout + done.stderr


def test_the_whole_project_is_formatted() -> None:
    """El formato es el de ruff format (PEP 8 de forma automática)."""
    done = run_ruff("format", "--check", ".")
    assert done.returncode == 0, done.stdout + done.stderr


def test_ruff_configuration_is_in_the_repo_with_the_agreed_settings() -> None:
    """La configuración está versionada y fija línea, versión y complejidad."""
    settings = tomllib.loads(RUFF_CONFIG.read_text())
    assert settings["line-length"] == 88
    assert settings["target-version"] == "py312"
    assert {"D1", "ANN", "C901"} <= set(settings["lint"]["extend-select"])
    assert settings["lint"]["mccabe"]["max-complexity"] == 10


def test_the_configuration_really_demands_docstrings_types_and_low_complexity(
    tmp_path: Path,
) -> None:
    """Un módulo sin docstring, sin tipos y demasiado complejo es rechazado."""
    branches = "\n".join(f"    if x == {n}:\n        return {n}" for n in range(12))
    (tmp_path / "malo.py").write_text(f"def f(x):\n{branches}\n    return 0\n")
    done = run_ruff(
        "check",
        "--config",
        str(RUFF_CONFIG),
        "--output-format",
        "json",
        str(tmp_path / "malo.py"),
    )
    codes = {item["code"] for item in json.loads(done.stdout)}
    assert {"D100", "D103", "ANN001", "ANN201", "C901"} <= codes


def test_ruff_version_is_pinned_in_requirements() -> None:
    """El linter está fijado, así que los resultados no cambian entre instalaciones."""
    lines = (config.ROOT_DIR / "requirements.txt").read_text().splitlines()
    assert any(line.startswith("ruff==") for line in lines)
