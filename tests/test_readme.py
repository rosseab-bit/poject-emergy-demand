"""Tests que evitan que el README quede desactualizado.

Comprueban que lo que el README nombra todavía existe: archivos y módulos, las
opciones de la línea de comandos, las páginas de la interfaz, los paquetes de
``requirements.txt`` y los comandos de calidad.
"""

import re
from pathlib import Path

import pytest

from app_pages.sidebar import PAGE_NAMES
from src import config, pipeline

README = (config.ROOT_DIR / "README.md").read_text(encoding="utf-8")
PATH_PATTERN = re.compile(
    r"`((?:src|app_pages|tests|docs|data|db|models|reports)/[\w./-]+"
    r"|[\w-]+\.(?:py|md|txt|toml|ini))`"
)
# Archivos que se generan al usar el proyecto: no existen en un clon limpio.
GENERATED = {"db/*.db", "models/*.joblib", "reports/*.pdf"}


def code_blocks(language: str) -> list[str]:
    """Líneas de los bloques de código del README en un lenguaje dado."""
    blocks = re.findall(rf"```{language}\n(.*?)```", README, flags=re.DOTALL)
    return [line for block in blocks for line in block.splitlines()]


def test_every_path_named_in_the_readme_exists() -> None:
    """Cada archivo o carpeta con nombre completo en el README existe."""
    named = {m for m in PATH_PATTERN.findall(README) if "*" not in m}
    assert len(named) >= 5, "el patrón dejó de encontrar rutas"
    missing = sorted(
        name
        for name in named
        if not (config.ROOT_DIR / name.rstrip("/")).exists() and name not in GENERATED
    )
    assert missing == []


def test_the_source_modules_listed_in_the_tree_exist() -> None:
    """Los módulos del árbol de la estructura existen, y ninguno de src/ falta."""
    tree = README.split("## Estructura del repositorio")[1].split("```")[1]
    listed = set(re.findall(r"\b([a-z_]+)\.py\b", tree))
    for module in listed - {"app"}:
        assert (config.ROOT_DIR / "src" / f"{module}.py").exists(), module
    real = {p.stem for p in (config.ROOT_DIR / "src").glob("*.py")}
    # El árbol agrupa report_*.py bajo «report*.py»; el resto se nombra uno a uno.
    unlisted = {m for m in real - listed - {"__init__"} if not m.startswith("report")}
    assert unlisted == set(), "módulos de src/ que el README no menciona"


def test_cli_options_in_the_readme_exist_in_the_parser() -> None:
    """Las opciones de ``python -m src.pipeline`` del README las acepta el parser."""
    parser = pipeline.build_parser()
    lines = [
        line
        for line in code_blocks("bash")
        if line.startswith("python -m src.pipeline")
    ]
    assert len(lines) >= 4
    for line in lines:
        command = line.split("#")[0].split()[3:]
        if "--help" in command:
            continue  # la ayuda termina el programa a propósito (código 0)
        parser.parse_args(command)  # sale con SystemExit si la opción no existe


def test_every_parser_option_is_documented_in_the_readme() -> None:
    """Ninguna opción de la línea de comandos queda sin documentar."""
    options = {
        flag
        for action in pipeline.build_parser()._actions
        for flag in action.option_strings
        if flag.startswith("--")
    }
    assert options  # --help, --seed, --solo-etl, --reporte
    assert [flag for flag in sorted(options) if flag not in README] == []


def test_interface_pages_in_the_readme_match_the_app() -> None:
    """La lista numerada de páginas del README coincide con las de la barra lateral."""
    uso = README.split("### Interfaz (Streamlit)")[1].split("### Línea de comandos")[0]
    listed = re.findall(r"^\d+\. \*\*(.+?):?\*\*", uso, flags=re.MULTILINE)
    assert tuple(listed) == PAGE_NAMES


def test_the_streamlit_command_points_to_an_existing_file() -> None:
    """``streamlit run app.py`` apunta a un archivo que existe."""
    match = re.search(r"streamlit run (\S+)", README)
    assert match is not None
    assert (config.ROOT_DIR / match.group(1)).is_file()


def test_the_dependencies_named_in_the_readme_are_in_requirements() -> None:
    """Los paquetes que el README dice que se instalan están fijados."""
    requirements = (config.ROOT_DIR / "requirements.txt").read_text()
    pinned = {line.split("==")[0] for line in requirements.splitlines() if "==" in line}
    named = {"pandas", "numpy", "scikit-learn", "xgboost", "streamlit", "reportlab"}
    named |= {"matplotlib", "joblib", "pytest", "ruff", "coverage"}
    assert named <= pinned
    assert all("==" in line for line in requirements.splitlines() if line[:1].isalpha())


def test_quality_commands_in_the_readme_are_configured() -> None:
    """Los comandos de calidad del README tienen su configuración en el repositorio."""
    bash = "\n".join(code_blocks("bash"))
    for command in ('pytest -m "not slow"', "ruff check", "coverage run"):
        assert command in bash, command
    pytest_ini = (config.ROOT_DIR / "pytest.ini").read_text()
    assert "slow:" in pytest_ini
    for config_file in ("ruff.toml", ".coveragerc"):
        assert (config.ROOT_DIR / config_file).is_file()


@pytest.mark.parametrize("pattern", sorted(GENERATED))
def test_generated_files_are_ignored_by_git(pattern: str) -> None:
    """Lo que el README dice que no está en el repositorio figura en ``.gitignore``."""
    ignored = (config.ROOT_DIR / ".gitignore").read_text().splitlines()
    assert pattern in ignored


def test_the_sql_examples_run_on_empty_databases(tmp_path: Path) -> None:
    """Las consultas SQL del README son válidas sobre las bases recién creadas."""
    import sqlite3

    from src import db

    structured, unstructured = tmp_path / "s.db", tmp_path / "u.db"
    db.init_databases(structured, unstructured)
    sql = "\n".join(code_blocks("sql"))
    assert "json_extract" in sql
    only_unstructured = sqlite3.connect(unstructured)
    both = sqlite3.connect(structured)
    both.execute(f"ATTACH DATABASE '{unstructured}' AS unstructured")
    try:
        for statement in sql.split(";"):
            body = "\n".join(
                line for line in statement.splitlines() if not line.startswith("--")
            ).strip()
            if not body or body.startswith("ATTACH"):
                continue
            body = body.replace("<RUN_ID>", "x")
            conn = only_unstructured if "FROM resultados_modelos" in body else both
            conn.execute(body).fetchall()
    finally:
        only_unstructured.close()
        both.close()
