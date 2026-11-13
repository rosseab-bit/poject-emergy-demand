"""Tests de la línea de comandos: ``python -m src.pipeline``."""

import logging
import subprocess
import sys
from pathlib import Path

import mongomock
import pytest
from support import STRUCTURED_TABLES, mongo_logs, orphan_logs, user_tables

from src import config, logger, mongo, pipeline, report

# main() inicializa las bases: que lo haga en una carpeta temporal, no en db/.
pytestmark = pytest.mark.usefixtures("tmp_config")


class Recorder:
    """Reemplaza run_pipeline, run_etl y save_report registrando cómo se los llamó."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Instala los reemplazos; las llamadas quedan en ``calls``."""
        self.calls: list[tuple[str, dict]] = []
        monkeypatch.setattr(
            pipeline, "run_pipeline", lambda **kw: self._record("pipeline", kw)
        )
        monkeypatch.setattr(pipeline, "run_etl", lambda **kw: self._record("etl", kw))
        monkeypatch.setattr(
            report, "save_report", lambda run_id, **kw: self._record("reporte", run_id)
        )

    def _record(self, name: str, detail: object) -> str:
        """Anota una llamada y devuelve un run_id de mentira."""
        self.calls.append((name, detail))
        return "run-falso" if name != "reporte" else Path("reports/reporte_x.pdf")

    @property
    def names(self) -> list[str]:
        """Los nombres de las llamadas, en orden."""
        return [name for name, _ in self.calls]


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    """Un Recorder ya instalado."""
    return Recorder(monkeypatch)


def test_parser_defaults_come_from_config() -> None:
    """Sin argumentos: semilla de config.py, con ETL y entrenamiento, sin reporte."""
    args = pipeline.build_parser().parse_args([])
    assert args.seed == config.SEED
    assert args.solo_etl is False
    assert args.reporte is False


def test_no_arguments_runs_the_whole_pipeline_once(recorder: Recorder) -> None:
    """Sin argumentos corre ETL, entrenamiento y guardado, con la semilla por defecto."""
    assert pipeline.main([]) == 0
    assert recorder.calls == [("pipeline", {"seed": config.SEED})]


def test_seed_is_passed_to_the_run(recorder: Recorder) -> None:
    """--seed llega a la corrida."""
    assert pipeline.main(["--seed", "7"]) == 0
    assert recorder.calls == [("pipeline", {"seed": 7})]


def test_solo_etl_does_not_train(recorder: Recorder) -> None:
    """--solo-etl corre solo el ETL (con la semilla pedida)."""
    assert pipeline.main(["--solo-etl", "--seed", "3"]) == 0
    assert recorder.calls == [("etl", {"seed": 3})]


def test_reporte_generates_the_pdf_of_the_run_after_training(
    recorder: Recorder,
) -> None:
    """--reporte genera el PDF de la corrida que se acaba de hacer."""
    assert pipeline.main(["--reporte"]) == 0
    assert recorder.names == ["pipeline", "reporte"]
    assert recorder.calls[1] == ("reporte", "run-falso")


def test_the_saved_report_notice_belongs_to_the_run(recorder: Recorder) -> None:
    """El aviso «Reporte guardado» queda en logs atado a la corrida, no huérfano."""
    logger.setup_logging()
    assert pipeline.main(["--reporte"]) == 0
    messages = [e["message"] for e in mongo_logs("run-falso")]
    assert any(m.startswith("Reporte guardado en") for m in messages)
    assert orphan_logs() == 0


def test_reporte_without_training_is_rejected(
    recorder: Recorder, capsys: pytest.CaptureFixture[str]
) -> None:
    """--reporte con --solo-etl no tiene sentido: error de uso y no corre nada."""
    with pytest.raises(SystemExit) as exit_info:
        pipeline.main(["--solo-etl", "--reporte"])
    assert exit_info.value.code == 2
    assert "no se puede usar con --solo-etl" in capsys.readouterr().err
    assert recorder.calls == []


@pytest.mark.parametrize("bad", ["-3", "abc", "1.5", ""])
def test_invalid_seed_is_a_usage_error(recorder: Recorder, bad: str) -> None:
    """Una semilla negativa o que no es entera da error de uso y no corre nada."""
    with pytest.raises(SystemExit) as exit_info:
        pipeline.main(["--seed", bad])
    assert exit_info.value.code == 2
    assert recorder.calls == []


def test_unknown_argument_is_a_usage_error(recorder: Recorder) -> None:
    """Un argumento que no existe no se ignora en silencio."""
    with pytest.raises(SystemExit) as exit_info:
        pipeline.main(["--rapido"])
    assert exit_info.value.code == 2
    assert recorder.calls == []


def test_help_prints_usage_and_runs_nothing(
    recorder: Recorder, capsys: pytest.CaptureFixture[str]
) -> None:
    """--help muestra las opciones y sus valores por defecto, sin ejecutar nada."""
    with pytest.raises(SystemExit) as exit_info:
        pipeline.main(["--help"])
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    for fragment in (
        "python -m src.pipeline",
        "--seed",
        "--solo-etl",
        "--reporte",
        "muestra esta ayuda y termina",
    ):
        assert fragment in out
    assert f"por defecto {config.SEED}" in out
    assert recorder.calls == []


def test_a_failed_run_returns_1_and_logs_the_error(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Si la corrida falla, el código de salida es 1 y el error queda a la vista."""

    def boom(**kwargs: object) -> str:
        raise RuntimeError("se cayó el entrenamiento")

    monkeypatch.setattr(pipeline, "run_pipeline", boom)
    with caplog.at_level(logging.ERROR, logger="src.pipeline"):
        assert pipeline.main([]) == 1
    assert "La corrida falló: RuntimeError: se cayó el entrenamiento" in caplog.text


def test_console_logging_is_configured(
    recorder: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """La línea de comandos muestra el log por consola, en nivel INFO."""
    configured: dict = {}
    monkeypatch.setattr(
        pipeline.logging, "basicConfig", lambda **kwargs: configured.update(kwargs)
    )
    pipeline.main([])
    assert configured["level"] == logging.INFO
    assert "%(message)s" in configured["format"]


def database_state() -> tuple[bool, float | None]:
    """Si existe la base de trabajo y cuándo se modificó por última vez."""
    path = config.STRUCTURED_DB_PATH
    return (True, path.stat().st_mtime) if path.exists() else (False, None)


def run_cli(*args: str) -> subprocess.CompletedProcess:
    """Ejecuta ``python -m src.pipeline`` en un proceso aparte."""
    return subprocess.run(
        [sys.executable, "-m", "src.pipeline", *args],
        cwd=config.ROOT_DIR,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


def test_help_in_a_real_process_exits_cleanly_without_touching_the_databases() -> None:
    """``--help`` de verdad: sale con 0, muestra la ayuda y no ejecuta ni escribe nada."""
    before = database_state()
    done = run_cli("--help")
    assert done.returncode == 0
    assert "--solo-etl" in done.stdout
    assert "Inicio: run_etl_steps" not in done.stderr + done.stdout
    assert database_state() == before


def test_a_bad_argument_in_a_real_process_exits_with_2_without_running() -> None:
    """Un argumento inválido termina con código 2 y no ejecuta el pipeline."""
    before = database_state()
    done = run_cli("--seed", "menos-uno")
    assert done.returncode == 2
    assert "no es un número entero" in done.stderr
    assert database_state() == before


def test_main_creates_both_databases_before_running(
    recorder: Recorder, mongo_db: mongomock.Database
) -> None:
    """La línea de comandos también deja la base y las colecciones listas, aunque no entrene."""
    assert not config.STRUCTURED_DB_PATH.exists()
    assert mongo_db.list_collection_names() == []
    assert pipeline.main(["--solo-etl"]) == 0
    assert set(user_tables(config.STRUCTURED_DB_PATH)) == STRUCTURED_TABLES
    assert set(mongo_db.list_collection_names()) == set(mongo.COLLECTIONS)


def test_main_without_mongodb_fails_before_running(
    recorder: Recorder, caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    """Sin ``.env`` ni conexión, sale con 1, lo explica y no empieza la corrida."""
    mongo.reset_client()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(config, "ENV_PATH", tmp_path / ".env")
        for name in (mongo.ENV_URI, mongo.ENV_PASSWORD, mongo.ENV_DB_NAME):
            patch.delenv(name, raising=False)
        with caplog.at_level(logging.ERROR):
            assert pipeline.main([]) == 1
    assert recorder.calls == []
    assert "No se pudo preparar MongoDB" in caplog.text
    assert ".env.example" in caplog.text
