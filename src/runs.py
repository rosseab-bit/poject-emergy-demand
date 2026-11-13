"""Corridas registradas en la colección ``runs`` de MongoDB.

Cada corrida (ETL, entrenamiento, predicción y reporte) lleva un ``run_id`` corto
que une todo lo que deja: sus logs, sus resultados y sus predicciones.
"""

import uuid
from datetime import UTC, datetime

from src import config, mongo

STATUS_RUNNING = "running"
STATUS_OK = "ok"
STATUS_ERROR = "error"
FINAL_STATUSES = (STATUS_OK, STATUS_ERROR)


def now_iso() -> str:
    """Devuelve la fecha y hora actual en UTC, formato ISO 8601."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def new_run_id() -> str:
    """Genera un ``run_id`` corto: 8 caracteres hexadecimales."""
    return uuid.uuid4().hex[:8]


def create_run(seed: int | None = None, notes: str | None = None) -> str:
    """Registra una corrida nueva con estado ``running``.

    Args:
        seed: semilla de la corrida; por defecto ``config.SEED``.
        notes: comentario libre opcional.

    Returns:
        El ``run_id`` generado.
    """
    run_id = new_run_id()
    mongo.get_database()[mongo.RUNS].insert_one(
        {
            "run_id": run_id,
            "started_at": now_iso(),
            "finished_at": None,
            "status": STATUS_RUNNING,
            "seed": config.SEED if seed is None else seed,
            "notes": notes,
        }
    )
    return run_id


def finish_run(run_id: str, status: str) -> None:
    """Cierra una corrida con su estado final.

    Args:
        run_id: identificador de la corrida.
        status: ``ok`` o ``error``.

    Raises:
        ValueError: si el estado no es válido o el ``run_id`` no existe.
    """
    if status not in FINAL_STATUSES:
        raise ValueError(f"Estado inválido: {status!r}")
    result = mongo.get_database()[mongo.RUNS].update_one(
        {"run_id": run_id}, {"$set": {"finished_at": now_iso(), "status": status}}
    )
    if result.matched_count == 0:
        raise ValueError(f"run_id inexistente: {run_id}")


def get_run(run_id: str) -> dict | None:
    """Devuelve el documento de una corrida, o ``None`` si no existe."""
    return mongo.get_database()[mongo.RUNS].find_one({"run_id": run_id}, {"_id": 0})
