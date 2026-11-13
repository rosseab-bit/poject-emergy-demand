"""Predicción del mes siguiente con los modelos guardados de una corrida."""

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src import data, db, features, logger, mongo, results, runs
from src.data import DATE_COLUMN, validate_monthly_dates

log = logging.getLogger(__name__)

SCENARIO = "A"  # el único que sirve para anticipar (no usa la potencia máxima)
CLASS_NAMES = {0: "normal", 1: "alta"}

# Rango razonable de la temperatura media mensual (°C) para el formulario.
TEMPERATURE_RANGE = (-5.0, 45.0)


@dataclass
class PredictionInput:
    """Entrada de una predicción: el mes, la temperatura y las variables del modelo.

    Attributes:
        month: primer día del mes a predecir.
        temperature: temperatura media esperada del mes (°C).
        features: una fila con las variables del escenario A.
        lags: rezagos completados desde ``datos_limpios`` (nombre a valor).
    """

    month: pd.Timestamp
    temperature: float
    features: pd.DataFrame
    lags: dict[str, float]


def next_month(clean: pd.DataFrame) -> pd.Timestamp:
    """Devuelve el mes siguiente al último dato disponible.

    Es el único mes que se puede predecir: sus rezagos de 1 y 12 meses ya se
    conocen. Para meses más lejanos faltaría la demanda del mes anterior.

    Args:
        clean: datos de ``datos_limpios``, ordenados por fecha.

    Returns:
        Primer día del mes siguiente al último dato.

    Raises:
        ValueError: si no hay datos.
    """
    if clean.empty:
        raise ValueError("No hay datos: primero ejecutá el ETL.")
    return clean[DATE_COLUMN].iloc[-1] + pd.offsets.MonthBegin(1)


def default_temperature(clean: pd.DataFrame, month: int) -> float:
    """Temperatura media histórica de un mes calendario, como valor inicial.

    Args:
        clean: datos de ``datos_limpios``.
        month: mes calendario (1 a 12).

    Returns:
        Promedio de ``temperatura_promedio`` de ese mes en todos los años.
    """
    same_month = clean[clean[DATE_COLUMN].dt.month == month]
    return float(same_month[features.TEMPERATURE_COLUMN].mean())


def validate_temperature(temperature: float) -> float:
    """Verifica que la temperatura sea un número dentro del rango razonable.

    Args:
        temperature: temperatura media esperada (°C).

    Returns:
        La temperatura como ``float``.

    Raises:
        ValueError: si no es un número finito o está fuera de ``TEMPERATURE_RANGE``.
    """
    value = float(temperature)
    low, high = TEMPERATURE_RANGE
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(
            f"La temperatura debe estar entre {low} y {high} °C: {temperature!r}"
        )
    return value


def build_input(clean: pd.DataFrame, temperature: float) -> PredictionInput:
    """Arma la fila de entrada del mes siguiente, con los rezagos desde la base.

    Agrega al historial una fila del mes siguiente con la temperatura esperada y
    calcula las variables con las mismas funciones del entrenamiento
    (``features``), así los rezagos salen solo de datos ya conocidos.

    Args:
        clean: datos de ``datos_limpios``, ordenados y sin meses faltantes.
        temperature: temperatura media esperada del mes (°C).

    Returns:
        ``PredictionInput`` con las variables del escenario A.

    Raises:
        ValueError: si no hay datos, la temperatura es inválida, faltan meses o
            el historial no alcanza para algún rezago.
    """
    month = next_month(clean)
    temperature = validate_temperature(temperature)
    validate_monthly_dates(clean[DATE_COLUMN])
    new_row = pd.DataFrame(
        {DATE_COLUMN: [month], features.TEMPERATURE_COLUMN: [temperature]}
    )
    history = pd.concat([clean, new_row], ignore_index=True)
    with logger.muted():  # las funciones de features no ensucian el historial
        history = features.add_temperature_features(history)
        history = features.add_calendar_features(history)
        history = features.add_lag_features(history)
    row = history.iloc[[-1]][features.FEATURES_A].reset_index(drop=True)
    missing = row.columns[row.isna().any()].tolist()
    if missing:
        raise ValueError(f"El historial no alcanza para calcular: {missing}")
    lags = {name: float(row.at[0, name]) for name in features.LAG_COLUMNS}
    return PredictionInput(month, temperature, row, lags)


def predict_models(
    run_id: str,
    prediction_input: PredictionInput,
    models_dir: Path | None = None,
) -> dict[str, dict[str, object]]:
    """Predice con los modelos guardados de una corrida (escenario A).

    Cada archivo ``.joblib`` se verifica contra el hash guardado antes de
    cargarse.

    Args:
        run_id: identificador de la corrida.
        prediction_input: salida de ``build_input``.
        models_dir: carpeta de modelos; por defecto ``config.MODELS_DIR``.

    Returns:
        Por modelo: ``probabilidad`` de demanda alta, ``clase`` (0 o 1),
        ``f1_holdout`` del modelo, ``f1_siempre_alta`` de la referencia y
        ``supera_referencia`` (si el F1 del modelo es mayor que el de la
        referencia).

    Raises:
        ValueError: si la corrida no tiene modelos del escenario A.
        ValueError: si algún archivo no coincide con su hash.
    """
    names = results.saved_models(run_id, SCENARIO)
    if not names:
        raise ValueError(
            f"La corrida {run_id} no tiene el escenario {SCENARIO} entrenado."
        )
    predictions = {}
    for name in names:
        doc = results.load_result(run_id, SCENARIO, name)
        estimator = results.load_model(doc, models_dir)
        x = prediction_input.features[doc["features"]]
        reference = doc["baselines"]["siempre_alta"]["f1"]
        predictions[name] = {
            "probabilidad": float(estimator.predict_proba(x)[0, 1]),
            "clase": int(estimator.predict(x)[0]),
            "f1_holdout": doc["holdout"]["f1"],
            "f1_siempre_alta": reference,
            "supera_referencia": doc["holdout"]["f1"] > reference,
        }
    return predictions


def build_prediction_doc(
    prediction_input: PredictionInput, predictions: dict[str, dict[str, object]]
) -> dict[str, object]:
    """Arma el documento JSON de una predicción para la colección ``predicciones``."""
    month = prediction_input.month
    return {
        "scenario": SCENARIO,
        "fecha": month.strftime("%Y-%m-%d"),
        "mes": int(month.month),
        "anio": int(month.year),
        "temperatura": prediction_input.temperature,
        "variables": {
            name: float(prediction_input.features.at[0, name])
            for name in prediction_input.features.columns
        },
        "rezagos": prediction_input.lags,
        "modelos": predictions,
    }


def save_prediction(run_id: str, doc: dict[str, object]) -> str:
    """Guarda una predicción en la colección ``predicciones``.

    Args:
        run_id: corrida cuyos modelos se usaron.
        doc: documento de ``build_prediction_doc``.

    Returns:
        El ``id`` corto (8 caracteres hexadecimales) de la predicción guardada.
    """
    prediction_id = runs.new_run_id()
    mongo.get_database()[mongo.PREDICTIONS].insert_one(
        results.to_document(
            doc, run_id=run_id, created_at=runs.now_iso(), id=prediction_id
        )
    )
    return prediction_id


@logger.log_step
def _predict_and_save(
    run_id: str,
    temperature: float,
    db_path: Path | None,
    models_dir: Path | None,
) -> dict[str, object]:
    """Arma la entrada, predice con los tres modelos y guarda el resultado."""
    prediction_input = build_input(data.load_clean_data(db_path), temperature)
    predictions = predict_models(run_id, prediction_input, models_dir)
    doc = build_prediction_doc(prediction_input, predictions)
    doc["id"] = save_prediction(run_id, doc)
    summary = ", ".join(
        f"{name} {p['probabilidad']:.2f}" for name, p in predictions.items()
    )
    log.info(
        "Predicción guardada (id %s): %s con %.1f °C -> %s",
        doc["id"],
        doc["fecha"],
        prediction_input.temperature,
        summary,
    )
    return doc


def predict_next_month(
    run_id: str,
    temperature: float,
    db_path: Path | None = None,
    models_dir: Path | None = None,
) -> dict[str, object]:
    """Predice el mes siguiente al último dato con los modelos de una corrida.

    Completa los rezagos desde ``datos_limpios``, predice con los modelos del
    escenario A y guarda el resultado en la colección ``predicciones`` de MongoDB. Los eventos
    quedan atados al ``run_id`` en ``logs``.

    Args:
        run_id: corrida entrenada cuyos modelos se usan.
        temperature: temperatura media esperada del mes (°C).
        db_path: ruta de ``structured.db``; por defecto la de ``config``.
        models_dir: carpeta de modelos; por defecto ``config.MODELS_DIR``.

    Returns:
        El documento de la predicción, con su ``id``.

    Raises:
        ValueError: si la corrida no existe, no está entrenada en el escenario
            A, la temperatura es inválida o falta historial.
    """
    db.init_structured_db(db_path)
    if runs.get_run(run_id) is None:
        raise ValueError(f"No existe la corrida {run_id}")
    logger.setup_logging()
    token = logger.current_run_id.set(run_id)
    try:
        return _predict_and_save(run_id, temperature, db_path, models_dir)
    finally:
        logger.current_run_id.reset(token)
        logger.flush_logs()
