"""Configuración central: semilla, rutas y constantes del proyecto.

La semilla y las rutas se definen solo acá; el resto del código las importa.
"""

from pathlib import Path

# Semilla única para toda la aleatoriedad del proyecto.
SEED: int = 42

# Rutas (relativas a la raíz del repositorio).
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
DATA_RAW_DIR: Path = ROOT_DIR / "data" / "raw"
DB_DIR: Path = ROOT_DIR / "db"
MODELS_DIR: Path = ROOT_DIR / "models"
REPORTS_DIR: Path = ROOT_DIR / "reports"

# Dataset original (no se modifica nunca).
RAW_CSV_PATH: Path = DATA_RAW_DIR / "demanda_electricidad_original_mensual2.csv"

# Bases SQLite.
STRUCTURED_DB_PATH: Path = DB_DIR / "structured.db"
UNSTRUCTURED_DB_PATH: Path = DB_DIR / "unstructured.db"

# MongoDB (Atlas): credenciales en un .env que no se sube a Git.
ENV_PATH: Path = ROOT_DIR / ".env"
MONGO_DB_NAME: str = "demanda_electrica"
MONGO_TIMEOUT_MS: int = 10_000

# Validación temporal: meses finales reservados como hold-out (12 positivos
# en los últimos 48 meses, contra 7 en 36) y cortes de TimeSeriesSplit en el
# resto (con 3 cortes cada validación tiene al menos 6 positivos).
HOLDOUT_MONTHS: int = 48
CV_SPLITS: int = 3

# Temperatura base (°C) para los grados-día de frío y calor.
BASE_TEMP_C: float = 18.0

# Target "demanda alta": cociente contra el mismo mes de los N años previos y
# umbral móvil = cuantil de los cocientes de los últimos M meses (solo pasado).
TARGET_YEARS_BACK: int = 3
TARGET_WINDOW_MONTHS: int = 36
TARGET_QUANTILE: float = 0.75
