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

# Validación temporal: meses finales reservados como hold-out.
# Valor provisorio: el plan (sección 8) deja el tamaño final por definir.
HOLDOUT_MONTHS: int = 36
