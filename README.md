# TP Final: clasificación de la demanda eléctrica mensual

Trabajo final individual de **Programación Avanzada en Ciencia de Datos**, presentado bajo el nombre **MatIAndo**.

Un clasificador que responde una pregunta: *¿la demanda eléctrica de un mes va a ser **alta** respecto de lo normal para esa época del año?* Usa scikit-learn (regresión logística, Random Forest y XGBoost), dos bases SQLite, un logger propio que escribe en una de ellas, una interfaz en Streamlit y un reporte PDF.

> **Resultado, dicho sin maquillaje.** Con 236 meses con target y un hold-out de solo 48 meses (12 de demanda alta), **ningún modelo supera con claridad a la regla trivial «siempre alta»** (F1 = 0,40) cuando se predice con anticipación (escenario A), y entre los tres modelos **no hay un ganador claro**: sus diferencias caen dentro del ruido de la validación cruzada. El valor del proyecto está en el método (sin fuga de información, validación temporal, comparación contra una referencia y trazabilidad), no en una precisión alta. Detalle en [Resultados](#resultados) y [Límites](#límites-del-modelo).

## Contenido

- [Qué se predice](#qué-se-predice)
- [Datos y fuente](#datos-y-fuente)
- [Resultados](#resultados)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Uso](#uso)
- [Decisiones de diseño](#decisiones-de-diseño)
- [Reproducibilidad](#reproducibilidad)
- [Las dos bases de datos](#las-dos-bases-de-datos)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Calidad](#calidad)
- [Límites del modelo](#límites-del-modelo)

## Qué se predice

**Nivel mensual**, no diario ni horario.

- **Target `demanda_alta`.** Para cada mes se calcula el cociente entre su demanda total y el promedio de la demanda del *mismo mes calendario* de los 3 años anteriores. El mes es «alto» (1) si ese cociente supera el cuantil 75 de los cocientes de los 36 meses anteriores. Es un umbral móvil que mira solo el pasado: descuenta la tendencia y la estacionalidad y evita la fuga de información. Hay 236 meses con target (desde 2007-01) y 60 positivos (25,4 %).
- **Escenario A, «anticipación» (el principal).** 14 variables que se conocen antes del mes: temperatura (y grados-día de frío y calor con base 18 °C), mes como seno y coseno, año, y rezagos de 1 y 12 meses de la demanda total y por sector. Es el único que sirve para predecir.
- **Escenario B, «diagnóstico».** Las mismas más la potencia máxima (15 variables). La potencia máxima ocurre en el mismo mes que la demanda (correlación 0,96), así que **no se conoce de antemano**: B sirve para medir cuánto aporta, no para anticipar.
- **Validación.** `TimeSeriesSplit` de 3 cortes y un hold-out temporal con los últimos 48 meses. Los tres modelos usan los mismos cortes y la misma semilla (`src/config.py`, 42). Todo lo que aprende de los datos (imputación, escalado) va dentro del `Pipeline` y se ajusta solo con entrenamiento.
- **Métrica principal.** F1 de la clase «alta», porque las clases están desbalanceadas (≈25 % de positivos) y el accuracy engaña.

## Datos y fuente

| | |
|---|---|
| Archivo | `data/raw/demanda_electricidad_original_mensual2.csv` (no se modifica nunca) |
| Fuente | Fuentes públicas del gobierno de Argentina. Portal, URL exacta, fecha de descarga y licencia: **no registrados** |
| Cobertura | 308 meses, de 2001-01 a 2026-08. Nacional |
| Columnas usadas | `demanda_total`, `demanda_residencial`, `comercio_e_industria`, `grandes_usuarios`, `temperatura_promedio`, `potencia_maxima` |
| Columnas no usadas | Las 24 columnas por distribuidora: la temperatura es nacional, así que no agregarían información independiente |
| Integridad | El SHA-256 del archivo se guarda en cada corrida (tabla `configuracion`) |

**Supuestos que conviene conocer**

- **Zona de la temperatura: Buenos Aires / AMBA, inferida y no confirmada por la fuente.** Se comparó el promedio mensual de `temperatura_promedio` (2005-2025) con las normales 1991-2020 de la ciudad de Buenos Aires: la forma estacional coincide y el dataset resulta entre 0,3 y 1,0 °C más cálido en todos los meses, coherente con el calentamiento reciente. Las normales salen de una fuente secundaria, no del SMN.
- **Grados-día con la temperatura media mensual.** El dataset no trae temperaturas diarias, así que los grados-día de frío y calor (base 18 °C) son una aproximación.
- **Potencia máxima repetida.** Dos valores repetidos (2002-03 y 2021-08) se tratan como nulos, se marcan con `potencia_repetida` y los imputa el `Pipeline`.
- **Sectores sin datos antes de 2005.** Las columnas por sector son nulas antes de 2005 y no se imputan. Se conserva igual la `demanda_total` de 2001-2004, porque el target y los rezagos la necesitan como historia.
- **Se guardan 308 filas en `datos_limpios`**, 259 completas. El target existe desde 2007-01 (236 filas).

## Resultados

Corrida completa con la grilla de hiperparámetros por defecto y la semilla 42. F1 de la clase «alta»; «cv» es el promedio ± desvío de las 3 validaciones; el hold-out tiene 48 meses (12 de demanda alta).

| Escenario | Modelo | Precisión | Recall | **F1 hold-out** | F1 cv (prom. ± desvío) |
|---|---|---|---|---|---|
| A | Regresión logística | 0,29 | 1,00 | **0,44** | 0,19 ± 0,17 |
| A | Random Forest | 0,26 | 1,00 | **0,41** | 0,22 ± 0,11 |
| A | XGBoost | 0,23 | 0,67 | **0,34** | 0,20 ± 0,17 |
| B | Regresión logística | 0,33 | 0,92 | **0,49** | 0,39 ± 0,07 |
| B | Random Forest | 0,25 | 1,00 | **0,40** | 0,18 ± 0,16 |
| B | XGBoost | 0,26 | 0,83 | **0,39** | 0,18 ± 0,07 |
| Referencia | «Siempre alta» | 0,25 | 1,00 | **0,40** | — |
| Referencia | Aleatorio proporcional | 0,11 | 0,08 | **0,10** | — |

Cómo leerlo:

- **A (con anticipación):** la mejor, la regresión logística, saca 0,44 contra 0,40 de «siempre alta». Esa diferencia es menor que el desvío de su propia validación cruzada (0,17), así que **no se puede afirmar que haya señal**. Los tres modelos empatan entre sí y el reporte no declara ganador.
- **B (con potencia máxima):** la regresión logística llega a 0,49 con un desvío de 0,07 y sí supera la referencia con holgura. Es coherente con lo esperado: la potencia máxima casi *es* la demanda del mes, pero no se conoce de antemano.
- Los números exactos pueden variar en las últimas cifras entre máquinas por las versiones de las librerías. Cada corrida guarda en la base la semilla, el hash del dataset y las versiones usadas.

## Requisitos

- **Python 3.12** (desarrollado y probado con 3.12.3), en Linux.
- Unos **1,1 GB de disco** para las dependencias. La mayor parte es `xgboost`, que en Linux instala también librerías de NVIDIA (≈205 MB) que este proyecto **no usa**: corre solo en CPU. Es esperado, no hay nada que quitar.
- Las versiones exactas están fijadas en `requirements.txt` (pandas, numpy, scikit-learn, xgboost, streamlit, reportlab, matplotlib, joblib; para desarrollo pytest, ruff y coverage).

## Instalación

```bash
git clone <URL_DEL_REPO>
cd <carpeta-del-repo>

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

No hay que crear las bases ni configurar nada más: **las dos bases SQLite se crean solas** (vacías, con todas sus tablas) la primera vez que se arranca la app o el pipeline. Por eso `db/*.db`, `models/*.joblib` y `reports/*.pdf` no están en el repositorio.

## Uso

### Interfaz (Streamlit)

```bash
streamlit run app.py
```

Se abre en el navegador (por defecto en `http://localhost:8501`). Las páginas se recorren en orden desde la barra lateral:

1. **Datos:** carga y limpieza (ETL) y las tablas resultantes.
2. **Entrenamiento:** semilla visible y botón para entrenar los tres modelos en los dos escenarios. Con la grilla completa tarda menos de un minuto (unos 30 s en una PC de escritorio, ETL y reporte incluidos).
3. **Comparación:** tabla, matrices de confusión y gráficos de los seis resultados, con la referencia «siempre alta» y la recomendación.
4. **Predicción:** mes siguiente al último dato, con la temperatura esperada como entrada. Devuelve la probabilidad de demanda alta de cada modelo y la guarda.
5. **Reporte:** genera y descarga el PDF de la corrida.
6. **Logs:** los eventos registrados, filtrables.

### Línea de comandos

El mismo código corre sin interfaz:

```bash
python -m src.pipeline                 # ETL + entrenamiento
python -m src.pipeline --reporte       # además, genera el PDF en reports/
python -m src.pipeline --solo-etl      # solo carga y limpieza
python -m src.pipeline --seed 7        # otra semilla (por defecto, la de src/config.py)
python -m src.pipeline --help
```

El código de salida es 0 si todo salió bien, 1 si la corrida falló y 2 si los argumentos son inválidos. Cada corrida recibe un `run_id` que se imprime en el log y une las dos bases.

## Decisiones de diseño

- **Solo los 308 meses nacionales, no el formato largo por distribuidora.** La temperatura es nacional: las 24 distribuidoras de un mes se mueven juntas, así que habría ~3.600 filas pero apenas ~150 meses independientes, y se perderían los años anteriores a 2014.
- **Target relativo con umbral móvil, no fijo.** El cociente medio bajó de ~1,13 (2005) a ~1,00 (desde 2018): con un umbral fijo los positivos se concentraban en los primeros años y el hold-out final quedaba con 1 o 2. El umbral móvil mantiene ~25 % de positivos en todo el período. Costo: el primer mes con target es 2007-01.
- **Sin fuga de información.** Se excluyen del modelo la demanda del propio mes y sus componentes (definen el target). El target y los rezagos usan solo datos del pasado y hay un test que lo verifica.
- **Dos escenarios (A y B)** para separar lo que sirve para predecir (A) de lo que solo describe (B).
- **Hold-out de 48 meses y 3 cortes.** Con 48 meses el hold-out tiene 12 positivos (con 36 serían 7); con 3 cortes cada validación tiene al menos 6 positivos, con 4 o 5 había validaciones con 2 a 4.
- **Desbalance:** `class_weight="balanced"` en regresión logística y Random Forest, `scale_pos_weight` en XGBoost.
- **Qué se llama «mejor modelo».** Mejor = mayor F1 en el hold-out. Otro modelo **empata** si la diferencia es menor que el mayor de los dos desvíos de validación cruzada; entre empatados se elige el más simple (regresión logística < Random Forest < XGBoost). Y un modelo **solo se recomienda si supera a «siempre alta» por más que su propio desvío**; si no, el reporte dice que ninguno es recomendable. Las reglas están en `src/recommend.py` y los textos se arman con reglas explícitas en `src/explain.py`, no con frases fijas.
- **La lógica vive en `src/`.** `app.py` y `app_pages/` solo tienen interfaz, y el mismo código corre por línea de comandos.
- **Logger propio** (`src/logger.py`): un `SQLiteHandler` escribe en la tabla `logs` y el decorador `@log_step` registra cada paso del pipeline. No se usa `print` para seguimiento.
- **Una conexión SQLite por operación y modo WAL**, porque los hilos de Streamlit no comparten conexiones. Al terminar cada corrida se hace un checkpoint para que los `.db` queden completos, sin archivos `-wal` pendientes.

## Reproducibilidad

Por cada corrida se guarda en `unstructured.db` (tabla `configuracion`): la **semilla**, el **SHA-256 del dataset**, las **versiones de las librerías**, el **commit de Git** (y si el árbol tenía cambios sin commitear), los hiperparámetros y la configuración del target y de la validación. Cada modelo entrenado se guarda como `.joblib` en `models/`.

- La semilla sale siempre de `src/config.py` y se puede cambiar con `--seed` o en la página Entrenamiento.
- Con el mismo dataset, las mismas versiones y la misma semilla se obtienen los mismos resultados.
- Los PDF son deterministas: con los mismos datos salen los mismos bytes.
- **Seguridad de los `.joblib`:** son archivos pickle y cargar uno ajeno puede ejecutar código arbitrario. Por eso se guarda el SHA-256 de cada modelo en su documento de resultados y se **verifica antes de cargarlo**. Aun así, solo se deben cargar modelos generados por este proyecto.

## Las dos bases de datos

Se crean solas en `db/` (ver [Instalación](#instalación)).

| Base | Tabla | Contenido |
|---|---|---|
| `structured.db` (relacional) | `datos_limpios` | Los 308 meses limpios (Tabla 1) |
| | `logs` | Eventos del logger: `ts`, `level`, `module`, `message`, `run_id` |
| | `runs` | Una fila por corrida: `run_id`, inicio, fin, estado, semilla |
| `unstructured.db` (JSON) | `resultados_modelos` | Un documento JSON por corrida, escenario y modelo: parámetros, cv, métricas, matriz de confusión, predicciones mes a mes (Tabla 2) |
| | `configuracion` | Semilla, hash, versiones, commit y parámetros de la corrida (Tabla 3) |
| | `predicciones` | Cada predicción hecha en la app: entradas y probabilidad por modelo |

Los documentos tienen restricción `json_valid` y se consultan con `json_extract`. El `run_id` es la clave compartida entre las dos bases.

```sql
-- En unstructured.db: F1 de cada modelo en el hold-out de una corrida
SELECT scenario, model, ROUND(json_extract(doc, '$.holdout.f1'), 2) AS f1
FROM resultados_modelos
WHERE run_id = '<RUN_ID>'
ORDER BY f1 DESC;

-- En structured.db: consulta cruzada entre las dos bases con ATTACH
ATTACH DATABASE 'db/unstructured.db' AS unstructured;
SELECT r.run_id, r.status, m.scenario, m.model,
       ROUND(json_extract(m.doc, '$.holdout.f1'), 2) AS f1
FROM runs r
JOIN unstructured.resultados_modelos m ON m.run_id = r.run_id
WHERE r.run_id = '<RUN_ID>';
```

Los `run_id` disponibles salen de `SELECT run_id, started_at, status FROM runs;`.

## Estructura del repositorio

```
app.py                 Interfaz de Streamlit (solo UI)
app_pages/             Una página por módulo: datos, entrenamiento, comparación,
                       predicción, reporte y logs
src/
  config.py            Semilla, rutas y constantes
  logger.py            SQLiteHandler y @log_step
  db.py                Acceso a las dos bases (WAL, ATTACH, checkpoint)
  mongo.py             Conexión a MongoDB: credenciales del .env y cliente compartido
  data.py              Carga y limpieza (ETL)
  features.py          Variables derivadas, rezagos y target
  models.py            Pipelines y grillas de hiperparámetros
  train.py             Validación, búsqueda, evaluación y guardado
  results.py, queries.py  Escritura y lectura de resultados
  predict.py           Predicción con un modelo guardado
  recommend.py, explain.py  Regla de recomendación y su texto
  report*.py           Reporte PDF (datos, gráficos, secciones y diseño)
  pipeline.py          Línea de comandos: python -m src.pipeline
data/raw/              CSV original (no se modifica)
db/  models/  reports/ Bases, modelos y PDFs generados (fuera de Git)
tests/                 Suite de pytest
docs/plan_proyecto.md  Plan y decisiones del proyecto
```

## Calidad

```bash
pip install -r requirements.txt     # incluye pytest, ruff y coverage

pytest                              # suite completa (~2,5 min: entrena modelos reales)
pytest -m "not slow"                # subconjunto rápido (~20 s)
ruff check . && ruff format --check .   # PEP 8, docstrings, anotaciones de tipo
coverage run -m pytest && coverage report   # cobertura de líneas y ramas
```

La cobertura es del 100 % de líneas y ramas. Los tests cubren, entre otras cosas, que el target no usa datos futuros, la regla de recomendación (ganador claro, empate y referencia), la generación del PDF y el recorrido de la interfaz con `streamlit.testing`.

## Límites del modelo

- **Pocos datos.** Hay 236 meses con target y el hold-out tiene 48 meses con solo 12 de demanda alta: una predicción de más o de menos mueve mucho el F1. Las métricas son ruidosas.
- **No hay un modelo claramente mejor**, y en el escenario A **ninguno supera con claridad a «siempre alta»** (ver [Resultados](#resultados)). Predecir «alta» siempre da recall 1 y F1 0,40: un modelo con recall 1 y precisión baja no demuestra tener señal.
- **Solo el escenario A sirve para anticipar.** El B usa la potencia máxima, que no se conoce de antemano.
- **Las probabilidades no están calibradas.** No son «90 % de chances»: sirven para ordenar y comparar, no como probabilidades reales.
- **Nivel mensual.** No predice días ni horas, y el target es relativo (alto *para su época*), no un valor en MWh.
- **Un solo clima y un dato de temperatura nacional.** La zona (AMBA) es una inferencia y la temperatura mensual media oculta olas de calor o de frío dentro del mes.
- **La predicción usa una temperatura esperada que el usuario ingresa**: su error se propaga al resultado.
- **Fuente sin documentar del todo** (portal, fecha y licencia no registrados).
