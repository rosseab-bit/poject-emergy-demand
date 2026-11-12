# Plan del Proyecto Final: Clasificación de demanda eléctrica (Opción B)

Programación Avanzada en Ciencia de Datos. Documento vivo: se va puliendo.

## 1. Objetivo
Solución end-to-end (datos → modelos → base de datos → Streamlit) que **clasifica si la demanda eléctrica mensual de un mes es "alta" respecto de lo normal para su época**, a partir del clima y el calendario. Predicción a **nivel mensual** (no diaria ni horaria). Reproducible por otra persona.

## 2. Decisiones tomadas (con justificación)

| Tema | Decisión | Por qué |
|---|---|---|
| Dataset | `demanda_electricidad_original_mensual2.csv`, nacional, 2005-01 a 2026-08 (260 meses con todas las variables) | Trae demanda total y sectorial, temperatura y potencia máxima, que es lo que pide la Opción B |
| Granularidad | Solo los 260 meses nacionales | El formato largo por distribuidora daría ~3.600 filas, pero la temperatura es nacional, así que no agrega información: las 24 distribuidoras de un mes se mueven juntas (~150 meses independientes en vez de 260) y se perderían los años 2005-2013 |
| Target | Binario `demanda_alta` (1 = mes en el 25% superior respecto de su contexto) como base. Multiclase (normal/alta/crítica) como extra opcional | ~60 meses positivos alcanzan para binario. La clase "crítica" tendría unas 30 filas y métricas muy inestables |
| Contexto del target | Cociente entre la demanda del mes y el promedio del mismo mes calendario de los **3 años anteriores**. El mes es "alto" si su cociente supera el **cuantil 75 de los cocientes de los 36 meses anteriores** (umbral móvil; solo pasado, nunca futuro) | Descuenta la tendencia y la estacionalidad, y evita fuga de información. Se usa `demanda_total` desde 2001 como historia. Un umbral fijo no sirve: el cociente medio bajó de ~1,13 (2005) a ~1,00 (desde 2018), así que los positivos se concentraban en los primeros años y el hold-out final quedaba con 1 o 2 positivos. El umbral móvil mantiene ~25 % de positivos en todo el período y deja 7 en los últimos 36 meses. Costo: el primer mes con target es 2007-01 (236 filas en vez de 260) |
| Predictores | Temperatura (y derivadas), mes (seno/coseno), año y **rezagos de la demanda total y sectorial** (mes anterior y mismo mes del año anterior). Se **excluyen** la demanda del propio mes y sus componentes | La consigna pide usar demanda total y sectorial; los rezagos cumplen eso porque se conocen de antemano. La demanda del mismo mes haría que el modelo "haga trampa", porque el target se define con ella |
| Potencia máxima | Se entrenan **los dos escenarios** con los tres modelos (6 resultados): **A (anticipación)** sin potencia máxima y **B (diagnóstico)** con ella. **A es el principal** (entrenamiento de referencia y pantalla de predicción) | La consigna nombra la potencia máxima como variable, y B la cubre. Pero ocurre en el mismo mes que la demanda (corr. 0,96), así que no se conoce de antemano y solo A sirve para predecir. Comparar A contra B muestra cuánto aporta la potencia máxima. El costo es bajo con 260 filas |
| Modelos | Logistic Regression (línea base), Random Forest, XGBoost, comparados entre sí | Pedido del equipo; cumple "al menos dos" de la consigna |
| Validación | `TimeSeriesSplit` de 3 cortes + hold-out temporal final de los últimos 48 meses | Son pocos datos y una serie temporal: no se puede mezclar el futuro con el pasado. Con 48 meses el hold-out tiene 12 positivos (25 %), contra 7 con 36. Con 3 cortes cada validación tiene al menos 6 positivos; con 4 o 5 hay validaciones con 2 a 4 |
| Base estructurada | `structured.db` (SQLite relacional): `datos_limpios` (Tabla 1), `logs`, `runs` | Datos con columnas fijas y consultas SQL |
| Base no estructurada | `unstructured.db` (SQLite con JSON): `resultados_modelos` (Tabla 2), `configuracion` (Tabla 3) y `predicciones` | Cada modelo tiene parámetros y resultados con estructura distinta |
| Conexión entre bases | `run_id` compartido, consulta cruzada con `ATTACH DATABASE` | Trazabilidad entre resultado y logs |
| Predicciones por mes | Dentro del documento JSON de cada resultado | Todo lo que sale del modelo queda en `unstructured.db` |
| Interfaz | Todo desde Streamlit (datos, entrenamiento, comparación, predicción, reporte PDF, logs) con la lógica en módulos `src/` | Reproducibilidad y modularidad; el mismo código corre también por línea de comandos |
| Tabla 3 | Se hace | Ya se guarda la configuración como JSON, el costo extra es casi nulo |
| Reporte PDF | Desde Streamlit, después del ETL, el entrenamiento y (si se hizo) la predicción, se descarga un PDF generado por `src/report.py` con ReportLab y gráficos de matplotlib, a partir de lo guardado en las bases y del `run_id`. Contiene: portada (run_id, fecha, semilla, hash del dataset, versiones), datos y ETL (filas, balance de clases, supuestos), resultados de cada modelo y escenario con matrices de confusión, tabla y gráfico comparativos, modelo recomendado con sus fundamentos, predicciones realizadas y límites | Resume la corrida y la deja trazable y reproducible, atada al `run_id` |
| Criterio de "mejor modelo" | Métrica principal: F1 de la clase "alta" en el hold-out, junto con el promedio y el desvío en validación cruzada. Si la diferencia entre modelos es menor que ese desvío, el reporte dice "sin ganador claro" y desempata por simplicidad (Logistic Regression < Random Forest < XGBoost). El "por qué" se redacta con reglas explícitas a partir de las métricas y de los coeficientes o importancias de variables, no con textos fijos. A y B se comparan por separado; la recomendación para predecir sale de A | Con ~236 meses con target y un hold-out de 48, las métricas son ruidosas: declarar un ganador sin mirar el ruido no sería honesto |
| Predicciones en el reporte | Cada predicción hecha en Streamlit se guarda en `unstructured.db` (colección `predicciones`: entradas, probabilidad por modelo y `run_id`) | Streamlit vuelve a ejecutar el script en cada interacción, así que el PDF solo puede incluir lo que quedó guardado |
| Git | Solo la persona usuaria hace commit, push y merge. Claude Code entrega el mensaje de commit sugerido, el momento de push/merge y los comandos. Una rama por etapa, una tarea por commit, merge local con `--no-ff`. Sin issues ni pull requests | Que el control del repositorio sea siempre humano y que el historial quede legible para el criterio de Git. Las reglas completas están en `CLAUDE.md` |
| Equipo y ritmo | Trabajo individual, desarrollado con Claude Code. Sin fecha fija: el avance depende del tiempo disponible (de 1 día a 1 semana) | El plan se organiza en tres niveles (sección 9) para poder cortar y entregar algo completo en cualquier punto |
| Zona de la temperatura | Se asume **Buenos Aires / AMBA** (inferido, no confirmado por la fuente) | La zona la define el dataset, no se puede elegir. Se comparó el promedio mensual de `temperatura_promedio` (2005-2025) con las normales 1991-2020 de la ciudad de Buenos Aires: la forma estacional coincide y el dataset resulta entre 0,3 y 1,0 °C más cálido en todos los meses, coherente con el calentamiento reciente. Evidencia: dataset = 25,6 / 24,8 / 22,3 / 18,9 / 15,2 / 12,5 / 11,8 / 13,3 / 15,4 / 18,2 / 21,5 / 24,0; normales de Buenos Aires = 24,9 / 23,8 / 22,0 / 18,2 / 14,8 / 12,0 / 11,0 / 13,0 / 14,9 / 17,9 / 20,9 / 23,6 (enero a diciembre; normales tomadas de una fuente secundaria, no del SMN). Debe figurar en el README como supuesto |

## 3. Reproducibilidad
Por cada corrida se guarda: **semilla**, **hash del dataset**, **versiones de librerías**, hiperparámetros, hash del commit de Git, y el **modelo entrenado** (`.joblib`). Con eso se puede reentrenar idéntico o cargar el modelo y predecir.

## 4. Estructura del repositorio
```
tp_demanda/
├── README.md
├── requirements.txt        # versiones fijadas (incluye xgboost, streamlit, reportlab)
├── CLAUDE.md               # contexto y convenciones para Claude Code
├── .gitignore
├── app.py                  # Streamlit (solo interfaz)
├── data/raw/               # CSV original
├── db/                     # structured.db y unstructured.db (se crean solas)
├── models/                 # modelos .joblib
├── reports/                # PDFs generados (en .gitignore)
├── src/
│   ├── config.py           # semilla, rutas, constantes
│   ├── logger.py           # SQLiteHandler + decorador @log_step
│   ├── db.py               # acceso a las dos bases
│   ├── data.py             # carga y limpieza
│   ├── features.py         # variables derivadas y target
│   ├── models.py           # pipelines y grillas
│   ├── train.py            # validación, búsqueda, evaluación, guardado
│   ├── predict.py          # predicción con modelo guardado
│   ├── report.py           # generación del reporte PDF
│   └── pipeline.py         # ejecución por línea de comandos
├── tests/
└── docs/                   # diapositivas y notas
```

## 5. Etapas (cada una cierra con commit y un criterio de "terminado")

**Etapa 0. Setup.** Repo, `.gitignore`, estructura, `requirements.txt`, `config.py` con la semilla, `CLAUDE.md` en la raíz y este plan copiado en `docs/plan_proyecto.md`. *Listo cuando:* `pip install -r requirements.txt` funciona en un entorno limpio.

**Etapa 1. Logger y capa de base de datos.** `SQLiteHandler`, `@log_step`, tablas `logs` y `runs`, `run_id`, conexión por operación (por los hilos de Streamlit), modo WAL. *Listo cuando:* un test escribe y lee eventos de `logs`.

**Etapa 2. Carga y limpieza → Tabla 1.** Leer el CSV, tratar los dos valores repetidos de potencia máxima (2002-03, 2021-08: se ponen en nulo y se marcan con `potencia_repetida`; el nulo lo imputa el `Pipeline`), validar que los nulos de sectores sean solo previos a 2005 (no se imputan), averiguar la zona de la temperatura. Guardar en `datos_limpios` los 308 meses (2001-01 a 2026-08), porque el target y los rezagos necesitan la `demanda_total` de 2001-2004 como historia. *Listo cuando:* la tabla tiene 308 filas, de las cuales 259 están completas (las 260 de 2005 en adelante, menos 2021-08 por la potencia repetida), y todo quedó logueado.

**Etapa 3. Variables y target.** Distancia a ~18 °C o grados-día de frío y calor, mes cíclico, año, rezagos de demanda total y sectorial (1 y 12 meses; el rezago de 12 meses de los sectores es nulo en 2005, pero esos meses no tienen target con el umbral móvil, así que no queda ningún nulo por tratar), target relativo con umbral móvil de 36 meses (solo pasado, ver sección 2), conjuntos A y B. Temperatura como grados-día de frío y calor con base de 18 °C (`config.py`), calculados con la temperatura media mensual (aproximación: el dataset no trae temperaturas diarias). *Listo cuando:* hay un test que verifica que el target no usa datos futuros y se informa el balance de clases (236 filas con target desde 2007-01, ~25 % de positivos).

**Etapa 4. Modelos.** `Pipeline` de scikit-learn (imputación, escalado, codificación) con los tres modelos y los dos escenarios (A y B), manejo del desbalance (`class_weight`, `scale_pos_weight`), búsqueda de hiperparámetros con `TimeSeriesSplit`, hold-out temporal final, métricas (accuracy, precision, recall, F1) y matriz de confusión. Misma semilla y mismos cortes para los tres. *Listo cuando:* los seis resultados (3 modelos × 2 escenarios) se entrenan por línea de comandos y dan métricas.

**Etapa 5. Guardado de resultados → Tabla 2 y 3.** Un JSON por modelo con parámetros, scores de validación cruzada, métricas, matriz de confusión y predicciones; `.joblib` del modelo; semilla, hash y versiones. *Listo cuando:* se puede recuperar un resultado con `json_extract` y el modelo guardado reproduce las mismas predicciones.

**Etapa 6. Streamlit.** Páginas: Datos, Entrenamiento (botón + semilla visible), Comparación de los tres modelos (tabla, matrices de confusión, gráficos), Predicción, Reporte (descarga del PDF), Logs. *Listo cuando:* alguien en otra máquina ejecuta `streamlit run app.py` y llega al resultado sin tocar código.

**Etapa 7. Predicción.** Formulario (mes, año, temperatura esperada) que carga el modelo guardado con el conjunto A, completa los rezagos desde `datos_limpios` y devuelve la probabilidad de demanda alta con cada uno de los tres modelos, con aclaración de que es nivel mensual. Cada predicción se guarda en `unstructured.db` (colección `predicciones`) para poder incluirla en el reporte. El uso natural es el mes siguiente al último dato disponible. *Listo cuando:* funciona después de reiniciar la app.

**Etapa 8. Reporte PDF.** `src/report.py` arma el PDF con ReportLab y gráficos de matplotlib, leyendo las bases y el `run_id` elegido. Secciones: portada (run_id, fecha, semilla, hash del dataset, versiones), datos y ETL (filas, balance de clases, supuestos), resultados por modelo y escenario con matrices de confusión, comparación (tabla y gráfico con promedio y desvío de validación cruzada), modelo recomendado y por qué (texto armado con reglas explícitas desde métricas, coeficientes e importancias; dice "sin ganador claro" si la diferencia está dentro del ruido), predicciones realizadas y límites. Botón de descarga en la página Reporte. *Listo cuando:* después de entrenar y predecir en Streamlit se descarga un PDF completo, y hay tests de la regla de recomendación (ganador claro y empate) y de la generación del PDF sin abrir la interfaz.

**Etapa 9. Calidad.** `pytest`, PEP 8 con `ruff` o `flake8`, docstrings, revisión de funciones y clases. *Listo cuando:* tests y linter pasan.

**Etapa 10. README y prueba en limpio.** README con fuentes, decisiones, pasos de ejecución y límites del modelo. Clonar en una carpeta nueva y ejecutar todo desde cero. *Listo cuando:* funciona sin ayuda.

**Etapa 11. Presentación.** 6 a 8 diapositivas (problema, datos, target, modelos, comparación, predicción, límites) y ensayo de 8 a 10 minutos. Mostrar una consulta SQL sobre `logs`.

**Opcional (si da el tiempo):** target multiclase.

## 6. Cosas fáciles de olvidar
- Documentar de dónde viene el dataset y la zona de la temperatura.
- Guardar semilla, hash del dataset y versiones (no solo la semilla).
- Dejar fuera del modelo las variables que componen el target.
- Dar el mismo esquema de validación a los tres modelos.
- Manejar el desbalance de clases.
- Aclarar que la predicción es mensual y que la potencia máxima no sirve para anticipar.
- Que las bases `.db` queden accesibles para la revisión.
- `.gitignore` (entorno virtual, `__pycache__`, modelos pesados si hace falta).
- XGBoost, Streamlit y ReportLab en `requirements.txt`.
- El PDF sale de lo guardado en las bases (no de la memoria de Streamlit) y no declara un ganador si la diferencia está dentro del ruido de la validación cruzada.
- Commits chicos y con mensajes claros (parte del criterio de Git).
- Probar todo en una máquina o carpeta limpia antes de entregar.

## 7. Cobertura de la rúbrica
| Criterio (25% c/u) | Dónde se cubre |
|---|---|
| Código y Git | Etapas 0, 1, 9, 10 |
| Base de datos y visualización | Etapas 1, 2, 5, 6, 8 |
| Modelado y resultados | Etapas 3, 4, 5, 8 y la tabla de decisiones |
| Presentación oral | Etapa 11 |

## 8. Pendientes por definir
- Confirmar la zona de la temperatura en la fuente original del dataset (hoy es un supuesto, ver sección 2).
- Cómo se entregan las bases `.db` (versionadas en el repo o generadas con un script), se decide al final.
- Si la app permite subir otro CSV: la propuesta es que no, y que use solo el CSV del repo, para que el experimento sea reproducible (ver sección 11).
- Target multiclase (opcional): definir sus umbrales con el mismo criterio móvil.

## 9. Ritmo de trabajo (sin fecha fija)
Estimaciones orientativas de dedicación con Claude Code; dependen del tiempo real. Cada nivel termina en algo entregable.

- **Nivel 1, núcleo (≈ 1 día intenso):** Etapas 0 a 5, una versión simple de la 6 (Streamlit con entrenamiento y comparación), README básico y presentación mínima. Cubre todos los requisitos de la consigna.
- **Nivel 2, completo (≈ 2 a 3 días):** suma la predicción (Etapa 7), el reporte PDF (8), tests y linter (9), prueba en carpeta limpia (10) y presentación ensayada (11).
- **Nivel 3, con extras (≈ 1 semana):** suma el target multiclase, pulido de gráficos e interfaz, e importancia de variables y análisis A vs B.

Regla: no pasar al nivel siguiente hasta que el anterior corra de punta a punta en una carpeta limpia.

## 10. Flujo de Git
Reglas completas en `CLAUDE.md`. Resumen:
- Solo la persona usuaria ejecuta `add`, `commit`, `push` y `merge`. Claude Code solo usa comandos de lectura.
- Una rama por etapa (`etapa-N-nombre`), una tarea por commit, merge local con `--no-ff` al cerrar la etapa.
- Claude Code se detiene después de cada tarea con el mensaje de commit sugerido y los comandos, y al cerrar la etapa con los comandos de push y merge.
- Sin issues ni pull requests: el seguimiento es por commits, ramas y merge.
- Mensajes de commit: `tipo: resumen en imperativo` (feat, fix, test, docs, refactor, chore).

## 11. Flujo en Streamlit (qué ve la persona usuaria)
Propuesta a confirmar. Una barra lateral y seis páginas, en este orden. Cada página posterior avisa "Primero ejecutá <paso>" si falta algo anterior. Todo queda guardado en las bases, así que al cerrar y reabrir la app los resultados siguen ahí.

**Barra lateral (siempre visible):** `run_id` de la corrida actual, semilla (editable, con el valor por defecto de `config.py`), hash del dataset, estado de cada paso (ETL, Entrenamiento, Predicción, Reporte: hecho o pendiente) y un selector para abrir una corrida anterior.

1. **Datos (ETL).** Resumen del proyecto y de los supuestos (zona de la temperatura, nivel mensual). Botón "Ejecutar ETL". Después muestra: filas leídas y filas limpias (260), correcciones hechas (valores repetidos, nulos), definición del target (3 años, umbral) con el balance de clases, la tabla `datos_limpios` y gráficos (serie de demanda, estacionalidad, temperatura vs. demanda).
2. **Entrenamiento.** Elección de escenarios (A, B o ambos), de modelos (Logistic Regression, Random Forest, XGBoost) y de la semilla. Botón "Entrenar" con progreso por modelo. Al terminar, tabla resumen de métricas y aviso de que los resultados se guardaron.
3. **Comparación.** Tabla de modelo × escenario con accuracy, precision, recall y F1 (promedio y desvío en validación cruzada, y valor en el hold-out), gráfico de barras, matrices de confusión lado a lado, coeficientes o importancias de variables, comparación A vs. B y un panel "Modelo recomendado" con el porqué (o "sin ganador claro").
4. **Predicción.** Formulario con mes, año y temperatura esperada. Muestra los rezagos que completa solo desde `datos_limpios`. Botón "Predecir" y resultado con la probabilidad de demanda alta de cada uno de los tres modelos y la clasificación (alta o normal), con la aclaración de que es a nivel mensual. Tabla con el historial de predicciones guardadas.
5. **Reporte.** Elección de la corrida, opción de incluir las predicciones, botón "Generar PDF" y descarga. Muestra el índice de lo que contiene.
6. **Logs.** Tabla de `logs` de la corrida con filtros por nivel y módulo, errores resaltados y lista de corridas (`runs`) con su estado.

### Checklist de completitud
| Qué se espera | Dónde se ve |
|---|---|
| ETL ejecutado desde la app y datos limpios en la base (Tabla 1) | Página 1 |
| Tres modelos entrenados con búsqueda de hiperparámetros | Página 2 |
| Accuracy, precision, recall, F1 y matriz de confusión | Página 3 |
| Comparación entre modelos y explicación de cuál es mejor | Página 3 y PDF |
| Predicción después de entrenar | Página 4 |
| PDF con entrenamiento, predicción y comparación | Página 5 |
| Todo lo que hace el código registrado en SQLite | Página 6 |
| Reproducibilidad (semilla, hash, versiones, `run_id`) | Barra lateral y PDF |
