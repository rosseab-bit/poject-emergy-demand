# Guion de ensayo: presentación de 8 a 10 minutos

Acompaña a `docs/presentacion.pptx` (8 diapositivas). Las mismas ideas están en las notas del orador de cada diapositiva. Los números salen de la corrida `6cc7dbd4` (semilla 42, grilla completa), guardada en MongoDB.

## Tiempos

| # | Diapositiva | Tiempo | Acumulado |
|---|---|---|---|
| 1 | Portada | 0:30 | 0:30 |
| 2 | El problema | 1:00 | 1:30 |
| 3 | Los datos | 1:00 | 2:30 |
| 4 | El target | 1:15 | 3:45 |
| 5 | Modelos y escenarios | 1:00 | 4:45 |
| 6 | Comparación | 1:30 | 6:15 |
| 7 | Predicción y trazabilidad (con demo) | 2:00 | 8:15 |
| 8 | Qué se puede afirmar y qué no | 0:45 | 9:00 |

Total: 9 minutos, con 1 de margen. Si te pasás, recortá la demo (diapositiva 7) a la consulta de MongoDB.

## Antes de empezar (2 minutos)

- Verificá que tenés **internet** y que el `.env` está completo: sin eso, la app no arranca (te lo dice con un mensaje claro).
- Tené la app abierta en otra pestaña: `streamlit run app.py`, con la corrida `6cc7dbd4` elegida en la barra lateral.
- Abrí en otra pestaña Atlas (*Browse Collections ▸ demanda_electrica ▸ logs ▸ Aggregation*) o MongoDB Compass, con la consulta lista para pegar (está más abajo).
- Verificá que la página **Predicción** carga sin errores: es la demo.
- Si falla la app o la conexión, la diapositiva 7 ya muestra la consulta y su resultado. Seguí con eso.

## Guion

**1. Portada (0:30).**
«Somos MatIAndo y presento un trabajo individual para Programación Avanzada en Ciencia de Datos. La pregunta: dado un mes, ¿su demanda eléctrica va a ser alta *para su época del año*? Les adelanto algo: el método quedó sólido, pero con tan pocos datos el resultado es modesto, y lo voy a mostrar tal cual.»

**2. El problema (1:00).**
«Es un clasificador binario a nivel mensual, no diario ni horario. El flujo completo: un CSV público, un ETL que deja los datasets limpios en SQLite, el cálculo de variables y target, tres modelos, y todo lo que registra cada corrida —logs, resultados y predicciones— en MongoDB, con la interfaz de Streamlit y un reporte PDF. Todo el código está en `src/` y corre igual desde la interfaz o con `python -m src.pipeline`.»

**3. Los datos (1:00).**
«308 meses nacionales, de 2001 a 2026. Guardo todos porque el target y los rezagos necesitan la demanda de 2001–2004 como historia, pero el target existe desde 2007: 236 meses, de los cuales 60 son de demanda alta, el 25 %. Dos supuestos que declaro: la fuente son datos públicos del gobierno argentino, pero no registré el portal exacto; y la temperatura es de Buenos Aires/AMBA, una zona que *inferí* comparando con normales climáticas, no la confirma la fuente.»

**4. El target (1:15).**
«"Alta" es relativo. En bruto el verano siempre gana, así que comparo cada mes con lo normal de su época: el cociente entre su demanda y el promedio del mismo mes de los 3 años anteriores. Es alta si ese cociente supera el cuantil 75 de los cocientes de los últimos 36 meses. Todo usa solo el pasado y hay un test que lo verifica. Probé un umbral fijo y no sirvió: los positivos quedaban concentrados al principio y el hold-out casi sin ejemplos. En el gráfico se ve que los positivos aparecen en todo el período, aunque por año varían de 0 a 9 de 12.»

**5. Modelos y escenarios (1:00).**
«Tres modelos: regresión logística como base, Random Forest y XGBoost. Dos escenarios. El A, el principal, usa 14 variables que se conocen de antemano. El B suma la potencia máxima, que ocurre en el mismo mes que la demanda, así que no sirve para anticipar: lo uso para medir cuánto aporta. La validación respeta el tiempo: tres cortes temporales y un hold-out con los últimos 48 meses, con 12 meses de demanda alta. Todo lo que aprende de los datos va dentro del Pipeline, ajustado solo con entrenamiento.»

**6. Comparación (1:30).**
«Métrica: F1 de la clase "alta". Pongo una referencia: la regla trivial de decir "alta" siempre, que da 0,40. En el escenario A la mejor es la regresión logística con 0,44, casi lo mismo, y la diferencia es menor que el desvío de su propia validación, 0,17. Los tres modelos empatan y el sistema no declara ganador, por una regla explícita. En B, con la potencia máxima, la logística llega a 0,49 con un desvío de 0,07 y sí supera la referencia, que es lo esperable: esa variable casi *es* la demanda del mes.»

**7. Predicción y trazabilidad (2:00).**
«*(Cambiar a la app.)* En Predicción tomo el mes siguiente al último dato, ingreso la temperatura esperada y el sistema completa los rezagos desde la base. *(Mover la temperatura y pulsar Predecir.)* Devuelve la probabilidad de "alta" de cada modelo y la guarda, así que sobrevive al reinicio. Aclaro que no está calibrada: no es "90 % de chances", sirve para comparar.
*(Cambiar a Atlas o a la diapositiva.)* Cada paso del pipeline queda en la colección `logs` de MongoDB, atado al `run_id`, con su nombre, el tipo de evento y la duración. Esta consulta lista los cuatro pasos más lentos de la corrida. Esos eventos están en MongoDB, y el mismo `run_id` une los logs con los resultados y las predicciones de la corrida.»

**8. Qué se puede afirmar y qué no (0:45).**
«Lo que sí: el método no usa datos futuros, los resultados se reproducen —en un clon limpio dieron idénticos y también al pasar a MongoDB— y hay 491 tests con cobertura total. Lo que no: no puedo afirmar que haya señal clara al anticipar, ni que un modelo sea mejor que otro, ni que las probabilidades estén calibradas. El valor del trabajo está en el método y la trazabilidad. Para seguir: más historia y temperatura diaria.»

## Demo paso a paso (diapositiva 7)

1. En la barra lateral, confirmá la corrida `6cc7dbd4`.
2. Entrá a **Predicción**: se ve el mes fijo y la temperatura por defecto.
3. Cambiá la temperatura (por ejemplo, 21,5) y pulsá **Predecir**.
4. Mostrá el mensaje «Predicción guardada», la tabla por modelo y la advertencia de calibración.
5. En Atlas (pestaña *Aggregation* de la colección `logs`) o en Compass, pegá este pipeline:
   ```js
   [
     { $match: { run_id: "6cc7dbd4", event: "fin" } },
     { $group: { _id: "$step", ms: { $max: "$duration_ms" } } },
     { $sort: { ms: -1 } },
     { $limit: 4 }
   ]
   ```
   Resultado esperado (corrida `6cc7dbd4`): `run_training_steps` 19298 ms, `train_all` 19277 ms, `train_model` 8496 ms y `search_hyperparameters` 8436 ms. Los pasos se anidan (el entrenamiento contiene a sus pasos internos), por eso los tiempos se superponen. Si repetís la consulta sobre otra corrida, los valores serán otros. Cada predicción o reporte nuevo sobre esa corrida suma eventos, pero no cambia estos cuatro pasos.
6. Alternativa desde Python, sin Atlas: `from src import mongo`, y la consulta equivalente con `mongo.get_database()["logs"].aggregate([...])` (hay ejemplos en el README).

## Preguntas probables

- **¿Por qué F1 y no accuracy?** Porque el 75 % de los meses son normales: decir «normal» siempre daría 75 % de accuracy sin aprender nada.
- **¿Por qué no se declara un ganador?** Con 12 positivos en el hold-out, la diferencia entre modelos es menor que el desvío de la validación cruzada: sería ruido.
- **¿Por qué ningún modelo supera a «siempre alta»?** Con pocos datos y un target relativo, la señal del clima y los rezagos no alcanza. Lo digo en el reporte en vez de esconderlo.
- **¿Para qué sirve el escenario B si no anticipa?** Para cuantificar cuánto aporta la potencia máxima; es un diagnóstico, no una herramienta de predicción.
- **¿Por qué un hold-out de 48 meses?** Con 36 el hold-out tenía 7 positivos; con 48, 12.
- **¿Hay fuga de información?** El target y los rezagos usan solo el pasado (hay un test), y imputación y escalado se ajustan dentro del Pipeline solo con entrenamiento.
- **¿De dónde salen los datos?** De fuentes públicas del gobierno de Argentina; el portal exacto y la fecha no quedaron registrados.
- **¿Se puede reproducir?** Cada corrida guarda semilla, hash del dataset, versiones y commit. En un clon limpio, las métricas salieron idénticas.
- **¿Por qué SQLite y MongoDB?** Los datasets, con columnas fijas, van a SQLite. Los logs, las corridas y los resultados son documentos con forma variable (cada modelo con sus parámetros, cada evento con o sin paso) y van a MongoDB. El `run_id` los une.
- **¿Qué pasa si no hay internet o falla MongoDB?** La app y el pipeline no arrancan y lo dicen con un mensaje claro, sin mostrar la clave. Es una dependencia que asumo y declaro en los límites.
- **¿Por qué los logs se mandan por lote?** Cada operación contra Atlas cuesta unos 30 ms; mandar los eventos de a uno sumaría varios segundos por corrida. Los errores salen al instante y, si el proceso muere de golpe, se pueden perder los eventos del paso en curso: es un compromiso que acepté.
