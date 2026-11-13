# CLAUDE.md: TP Final Programación Avanzada (demanda eléctrica)

## Contexto
Proyecto individual de clasificación de demanda eléctrica mensual ("demanda alta" respecto de lo normal para su época), con scikit-learn, SQLite para los datasets y MongoDB para los logs, las corridas y los resultados, logger propio e interfaz en Streamlit. Todas las decisiones y las etapas están en `docs/plan_proyecto.md`. **Leerlo antes de empezar cada etapa.** Si algo del plan parece mal, proponer el cambio y esperar respuesta; no cambiarlo por cuenta propia.

## Reglas de Git (obligatorias)

1. **Solo la persona usuaria hace commit, push y merge.** Claude Code nunca ejecuta comandos de git que modifiquen el repositorio: `add`, `commit`, `push`, `merge`, `switch`, `checkout`, `branch`, `tag`, `stash`, `rebase`, `reset`, `cherry-pick`.
2. Comandos de git permitidos (solo lectura): `status`, `diff`, `log`, `show`, `branch --show-current`.
3. **Una tarea = un commit.** Al terminar cada tarea, Claude Code se detiene y entrega el bloque "Listo para commit" (formato abajo). No sigue con la tarea siguiente hasta que la persona usuaria confirme que hizo el commit.
4. **Una rama por etapa** (`etapa-N-nombre`). `main` siempre debe quedar funcional. Si la rama actual no corresponde a la etapa, avisar antes de escribir código y dar el comando para cambiarla.
5. Al cerrar la etapa (cuando se cumple su "Listo cuando", pasan los tests y pasa el linter), Claude Code entrega el bloque "Momento de push y merge" con los comandos exactos.
6. **Sin issues ni pull requests.** El seguimiento es solo por commits, ramas y merge local con `--no-ff`.
7. Nunca sugerir `push --force`, `reset --hard` ni `rebase` salvo pedido explícito. Si hay conflicto en un merge, explicar qué pasó y proponer la resolución sin comandos destructivos.
8. No pasar a la etapa siguiente hasta que la persona usuaria confirme el merge.

### Formato: "Listo para commit"
```
Tarea: <qué se hizo, una línea>
Archivos: <lista de archivos creados o modificados>
Verificación: <qué se probó y resultado>

git add <archivos>
git commit -m "<mensaje sugerido>"
```

### Formato: "Momento de push y merge"
```
Etapa N cerrada. Criterio cumplido: <cuál>. Tests y linter: OK.

git push -u origin etapa-N-nombre
git switch main
git pull
git merge --no-ff etapa-N-nombre -m "merge: etapa N, <nombre>"
git push
git branch -d etapa-N-nombre
```

### Comandos de integración (plantilla)
Primera vez, una sola vez:
```
git init
git branch -M main
git remote add origin <URL_DEL_REPO>
git add README.md .gitignore
git commit -m "chore: inicializa repositorio"
git push -u origin main
```
Al empezar cada etapa:
```
git switch main
git pull
git switch -c etapa-N-nombre
```

### Ramas por etapa
`etapa-0-setup`, `etapa-1-logger-db`, `etapa-2-limpieza`, `etapa-3-features-target`, `etapa-4-modelos`, `etapa-5-resultados`, `etapa-6-streamlit`, `etapa-7-prediccion`, `etapa-8-reporte-pdf`, `etapa-9-calidad`, `etapa-10-readme`, `etapa-11-presentacion`.

### Convención de commits
`tipo: resumen en imperativo`, en español, hasta 72 caracteres. Tipos: `feat`, `fix`, `test`, `docs`, `refactor`, `chore`. Ejemplo: `feat: agrega SQLiteHandler para el logger`.

## Convenciones de código
- Python con PEP 8 (verificar con `ruff`), docstrings en funciones y clases, type hints, funciones chicas.
- La lógica vive en `src/`. `app.py` solo tiene interfaz y llama a los módulos. El mismo código debe poder correr por línea de comandos (`python -m src.pipeline`).
- La semilla sale de `src/config.py`. Nunca escribirla a mano en otro lado.
- Nada de `print` para seguimiento: usar el logger, que escribe en la colección `logs` de MongoDB. Decorar los pasos del pipeline con `@log_step`.
- Todo lo que **aprende de los datos** (imputación, escalado, codificación) va dentro del `Pipeline` de scikit-learn y se ajusta solo con entrenamiento. El target y los rezagos son reglas fijas y se calculan en el ETL usando solo datos del pasado, con un test que lo verifique.
- Los datos crudos de `data/raw/` no se modifican nunca.
- El reporte PDF se genera en `src/report.py` a partir de lo guardado en las bases, no de la memoria de Streamlit. La conclusión de "qué modelo es mejor" se redacta con reglas explícitas sobre las métricas y no declara ganador si la diferencia está dentro del ruido de la validación cruzada.
- Conexiones SQLite abiertas por operación (los hilos de Streamlit no las comparten), modo WAL. MongoDB, en cambio, usa un solo `MongoClient` compartido (`src/mongo.py`) y sus credenciales salen del `.env`, que nunca se sube.
- Cada dependencia nueva se agrega a `requirements.txt` con versión fijada y se avisa antes.
- `.db`, `.joblib`, los PDF de `reports/` y el `.env` van en `.gitignore`. La base de MongoDB vive en el cluster y no se versiona. Cómo se entregan los datos para la revisión se decide al final.

## Flujo de trabajo por etapa
1. Leer la etapa en `docs/plan_proyecto.md` y confirmar la rama.
2. Dividir la etapa en tareas chicas, una por commit.
3. Por cada tarea: implementar, probar, entregar el bloque "Listo para commit" y esperar confirmación.
4. Al cerrar la etapa: verificar el "Listo cuando", correr tests y linter, entregar el bloque "Momento de push y merge".
5. Esperar la confirmación del merge antes de empezar la etapa siguiente.
