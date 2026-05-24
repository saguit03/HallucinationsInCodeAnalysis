# TFM - Análisis de la propagación de alucinaciones en sistemas multiagente mediante minería de trazas

Este repositorio contiene el flujo completo para generar trazas con ChatDev, etiquetarlas y analizar la propagación de alucinaciones sobre los logs resultantes.  

Este proyecto ha sido desarrollado por la estudiante *Sara Guillén Torrado* como parte del **Trabajo Final de Máster en Ciencia de Datos** en la *Universitat Oberta de Catalunya* (UOC).

## 0. Requisitos

### Entorno mínimo

- Python 3.12 o superior.
- `ollama` instalado y accesible desde terminal.
- Un entorno virtual creado en la raíz del repositorio en `.venv`.
- Acceso al modelo base que se va a utilizar en ChatDev y al modelo detector para la fase de etiquetado.

### Dependencias Python

Las dependencias declaradas en [pyproject.toml](pyproject.toml) cubren la creación de trazas, la lectura de los artefactos y el análisis posterior. Entre ellas están `chatdev`, `pandas`, `networkx`, `matplotlib`, `seaborn`, `scikit-learn`, `pyarrow`, `tqdm`, `requests`, `python-dotenv` e `ipykernel`.

### Preparación recomendada

1. Crear y activar el entorno virtual. Se recomienda usar `uv`.
2. Instalar las dependencias del proyecto.
3. Verificar que `ollama` responde y que los modelos necesarios están disponibles localmente.

En Windows, los scripts PowerShell asumen que el entorno se activa desde `.venv\Scripts\Activate.ps1`. En Unix/Linux, los wrappers usan `.venv/bin/activate`.

## Instalación rápida

```bash
git clone https://github.com/saguit03/HallucinationsInCodeAnalysis
cd HallucinationsInCodeAnalysis
uv sync
```

Después de instalar, comprueba que `ollama` está disponible y carga los modelos que vayas a usar antes de lanzar los scripts de trazas o detección.

## 1. Creación de trazas con ChatDev

La primera fase genera ejecuciones multiagente a partir del dataset del proyecto. La entrada principal está en [data/programdev_dataset.json](data/programdev_dataset.json), y la definición del workflow está en [yaml_instance/ChatDev_v1.yaml](yaml_instance/ChatDev_v1.yaml).

El script principal de esta fase es [create_chatdev_traces.py](create_chatdev_traces.py). Ese script:

- carga las variables de entorno desde `.env`, incluyendo `API_KEY`, `BASE_URL` y `NUM_ITERATIONS`;
- prepara la configuración de agentes para ChatDev;
- recorre los proyectos definidos en el dataset;
- ejecuta el flujo de ChatDev y guarda los artefactos en [WareHouse/](WareHouse).

Se proporciona un wrapper de lanzamiento que inicializa Ollama y el entorno virtual antes de ejecutar el script. En Unix/Linux es [scripts/create_traces_with_ollama.sh](scripts/create_traces_with_ollama.sh) y en Windows es [scripts/create_traces_with_ollama.ps1](scripts/create_traces_with_ollama.ps1).

El modelo que usa por defecto el wrapper de PowerShell es `qwen2.5-coder`. El script acepta como primer argumento el nombre del modelo a cargar.

Como salida, cada ejecución genera un directorio dentro de [WareHouse/](WareHouse) con el `execution_logs.json` correspondiente y el resto de artefactos de ChatDev. Ese directorio es la entrada de la siguiente fase.

## 2. Etiquetado de las trazas

La segunda fase recorre los runs almacenados en [WareHouse/](WareHouse) y etiqueta cada salida como `HALLUCINATION DETECTED` o `NO HALLUCINATION`.

El script de referencia es [scripts/detect_qwen.py](scripts/detect_qwen.py). Su comportamiento principal es:

- descubrir todos los runs que contienen `execution_logs.json`;
- extraer los eventos `NODE_END`;
- construir un prompt de evaluación con el contexto del workflow;
- llamar al modelo detector vía Ollama;
- escribir un `qwen_results.jsonl` por cada run.

El wrapper de lanzamiento con Ollama es [scripts/detect_qwen_with_ollama.sh](scripts/detect_qwen_with_ollama.sh). En Windows existe la variante [scripts/detect_qwen_with_ollama.ps1](scripts/detect_qwen_with_ollama.ps1).

Por defecto, el wrapper de PowerShell lanza el detector `qwen2.5-coder:latest`.

## 3. Análisis de las trazas

La fase de análisis parte de los runs ya etiquetados y extrae métricas para estudiar propagación, latencias y estructura del grafo de ejecución.

El punto de entrada interactivo es [analysis.ipynb](analysis.ipynb). Ese notebook usa directamente los módulos de [operations/](operations):

- [operations/traces.py](operations/traces.py) para cargar `execution_logs.json` y `qwen_results.jsonl` desde [WareHouse/](WareHouse);
- [operations/graphs.py](operations/graphs.py) para construir grafos dirigidos y métricas topológicas;
- [operations/propagation.py](operations/propagation.py) para estudiar contagio, nodos críticos y relaciones entre nodos;
- [operations/latency.py](operations/latency.py) para calcular latencias entre eventos;
- [operations/metrics_extraction.py](operations/metrics_extraction.py) para agregaciones por distancia y correlaciones;
- [operations/visualization.py](operations/visualization.py) para gráficos y figuras finales.

En el notebook se generan, entre otros, los CSV que ya están versionados como ejemplo en [results/](results):

- [results/warehouse_runs.csv](results/warehouse_runs.csv)
- [results/logs_df.csv](results/logs_df.csv)
- [results/metricas_grafos.csv](results/metricas_grafos.csv)

## Flujo completo reproducible

Si quieres ejecutar el pipeline entero de la generación y etiquetado de trazas:

```bash
./full_workflow_create_detect.sh
```

Ese wrapper delega en [scripts/full_workflow_create_detect.sh](scripts/full_workflow_create_detect.sh), que primero crea trazas con ChatDev y después ejecuta la detección con Qwen.

En Windows:

```powershell
.\full_workflow_create_detect.ps1
```

## Estructura de entradas y salidas

- Entrada del experimento: [data/programdev_dataset.json](data/programdev_dataset.json)
- Definición del workflow: [yaml_instance/ChatDev_v1.yaml](yaml_instance/ChatDev_v1.yaml)
- Definición del modelo ChatDev: [Modelfile](Modelfile)
- Salida de trazas: [WareHouse/](WareHouse)
- Salida etiquetada por run: `qwen_results.jsonl`
- Salidas agregadas del análisis: [results/](results)

## Notas prácticas

- Si vas a repetir tus propios experimentos, asegúrate de usar una copia limpia del repositorio o una rama de trabajo.
- Los scripts asumen que `ollama` está funcionando localmente antes de lanzar los modelos.
- El valor de `NUM_ITERATIONS` en `.env` controla cuántas veces se ejecuta cada proyecto del dataset.
- Si cambias el dataset o el YAML del workflow, el contenido de [WareHouse/](WareHouse) y de los análisis derivados también cambiará.

---

Aclaración: Este README ha sido generado por Copilot a partir de la estructura del proyecto y los scripts disponibles, y luego se ha revisado para asegurar que la información es precisa y coherente.
