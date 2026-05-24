"""
Script de automatizacion para generacion de proyectos con ChatDev.

Lee proyectos desde data/programdev_dataset.json y ejecuta cada uno mediante
un flujo de ChatDev predefinido (ChatDev_v1.yaml).
Los artefactos de ejecucion se almacenan en WareHouse/ por el SDK de ChatDev.
"""

import json
import logging
import os
import sys
import time
import traceback
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv
from tqdm import tqdm

try:
    from chatdev import AgentConfig, run_workflow
except ImportError:
    print("Error: No se pudo importar chatdev. Asegurate de que esta instalado.")
    exit(1)

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT_DIR / "data" / "programdev_dataset.json"
YAML_PATH = ROOT_DIR / "yaml_instance" / "ChatDev_v1.yaml"
ENV_PATH = ROOT_DIR / ".env"


# ============================================================================
# CONFIGURACION DE LOGGING
# ============================================================================

def configure_logging():
    """Configura el logger para mostrar eventos por consola."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


logger = configure_logging()


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def load_environment_variables() -> dict:
    """
    Carga las variables de entorno desde el archivo .env.

    Returns:
        dict: Diccionario con API_KEY y BASE_URL.
    """
    load_dotenv(ENV_PATH)
    api_key = os.getenv("API_KEY")
    base_url = os.getenv("BASE_URL")
    num_iterations = os.getenv("NUM_ITERATIONS", "5")

    if not api_key or not base_url:
        raise ValueError("Error: API_KEY o BASE_URL no estan configuradas en .env")

    logger.info("Variables de entorno cargadas")
    logger.info(f"  - BASE_URL: {base_url}")
    logger.info(f"  - NUM_ITERATIONS: {num_iterations}")

    return {"api_key": api_key, "base_url": base_url, "num_iterations": int(num_iterations)}


def load_projects_dataset(dataset_path: Path) -> list:
    """
    Carga el archivo JSON con las descripciones de proyectos.

    Args:
        dataset_path: Ruta al archivo JSON.

    Returns:
        list: Lista de proyectos (dicts con 'project_name' y 'description').

    Raises:
        FileNotFoundError: Si el archivo no existe.
        json.JSONDecodeError: Si el JSON es invalido.
        ValueError: Si la estructura del JSON no es valida.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"No se encontro el archivo: {dataset_path}")

    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("El archivo JSON debe contener un array de proyectos")

    for idx, project in enumerate(data):
        if not isinstance(project, dict):
            raise ValueError(f"Proyecto {idx} no es un objeto JSON valido")
        if "project_name" not in project or "description" not in project:
            raise ValueError(
                f"Proyecto {idx} no contiene 'project_name' o 'description'"
            )

    logger.info(f"Dataset cargado: {len(data)} proyectos encontrados")
    return data


def build_agent_configs(env_vars: dict) -> dict:
    """
    Construye la configuracion de agentes para ChatDev.

    Args:
        env_vars: Diccionario con api_key y base_url.

    Returns:
        dict: Configuracion de agentes (agent_configs para run_workflow).
    """
    api_key = env_vars["api_key"]
    base_url = env_vars["base_url"]

    base_config = AgentConfig(
        provider="openai",
        model="qwen2.5-coder",
        api_key=api_key,
        base_url=base_url,
        temperature=1.0,
    )

    return {
        "Chief Executive Officer": base_config,
        "Chief Product Officer": base_config,
        "Chief Technology Officer": base_config,
        "Programmer": base_config,
        "Programmer Coding": base_config,
        "Programmer Code Complete": base_config,
        "Code Reviewer": base_config,
        "Programmer Code Review": base_config,
        "Programmer Test Error Summary": base_config,
        "Software Test Engineer": base_config,
        "Programmer Test Modification": base_config,
    }


@contextmanager
def suppress_console_output():
    """Suprime la salida por consola durante ChatDev."""
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        original_emit = logging.StreamHandler.emit

        def quiet_console_emit(handler, record):
            stream = getattr(handler, "stream", None)
            if stream in (sys.stdout, sys.stderr):
                return
            return original_emit(handler, record)

        logging.StreamHandler.emit = quiet_console_emit

        try:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                yield
        finally:
            logging.StreamHandler.emit = original_emit


def execute_chatdev_workflow(project_name: str, description: str, agent_configs: dict) -> dict:
    """
    Ejecuta el flujo de ChatDev para un proyecto especifico.

    Args:
        project_name: Nombre del proyecto.
        description: Descripcion/prompt del proyecto.
        agent_configs: Configuracion de agentes.

    Returns:
        dict: Registro con estado, tiempos y error (si aplica).
    """
    started_at = datetime.now()
    started_perf = time.perf_counter()

    try:
        logger.info(f"  -> Ejecutando flujo ChatDev para: {project_name}")

        with suppress_console_output():
            run_workflow(
                yaml_file=str(YAML_PATH),
                task_prompt=description,
                agent_configs=agent_configs,
            )

        finished_at = datetime.now()
        duration_seconds = round(time.perf_counter() - started_perf, 3)

        logger.info(f"  OK Completado: {project_name} ({duration_seconds:.3f}s)")

        return {
            "project_name": project_name,
            "status": "success",
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_seconds": duration_seconds,
            "error": "",
        }

    except Exception as e:
        error_msg = str(e)
        finished_at = datetime.now()
        duration_seconds = round(time.perf_counter() - started_perf, 3)

        logger.error(
            f"  ERROR en {project_name} ({duration_seconds:.3f}s): "
            f"{error_msg}"
        )

        return {
            "project_name": project_name,
            "status": "error",
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_seconds": duration_seconds,
            "error": error_msg,
        }


def warmup_ollama(base_url: str, model: str):
    """Realiza una petición vacía para cargar el modelo en RAM/VRAM."""
    native_url = base_url.replace("/v1", "/api/generate")
    if "/api/generate" not in native_url:
        native_url = f"{base_url.rstrip('/')}/api/generate"

    logger.info(f"Preparando motores: Cargando modelo '{model}' en Ollama...")
    try:
        requests.post(
            native_url,
            json={"model": model, "prompt": "", "stream": False},
            timeout=60,
        )
        logger.info("Modelo cargado y listo.")
    except Exception as e:
        logger.warning(f"El warm-up ha fallado (pero intentaremos seguir): {e}")


def main():
    """
    Funcion principal: orquesta la automatizacion de generacion de proyectos.
    """
    logger.info("=" * 70)
    logger.info("Iniciando automatizacion de generacion de proyectos con ChatDev")
    logger.info("=" * 70)

    try:
        env_vars = load_environment_variables()
        warmup_ollama(env_vars["base_url"], "qwen2.5-coder")

        projects = load_projects_dataset(DATASET_PATH)
        agent_configs = build_agent_configs(env_vars)

        successful_projects = 0
        failed_projects = 0

        num = env_vars["num_iterations"]
        logger.info(f"\nIniciando ejecucion de {len(projects)} proyectos...\n")
        logger.info(f"\nCada proyecto se ejecutará {num} veces...\n")

        total_experiments = len(projects) * num

        with tqdm(total=total_experiments, desc="Ejecutando experimentos", unit="exp") as progress_bar:
            for project in projects:
                base_name = project["project_name"]
                description = project["description"]
                target_project = projects[0]

                logger.info(f"\nIniciando ejecución del proyecto {base_name} {num} veces...\n")

                for i in range(num):
                    project_name = f"{base_name}_{i+1}"
                    description = target_project["description"]

                    execution_record = execute_chatdev_workflow(
                        project_name,
                        description,
                        agent_configs,
                    )

                    if execution_record["status"] == "success":
                        successful_projects += 1
                    else:
                        failed_projects += 1
                    progress_bar.update(1)

        logger.info("\n" + "=" * 70)
        logger.info("Generacion completada")
        logger.info(f"   - Proyectos exitosos: {successful_projects}")
        logger.info(f"   - Proyectos fallidos: {failed_projects}")
        logger.info(f"   - Total procesados: {len(projects)}")
        logger.info("   - Artefactos disponibles en: WareHouse/")
        logger.info("\n" + "=" * 70)

    except Exception as e:
        logger.error("\n" + "=" * 70)
        logger.error(f"Error critico en la ejecucion: {str(e)}")
        logger.error("\n" + "=" * 70)
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
