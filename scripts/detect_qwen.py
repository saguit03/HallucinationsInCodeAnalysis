"""
Script de detección de hallucinations en outputs de ChatDev.

Lee ejecuciones desde WareHouse/ y etiqueta cada output usando el modelo
qwen2.5-coder-hallu-detector a través de Ollama.
Los resultados se almacenan en qwen_results.jsonl por cada run.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_WAREHOUSE_DIR = ROOT_DIR / "WareHouse"


def build_hallucination_prompt(
    output: str,
    previous_output: str | None = None,
    project_context: str | None = None,
    node_id: str | None = None,
) -> str:
    previous_context = f"\nPrevious output from the workflow:\n{previous_output}\n" if previous_output else ""
    node_context = f"Node being reviewed: {node_id or 'unknown'}\n" if node_id else ""

    return f"""You are an expert Software Engineering Evaluator and an objective hallucination detector for ChatDev workflow outputs. Your core expertise is analyzing software development life cycle (SDLC) traces, encompassing both natural language (requirements, design, manuals), file operations, and source code.

Task:
Evaluate the current output to determine if it contains hallucinations, logical inconsistencies, or scope deviations based on the GLOBAL project context, standard software engineering practices, and the type of content provided.

Global Project Context:
{project_context or 'No explicit context provided. Use only the previous output and inherent software development logic to infer the project context.'}

Constraints and guidelines (Content-Type Specific Evaluation):

1. WHEN THE OUTPUT CONTAINS ACTUAL CODE:
   - Primary Focus: Syntax correction, structural validity, and logical viability.
   - Context Handling: DO NOT over-analyze the global domain context for general programming tasks. Boilerplate code, project setup (e.g., creating virtual environments, installing packages, generic `main` modules, setting up imports), and standard testing procedures are valid starting points for ANY project. They are NO HALLUCINATION even if they do not explicitly mention the specific domain (e.g., Checkers).
   - Only mark code as hallucination if it contains blatant syntax errors, completely invalid programming logic, or implements highly specific business logic from a totally unrelated domain.

2. WHEN THE OUTPUT IS PURE NATURAL LANGUAGE (No Code):
   - Primary Focus: Global Project Coherence and relation to the context.
   - The text must logically align with the target project domain. It is completely normal for the output to deduce inherent rules or tasks (e.g., deducing a board layout for a game). 
   - Mark as hallucination ONLY if it introduces features or concepts from a completely unrelated domain (e.g., discussing Chess rules in a Checkers project) or drastically changes the project scope without reason.

3. WHEN THE OUTPUT IS PSEUDOCODE OR HIGH-LEVEL DESIGN:
   - Primary Focus: Architectural coherence and logical progression.
   - Placeholders, `pass` statements, missing implementations, and incomplete functions are fully expected in planning or early design phases. Do not penalize them. Mark as hallucination only if the architecture fundamentally contradicts the project goal.

4. STANDARD SDLC OPERATIONS & TOOL CALLS:
   - File system manipulations (creating folders, saving files, creating `__init__.py`) and workflow tool calls (e.g., `describe_available_files`, `modify_code_based_on_error`) are inherent to the development process. They are NEVER hallucinations.

5. FINAL VERDICT:
   - Write your detailed step-by-step reasoning first in the Explanation field.
   - Based strictly on your explanation, output exactly one of these labels at the very end: HALLUCINATION DETECTED or NO HALLUCINATION. Be coherent.

Response format:
Explanation: [Step-by-step justification based on whether the output is code, text, or pseudocode, evaluating syntax/logic or domain coherence accordingly]
Label: [HALLUCINATION DETECTED / NO HALLUCINATION]

Few-shot examples:

Example 1 (Pure Natural Language):
Previous output from the workflow:
"We are developing a Checkers game. I have created the Board class."
Current output to analyze:
"Next, I will implement the Knight piece, which moves in an L-shape, and the Bishop, which moves diagonally."
Response:
Explanation: The output contains no code. Evaluating based on Global Project Coherence, it introduces elements (Knight, Bishop) that belong to Chess, not Checkers. This contradicts the domain context.
Label: HALLUCINATION DETECTED

Example 2 (Pseudocode):
Previous output from the workflow:
"Goal: Create a Checkers game. Let's design the piece movement."
Current output to analyze:
"class Piece: \n    def move(self, start, end):\n        # TODO: Implement logic\n        pass"
Response:
Explanation: The output is pseudocode/incomplete code during a design phase. Evaluating for architectural coherence, setting up a Piece class with a move method placeholder is completely logical. The incomplete state is expected.
Label: NO HALLUCINATION

Example 3 (Actual Code / Boilerplate):
Previous output from the workflow:
"Let's start the implementation of the Checkers game."
Current output to analyze:
"First, I will set up the project. I will create a python virtual environment, install necessary basic packages, and create a standard `main.py` entry point that prints 'Starting application...'."
Response:
Explanation: The output describes writing actual setup code and project configuration. Evaluating based on actual code rules, creating virtual environments and boilerplate `main` modules are standard, correct SDLC operations. We do not over-analyze the lack of Checkers-specific logic here because it is a general setup step.
Label: NO HALLUCINATION

{node_context}{previous_context}
Current output to analyze:
{output}

Response:"""


def discover_runs(warehouse_dir: Path) -> list[Path]:
    """Descubre todos los runs en WareHouse que contienen execution_logs.json."""
    return sorted(path.parent for path in warehouse_dir.rglob("execution_logs.json"))


def extract_node_outputs(execution_log_path: Path) -> tuple[str | None, list[dict[str, Any]]]:
    """
    Extrae todos los outputs de los NODE_END events en execution_logs.json.

    Returns:
        tuple: (workflow_id, lista de outputs con metadatos)
    """
    with execution_log_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    workflow_id = data.get("workflow_id")
    outputs: list[dict[str, Any]] = []

    for log in data.get("logs", []):
        if log.get("event_type") != "NODE_END":
            continue
        details = log.get("details", {})
        output = details.get("output")
        if not isinstance(output, str):
            continue
        output = output.strip()
        if not output:
            continue
        outputs.append(
            {
                "node_id": log.get("node_id") or "Desconocido",
                "timestamp": log.get("timestamp"),
                "workflow_id": workflow_id,
                "texto": output,
            }
        )

    return workflow_id, outputs


def extract_response_fields(raw_response: str) -> tuple[str, str]:
    label = ""
    explanation = ""
    label_match = re.search(r"(?im)^\s*label\s*:\s*(.+?)\s*$", raw_response)
    if label_match:
        label = label_match.group(1).strip()
    else:
        for candidate in ("HALLUCINATION DETECTED", "NO HALLUCINATION", "UNKNOWN"):
            if candidate in raw_response.upper():
                label = candidate
                break
    explanation_match = re.search(r"(?ims)^\s*explanation\s*:\s*(.*?)(?:^\s*label\s*:|\Z)", raw_response)
    if explanation_match:
        explanation = explanation_match.group(1).strip()
    return label, explanation


def normalize_qwen_score(label: str, explanation: str, raw_response: str) -> str:
    """Normaliza la salida del modelo a una de las dos etiquetas permitidas."""
    combined_text = f"{label}\n{explanation}\n{raw_response}".upper()

    negative_markers = (
        "NO HALLUCINATION",
        "NO HALLUCINATIONS",
        "DOES NOT CONSTITUTE HALLUCINATION",
        "DOESN'T CONSTITUTE HALLUCINATION",
        "DO NOT CONSTITUTE HALLUCINATION",
        "THERE ARE NO HALLUCINATIONS DETECTED",
        "THERE IS NO HALLUCINATION",
        "THIS DOES NOT CONSTITUTE HALLUCINATION",
        "THIS DOES NOT NECESSARILY INDICATE HALLUCINATION",
        "NO HALLUCINATION DETECTED",
        "NOT A HALLUCINATION",
    )
    positive_markers = (
        "HALLUCINATION DETECTED",
        "FALSE OR FABRICATED",
        "FALSE",
        "FABRICATED",
        "MISLEADING",
        "INACCURATE",
        "INVALID CODE",
        "SYNTAX ERROR",
        "LOGIC ERROR",
        "DOES NOT SATISFY",
        "PLACEHOLDER",
        "UNIMPLEMENTED",
        "BUG",
        "BROKEN",
    )

    if any(marker in combined_text for marker in positive_markers):
        return "HALLUCINATION DETECTED"

    if any(marker in combined_text for marker in negative_markers):
        return "NO HALLUCINATION"

    if label.upper() in {"NO HALLUCINATION", "HALLUCINATION DETECTED"}:
        return label.upper()

    return "HALLUCINATION DETECTED"


def parse_qwen_response(raw_response: str) -> dict[str, Any]:
    """
    Parsea la respuesta del modelo Qwen hallucination detector.

    Busca las palabras clave: HALLUCINATION DETECTED, NO HALLUCINATION.
    """
    label, explanation = extract_response_fields(raw_response)
    score = normalize_qwen_score(label=label, explanation=explanation, raw_response=raw_response)

    if not explanation:
        explanation = "There was no explanation provided by the detector; the output was normalized using available textual cues."

    return {
        "SCORE": score,
        "EXPLANATION": explanation,
        "RAW_RESPONSE": raw_response,
    }


def evaluar_con_qwen(
    output_text: str,
    previous_output_text: str | None,
    project_context: str | None,
    node_id: str | None,
    ollama_url: str,
    model: str,
    timeout_seconds: int,
) -> tuple[dict[str, Any], str]:
    """
    Evalúa un output con el modelo Qwen hallucination detector.

    Args:
        output_text: El texto a evaluar
        ollama_url: URL del endpoint /api/generate de Ollama
        model: Nombre del modelo
        timeout_seconds: Timeout para la request
        project_context: Contexto del proyecto

    Returns:
        tuple: (respuesta_parseada, respuesta_raw)
    """
    prompt = build_hallucination_prompt(
        output=output_text,
        previous_output=previous_output_text,
        project_context=project_context,
        node_id=node_id,
    )

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "6h",
        "options": {
            "num_predict": 150,
            "stop": ["<|endoftext|>", "<|im_end|>"]
        }
    }

    response = requests.post(ollama_url, json=payload, timeout=timeout_seconds)
    response.raise_for_status()
    raw_response = response.json().get("response", "")
    parsed_response = parse_qwen_response(raw_response)
    return parsed_response, raw_response


def evaluate_run(
    run_dir: Path,
    execution_log_path: Path,
    ollama_url: str,
    model: str,
    timeout_seconds: int,
    retries: int,
    overwrite: bool,
) -> dict[str, Any]:
    """
    Procesa todos los outputs de un run y detecta hallucinations.

    Genera qwen_results.jsonl con un registro por cada output.
    """
    results_path = run_dir / "qwen_results.jsonl"
    run_name = run_dir.name
    run_summary: dict[str, Any] = {
        "run": run_name,
        "workflow_id": None,
        "status": "ok",
        "evaluations": 0,
        "counts": {
            "HALLUCINATION DETECTED": 0,
            "NO HALLUCINATION": 0,
            "ERROR": 0,
            "UNKNOWN": 0,
        },
        "errors": [],
        "result_file": str(results_path),
    }

    if results_path.exists() and not overwrite:
        run_summary["status"] = "skipped_existing"
        return run_summary

    workflow_id, node_outputs = extract_node_outputs(execution_log_path)
    run_summary["workflow_id"] = workflow_id

    if len(node_outputs) == 0:
        run_summary["status"] = "no_outputs"
        if overwrite:
            results_path.write_text("", encoding="utf-8")
        return run_summary

    with results_path.open("w", encoding="utf-8") as out_file:
        previous_output_text: str | None = None
        project_context: str | None = None
        evaluation_index = 0

        iterable = node_outputs
        start_idx = 1
        if node_outputs and node_outputs[0]["node_id"] == "USER":
            previous_output_text = node_outputs[0]["texto"]
            project_context = previous_output_text
            iterable = node_outputs[1:]
            start_idx = 2

        for idx, node_output in tqdm(enumerate(iterable, start=start_idx), total=len(node_outputs), desc=f"Evaluando outputs de {run_name}"):
            current_output_text = node_output["texto"]

            evaluation_index += 1
            eval_record: dict[str, Any] = {
                "run": run_name,
                "workflow_id": workflow_id,
                "index": evaluation_index,
                "source_index": idx,
                "node_id": node_output["node_id"],
                "timestamp": node_output["timestamp"],
                "output": current_output_text,
            }

            parsed_result: dict[str, Any] = {}
            raw_response = ""
            final_error = None

            for attempt in range(1, retries + 1):
                try:
                    parsed_result, raw_response = evaluar_con_qwen(
                        output_text=current_output_text,
                        previous_output_text=previous_output_text,
                        project_context=project_context,
                        node_id=node_output["node_id"],
                        ollama_url=ollama_url,
                        model=model,
                        timeout_seconds=timeout_seconds,
                    )
                    final_error = None
                    break
                except Exception as exc:
                    final_error = f"Intento {attempt}/{retries}: {exc}"

            if final_error is not None:
                parsed_result = {
                    "SCORE": "ERROR",
                    "EXPLANATION": f"Error de conexion/formato: {final_error}",
                }
                raw_response = ""
                run_summary["errors"].append(
                    {
                        "index": evaluation_index,
                        "source_index": idx,
                        "node_id": node_output["node_id"],
                        "error": final_error,
                    }
                )

            score = str(parsed_result.get("SCORE", "UNKNOWN")).upper()
            if score not in run_summary["counts"]:
                score = "UNKNOWN"

            run_summary["counts"][score] += 1
            run_summary["evaluations"] += 1

            eval_record["qwen"] = parsed_result
            eval_record["qwen_raw_response"] = raw_response
            eval_record["score"] = score

            out_file.write(json.dumps(eval_record, ensure_ascii=False) + "\n")

            previous_output_text = current_output_text

    return run_summary


def build_global_summary(warehouse_dir: Path, run_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Construye resumen global de todas las evaluaciones."""
    summary: dict[str, Any] = {
        "warehouse_dir": str(warehouse_dir),
        "runs_total": len(run_summaries),
        "runs_procesadas": 0,
        "runs_ok": 0,
        "runs_no_outputs": 0,
        "runs_skipped_existing": 0,
        "runs_error": 0,
        "scores": {
            "HALLUCINATION DETECTED": 0,
            "NO HALLUCINATION": 0,
            "ERROR": 0,
            "UNKNOWN": 0,
        },
        "errors_by_run": {},
        "runs": run_summaries,
    }

    for run in run_summaries:
        status = run.get("status")
        if status == "ok":
            summary["runs_ok"] += 1
            summary["runs_procesadas"] += 1
        elif status == "no_outputs":
            summary["runs_no_outputs"] += 1
            summary["runs_procesadas"] += 1
        elif status == "skipped_existing":
            summary["runs_skipped_existing"] += 1
        else:
            summary["runs_error"] += 1

        counts = run.get("counts", {})
        for score_key in summary["scores"]:
            summary["scores"][score_key] += int(counts.get(score_key, 0))

        run_errors = run.get("errors", [])
        if run_errors:
            summary["errors_by_run"][run.get("run", "desconocido")] = run_errors

    return summary


def parse_args() -> argparse.Namespace:
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Detecta hallucinations con Qwen en todas las ejecuciones de WareHouse."
    )
    parser.add_argument(
        "--warehouse-dir",
        default=str(DEFAULT_WAREHOUSE_DIR),
        help="Directorio raíz con runs de ChatDev (default: WareHouse).",
    )
    parser.add_argument(
        "--model",
        help="Modelo de Ollama para hallucination detection.",
        required=True,
    )
    parser.add_argument(
        "--ollama-url",
        default="http://127.0.0.1:11434/api/generate",
        help="URL del endpoint /api/generate de Ollama.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Timeout por request a Ollama en segundos.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Cantidad de intentos por evaluación (default: 2).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Sobrescribe qwen_results.jsonl existente en cada run.",
    )
    return parser.parse_args()


def main() -> None:
    """Función principal: orquesta la detección de hallucinations."""
    args = parse_args()
    warehouse_dir = Path(args.warehouse_dir)

    if not warehouse_dir.exists():
        raise FileNotFoundError(
            f"No existe el directorio de entrada: {warehouse_dir}"
        )

    run_dirs = discover_runs(warehouse_dir)
    print(
        f"Se encontraron {len(run_dirs)} runs con execution_logs.json en {warehouse_dir}\n"
    )

    print(f"Cargando modelo '{args.model}' en memoria (warm-up)...")
    try:
        requests.post(
            args.ollama_url,
            json={"model": args.model, "prompt": "", "stream": False},
            timeout=args.timeout,
        )
    except Exception:
        print("Aviso: El warm-up ha fallado o ha excedido el tiempo, intentando continuar...")

    run_summaries: list[dict[str, Any]] = []
    for _, run_dir in tqdm(
        enumerate(run_dirs, start=1),
        total=len(run_dirs),
        desc="Procesando ejecuciones",
    ):
        execution_log_path = run_dir / "execution_logs.json"
        try:
            run_summary = evaluate_run(
                run_dir=run_dir,
                execution_log_path=execution_log_path,
                ollama_url=args.ollama_url,
                model=args.model,
                timeout_seconds=args.timeout,
                retries=max(args.retries, 1),
                overwrite=args.overwrite,
            )
        except Exception as exc:
            run_summary = {
                "run": run_dir.name,
                "workflow_id": None,
                "status": "error",
                "evaluations": 0,
                "counts": {
                    "HALLUCINATION DETECTED": 0,
                    "NO HALLUCINATION": 0,
                    "ERROR": 0,
                    "UNKNOWN": 0,
                },
                "errors": [{"error": str(exc)}],
                "result_file": str(run_dir / "qwen_results.jsonl"),
            }

        run_summaries.append(run_summary)
        print(
            f"  estado={run_summary['status']} evals={run_summary['evaluations']} "
            f"scores={run_summary['counts']}"
        )

    summary = build_global_summary(warehouse_dir=warehouse_dir, run_summaries=run_summaries)
    summary_path = warehouse_dir / "qwen_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("Resumen global generado:")
    print(f"  {summary_path}")
    print(
        f"  runs_total={summary['runs_total']} runs_procesadas={summary['runs_procesadas']} "
        f"ok={summary['runs_ok']} no_outputs={summary['runs_no_outputs']} "
        f"skipped={summary['runs_skipped_existing']} error={summary['runs_error']}"
    )
    print(f"  scores={summary['scores']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
