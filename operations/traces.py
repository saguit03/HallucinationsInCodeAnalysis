import json
from pathlib import Path
import pandas as pd
from .globals import WAREHOUSE_PATH, EXECUTION_FILE, QWEN_FILE

def read_structured_file(file_path: str | Path):
    """Load .json as a single JSON object and .jsonl as a list of JSON objects."""
    path = Path(file_path)
    with path.open("r", encoding="utf-8") as file_handle:
        if path.suffix.lower() == ".jsonl":
            return [json.loads(line) for line in file_handle if line.strip()]
        return json.load(file_handle)
    
def load_warehouse_run_jsons(warehouse_dir: str | Path = WAREHOUSE_PATH) -> pd.DataFrame:
    warehouse_path = Path(warehouse_dir).resolve()
    if not warehouse_path.exists():
        raise FileNotFoundError(f"No existe el directorio: {warehouse_path}")

    rows = []
    for run_dir in sorted(p for p in warehouse_path.iterdir() if p.is_dir()):
        execution_path = run_dir / EXECUTION_FILE
        qwen_path = run_dir / QWEN_FILE

        row = {
            "run_name": run_dir.name,
            "run_path": str(run_dir),
            "execution_logs": None,
            "qwen_results": None
        }

        if execution_path.exists():
            try:
                row["execution_logs"] = read_structured_file(execution_path)
            except Exception as ex:
                row["execution_logs_error"] = str(ex)

        if qwen_path.exists():
            try:
                row["qwen_results"] = read_structured_file(qwen_path)
            except Exception as ex:
                row["qwen_results_error"] = str(ex)
        rows.append(row)

    return pd.DataFrame(rows)

def get_logs_from_warehouse(warehouse_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for idx, run_row in warehouse_df.iterrows():
        exec_log = run_row.get('execution_logs')
        if not exec_log:
            continue

        if isinstance(exec_log, dict):
            logs = exec_log.get('logs')
        elif isinstance(exec_log, list):
            logs = exec_log
        else:
            logs = None

        if not logs:
            continue

        for log in logs:
            if not isinstance(log, dict):
                continue

            rows.append({
                'run_name': run_row.get('run_name'),
                'timestamp': log.get('timestamp'),
                'level': log.get('level'),
                'node_id': log.get('node_id'),
                'event_type': log.get('event_type'),
                'message': log.get('message'),
                'details': log.get('details'),
                'duration': log.get('duration')
            })
    logs_df = pd.DataFrame(rows)
    if 'timestamp' in logs_df.columns:
        logs_df['timestamp'] = pd.to_datetime(logs_df['timestamp'], errors='coerce')
    return logs_df