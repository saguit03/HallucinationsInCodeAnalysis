#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL="${1:-saguit03uoc/qwen2.5-coder-hallu-detector:latest}"
VENV_ACTIVATE="${2:-$ROOT_DIR/.venv/bin/activate}"
MAIN_SCRIPT="${3:-$ROOT_DIR/scripts/detect_qwen.py}"

stop_ollama_session() {
    local pid=$1
    local model_name=$2

    if ! kill -0 "$pid" 2>/dev/null; then
        return
    fi

    echo "Deteniendo modelo $model_name..."
    ollama stop "$model_name" > /dev/null 2>&1

    for i in {1..15}; do
        if ! kill -0 "$pid" 2>/dev/null; then
            return
        fi
        sleep 1
    done

    kill -9 "$pid" 2>/dev/null
}

OLLAMA_PID=""

cleanup() {
    if [ -n "$OLLAMA_PID" ]; then
        stop_ollama_session "$OLLAMA_PID" "$MODEL"
    fi
}

trap cleanup EXIT

echo "[1/4] Iniciando Ollama en segundo plano con '$MODEL'..."
ollama run "$MODEL" > /dev/null 2>&1 &
OLLAMA_PID=$!

sleep 20

echo "[2/4] Activando entorno virtual..."
if [ -f "$VENV_ACTIVATE" ]; then
    source "$VENV_ACTIVATE"
else
    echo "Error: No se encontró el script de activación en $VENV_ACTIVATE"
    exit 1
fi

export OPENAI_API_KEY="ollama"
export OPENAI_BASE_URL="http://localhost:11434/v1"
export CHATDEV_MODEL="$MODEL"
export CHAT_MODEL="$MODEL"

echo "[3/4] Ejecutando detección de alucinaciones con Qwen..."
python3 "$MAIN_SCRIPT" --model "$MODEL" --ollama-url "http://localhost:11434/api/generate"

echo "[4/4] Cerrando sesión de Ollama..."
cleanup
OLLAMA_PID=""

echo "Proceso completado."
