#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

convertir_tiempo() {
    local T=$1
    local H=$((T/3600))
    local M=$((T%3600/60))
    local S=$((T%60))
    printf "%02dh %02dm %02ds" $H $M $S
}

SECONDS=0
echo "=============================================="
echo "=============================================="
echo "Ejecutando workflow completo para crear trazas de detección con Ollama..."
echo "=============================================="
echo "=============================================="

echo "----------------------------------------------"
echo "PARTE 1: Creando trazas de ChatDev..."
echo "----------------------------------------------"

inicio_p1=$SECONDS
"$ROOT_DIR/scripts/create_chatdev_traces_with_ollama.sh" qwen2.5-coder:3b
fin_p1=$SECONDS
tiempo_p1=$((fin_p1 - inicio_p1))

echo "----------------------------------------------"
echo -e "Tiempo que ha tardado la primera parte (ChatDev):\t$(convertir_tiempo $tiempo_p1) ($tiempo_p1 segundos)"
echo "----------------------------------------------"

echo "=============================================="
echo "=============================================="
echo "=============================================="

echo "----------------------------------------------"
echo "PARTE 2: Ejecutando script de detección de alucinaciones..."
echo "----------------------------------------------"

inicio_p2=$SECONDS
"$ROOT_DIR/scripts/detect_qwen_with_ollama.sh"
fin_p2=$SECONDS

tiempo_p2=$((fin_p2 - inicio_p2))
tiempo_total=$SECONDS
echo "----------------------------------------------"
echo -e "Tiempo que ha tardado la segunda parte (Detección):\t$(convertir_tiempo $tiempo_p2) ($tiempo_p2 segundos)"
echo "----------------------------------------------"

echo "=============================================="
echo "=============================================="
echo "=============================================="
echo "----------------------------------------------"
echo "Workflow completo ejecutado con éxito."
{
    echo "----------------------------------------------"
    echo "RESUMEN DE TIEMPOS"
    echo -e "Parte 1 (ChatDev):\t$(convertir_tiempo $tiempo_p1) ($tiempo_p1 segundos)"
    echo -e "Parte 2 (Detección):\t$(convertir_tiempo $tiempo_p2) ($tiempo_p2 segundos)"
    echo -e "Tiempo total:\t\t$(convertir_tiempo $tiempo_total) ($tiempo_total segundos)"
    echo "----------------------------------------------"
} | tee "$ROOT_DIR/WareHouse/time.txt"
echo "=============================================="
echo "=============================================="
echo "=============================================="
sudo chown -R $USER:$USER "$ROOT_DIR/WareHouse/"
