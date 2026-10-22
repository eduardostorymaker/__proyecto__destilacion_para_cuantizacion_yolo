#!/usr/bin/env bash
# Fase 8: eficiencia (latencia, FPS, tamaño) de los engines registrados. Ejecutar con el equipo SIN otra carga.
# Uso (en WSL):  bash scripts/benchmark.sh            # India
#                bash scripts/benchmark.sh --dataset japan
set -uo pipefail

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJ_DIR}"
source "${VENV_DIR:-$HOME/.venvs/yolokd}/bin/activate"

mkdir -p runs/logs
LOG="runs/logs/benchmark_$(date +%Y%m%d_%H%M%S).log"
echo ">> Log en ${LOG}"
python -m src.bench.benchmark "$@" 2>&1 | tee "${LOG}"
