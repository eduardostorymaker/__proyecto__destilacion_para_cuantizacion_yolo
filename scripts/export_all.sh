#!/usr/bin/env bash
# Fase 5: exportación y cuantización PTQ de todos los modelos (TensorRT y OpenVINO).
# Reanudable: si se corta, relanzarlo continúa con lo que falte.
# Uso (en WSL; recomendado dentro de tmux):
#   bash scripts/export_all.sh                              # todos
#   bash scripts/export_all.sh --modelos student_kd_s0      # uno o varios
set -uo pipefail

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJ_DIR}"
source "${VENV_DIR:-$HOME/.venvs/yolokd}/bin/activate"

mkdir -p runs/logs
LOG="runs/logs/export_all_$(date +%Y%m%d_%H%M%S).log"
echo ">> Log en ${LOG}"
python -m src.quant.export_all "$@" 2>&1 | tee "${LOG}"
