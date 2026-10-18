#!/usr/bin/env bash
# Fase 6: evaluación en TEST de todos los modelos (reanudable).
# Uso (en WSL; recomendado dentro de tmux):  bash scripts/evaluate.sh
set -uo pipefail

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJ_DIR}"
source "${VENV_DIR:-$HOME/.venvs/yolokd}/bin/activate"

mkdir -p runs/logs
LOG="runs/logs/evaluate_$(date +%Y%m%d_%H%M%S).log"
echo ">> Log en ${LOG}"
python -m src.eval.evaluate "$@" 2>&1 | tee "${LOG}"
