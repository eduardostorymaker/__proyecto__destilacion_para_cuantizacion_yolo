#!/usr/bin/env bash
# Fase 5b: QAT INT8 con ModelOpt (control "ModelOpt PTQ" + ajuste fino + exportación). Reanudable.
# Uso (en WSL; recomendado dentro de tmux):
#   bash scripts/qat.sh                                # los 6 alumnos
#   bash scripts/qat.sh --alumnos student_kd_s0        # uno o varios
set -uo pipefail

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJ_DIR}"
source "${VENV_DIR:-$HOME/.venvs/yolokd}/bin/activate"

mkdir -p runs/logs
LOG="runs/logs/qat_$(date +%Y%m%d_%H%M%S).log"
echo ">> Log en ${LOG}"
python -m src.quant.qat "$@" 2>&1 | tee "${LOG}"
