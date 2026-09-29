#!/usr/bin/env bash
# Fase 1b: pruebas de tiempo (3 casos de 3 épocas + forward del maestro + resumen).
# Uso (en WSL):  bash scripts/fase1b_tiempos.sh
set -euo pipefail

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJ_DIR}"
source "${VENV_DIR:-$HOME/.venvs/yolokd}/bin/activate"

mkdir -p runs/logs
LOG="runs/logs/fase1b_$(date +%Y%m%d_%H%M%S).log"
echo ">> Log en ${LOG}"
{
    for caso in maestro_amp maestro_fp32 alumno_amp; do
        echo ">> Caso ${caso}"
        python -m src.train.timing_test --caso "${caso}"
    done
    python -m src.train.timing_test --forward-maestro
    python -m src.train.timing_test --resumen
} 2>&1 | tee "${LOG}"
