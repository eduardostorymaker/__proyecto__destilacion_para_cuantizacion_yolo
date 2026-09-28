#!/usr/bin/env bash
# Fase 2: entrenamiento del maestro YOLO11m.
# Uso (en WSL, desde cualquier carpeta):
#   bash scripts/train_teacher.sh           # entrenamiento completo (~4-6 h)
#   bash scripts/train_teacher.sh --humo    # prueba rápida
# Recomendado lanzarlo dentro de tmux para que sobreviva si se cierra la terminal.
set -euo pipefail

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJ_DIR}"
source "${VENV_DIR:-$HOME/.venvs/yolokd}/bin/activate"

mkdir -p runs/logs
LOG="runs/logs/teacher_$(date +%Y%m%d_%H%M%S)${1:+_humo}.log"
echo ">> Log en ${LOG}"
python -m src.train.train_teacher "$@" 2>&1 | tee "${LOG}"
