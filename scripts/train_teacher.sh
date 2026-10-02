#!/usr/bin/env bash
# Fase 2: entrenamiento de un maestro. Los argumentos pasan a train_teacher.py.
# Uso (en WSL, desde cualquier carpeta; recomendado dentro de tmux):
#   bash scripts/train_teacher.sh                       # YOLO11m (teacher_s0)
#   bash scripts/train_teacher.sh --rol maestro_l       # YOLO11l (teacher_l_s0)
#   bash scripts/train_teacher.sh --rol maestro_l --humo
set -euo pipefail

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJ_DIR}"
source "${VENV_DIR:-$HOME/.venvs/yolokd}/bin/activate"

mkdir -p runs/logs
ETIQUETA="$(echo "$*" | tr -d '-' | tr ' ' '_')"
LOG="runs/logs/teacher_$(date +%Y%m%d_%H%M%S)${ETIQUETA:+_${ETIQUETA}}.log"
echo ">> Log en ${LOG}"
python -m src.train.train_teacher "$@" 2>&1 | tee "${LOG}"
