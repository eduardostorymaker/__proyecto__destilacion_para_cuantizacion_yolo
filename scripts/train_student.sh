#!/usr/bin/env bash
# Fases 3 y 4: entrenamiento de alumnos YOLO11n (base o KD). Los argumentos pasan a train_student.py.
# Uso (en WSL, desde cualquier carpeta; recomendado dentro de tmux):
#   bash scripts/train_student.sh --kd off --seed 0
#   bash scripts/train_student.sh --kd on  --piloto 1.0
#   bash scripts/train_student.sh --kd on  --seed 0 --humo
set -euo pipefail

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJ_DIR}"
source "${VENV_DIR:-$HOME/.venvs/yolokd}/bin/activate"

mkdir -p runs/logs
ETIQUETA="$(echo "$*" | tr -d '-' | tr ' ' '_')"
LOG="runs/logs/student_$(date +%Y%m%d_%H%M%S)_${ETIQUETA}.log"
echo ">> Log en ${LOG}"
python -m src.train.train_student "$@" 2>&1 | tee "${LOG}"
