#!/usr/bin/env bash
# Fase 4b (parte 1): piloto de λ. Entrena un alumno KD por cada λ de configs/kd.yaml (seed 0, 50 épocas)
# y al final analiza las corridas y propone λ (no modifica kd.yaml).
# Uso (en WSL; recomendado dentro de tmux):  bash scripts/pilot_lambda.sh
set -uo pipefail

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJ_DIR}"
source "${VENV_DIR:-$HOME/.venvs/yolokd}/bin/activate"

LAMBDAS=$(python -c "from src.utils.config import cargar_yaml; print(' '.join(str(float(l)) for l in cargar_yaml('configs/kd.yaml')['piloto']['lambdas']))")
echo ">> Piloto de λ: ${LAMBDAS}"
for l in ${LAMBDAS}; do
    bash scripts/train_student.sh --kd on --piloto "${l}" || echo "!! Falló λ=${l}; se sigue con el siguiente"
done

mkdir -p runs/logs
python -m src.train.select_lambda 2>&1 | tee "runs/logs/pilot_lambda_seleccion_$(date +%Y%m%d_%H%M%S).log"
