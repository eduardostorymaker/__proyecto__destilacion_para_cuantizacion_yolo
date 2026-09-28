#!/usr/bin/env bash
# Crea el entorno del proyecto en WSL2 (Ubuntu).
# - Instala uv (gestor de Python) si no existe.
# - Crea un venv con Python 3.13 en el sistema de archivos de Linux (~/.venvs/yolokd),
#   porque importar paquetes desde /mnt/d es muy lento.
# - Instala requirements.txt y congela el lock completo en requirements.lock.txt.
#
# Uso (desde la raíz del proyecto, dentro de WSL):
#   bash scripts/setup_wsl.sh
set -euo pipefail

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/yolokd}"
PY_VERSION="3.13"

# Paquetes del sistema que necesitan OpenCV y la compilación de algunas ruedas
if ! dpkg -s libgl1 >/dev/null 2>&1 || ! dpkg -s libglib2.0-0t64 >/dev/null 2>&1; then
    echo ">> Instalando dependencias del sistema (requiere sudo)"
    sudo apt-get update
    sudo apt-get install -y libgl1 libglib2.0-0t64 curl
fi

if ! command -v uv >/dev/null 2>&1; then
    echo ">> Instalando uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo ">> Creando venv Python ${PY_VERSION} en ${VENV_DIR}"
uv python install "${PY_VERSION}"
uv venv --python "${PY_VERSION}" "${VENV_DIR}"

echo ">> Instalando dependencias"
# index-strategy: permite tomar torch del índice de PyTorch y el resto de PyPI
VIRTUAL_ENV="${VENV_DIR}" uv pip install \
    --index-strategy unsafe-best-match \
    -r "${PROJ_DIR}/requirements.txt"

echo ">> Congelando versiones exactas en requirements.lock.txt"
VIRTUAL_ENV="${VENV_DIR}" uv pip freeze > "${PROJ_DIR}/requirements.lock.txt"

echo
echo "Listo. Activa el entorno con:"
echo "  source ${VENV_DIR}/bin/activate"
echo "y verifica con:"
echo "  python scripts/check_env.py"
