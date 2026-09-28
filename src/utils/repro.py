"""Utilidades de reproducibilidad: fijar semillas y registrar versiones de librerías."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Paquetes cuya versión se registra en cada corrida
PAQUETES_REGISTRADOS = [
    "ultralytics",
    "torch",
    "torchvision",
    "tensorrt",
    "tensorrt-cu13",
    "openvino",
    "nncf",
    "onnx",
    "onnxslim",
    "numpy",
    "scipy",
    "pandas",
    "pycocotools",
]


def fijar_semillas(seed: int, deterministico: bool = True) -> None:
    """Fija las semillas de Python, NumPy y PyTorch."""
    import numpy as np

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministico:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def version_paquete(nombre: str) -> str | None:
    try:
        return importlib.metadata.version(nombre)
    except importlib.metadata.PackageNotFoundError:
        return None


def commit_git() -> str | None:
    """Hash del commit actual (con sufijo '-dirty' si hay cambios sin commit)."""
    raiz = Path(__file__).resolve().parents[2]
    try:
        h = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=raiz, text=True).strip()
        sucio = subprocess.check_output(["git", "status", "--porcelain"], cwd=raiz, text=True).strip()
        return h + ("-dirty" if sucio else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def info_entorno() -> dict:
    """Versiones de librerías, hardware y commit para guardar junto a cada corrida."""
    info = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "plataforma": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "git_commit": commit_git(),
        "paquetes": {p: version_paquete(p) for p in PAQUETES_REGISTRADOS},
    }
    try:
        import torch

        info["cuda_torch"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version()
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        pass
    return info


def guardar_info_entorno(destino: str | Path) -> dict:
    """Escribe info_entorno() como JSON en `destino` y la devuelve."""
    info = info_entorno()
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return info
