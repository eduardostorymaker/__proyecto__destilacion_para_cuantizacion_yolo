"""Selección del dataset (India o la réplica en Japón, D9): cada uno tiene su carpeta de configuración.

India usa `configs/*.yaml`; Japón, `configs/japan/*.yaml` con los mismos nombres de archivo. Todos los scripts
de las fases 2–8 aceptan `--dataset {india,japan}` (por defecto india) y leen sus yaml con `ruta_config`.
"""

from __future__ import annotations

import argparse

CARPETAS = {"india": "configs", "japan": "configs/japan"}


def ruta_config(nombre: str, dataset: str = "india") -> str:
    """Ruta relativa del yaml `nombre` para el dataset, p. ej. ('kd.yaml', 'japan') → 'configs/japan/kd.yaml'."""
    return f"{CARPETAS[dataset]}/{nombre}"


def agregar_argumento(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--dataset", choices=list(CARPETAS), default="india",
                    help="india (configs/) o japan (configs/japan/, réplica D9)")
