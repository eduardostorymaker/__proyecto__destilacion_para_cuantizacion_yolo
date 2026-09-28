"""Carga de configuraciones YAML y resolución de rutas del proyecto."""

from __future__ import annotations

from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parents[2]


def cargar_yaml(ruta: str | Path) -> dict:
    ruta = resolver_ruta(ruta)
    with open(ruta, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolver_ruta(ruta: str | Path) -> Path:
    """Expande '~' y convierte rutas relativas en rutas relativas a la raíz del proyecto."""
    p = Path(ruta).expanduser()
    return p if p.is_absolute() else RAIZ / p
