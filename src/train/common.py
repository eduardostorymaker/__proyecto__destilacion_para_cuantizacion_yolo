"""Utilidades comunes de entrenamiento para maestro y alumnos."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from src.utils.config import resolver_ruta
from src.utils.repro import info_entorno


def argumentos_entrenamiento(cfg: dict, rol: str, seed: int, name: str, humo: bool) -> dict:
    """Arma los kwargs de `model.train()` a partir de configs/train.yaml.

    `rol` es 'maestro' o 'alumno'; su bloque `sobrescribe` (si existe) reemplaza valores de `comun`.
    """
    args = dict(cfg["comun"]) | cfg[rol].get("sobrescribe", {})
    args["data"] = str(resolver_ruta(args["data"]))
    project = cfg[rol]["project"]
    args.update(seed=seed, project=str(resolver_ruta(project)), name=name, exist_ok=False)
    if humo:
        h = cfg["humo"]
        args.update(epochs=h["epochs"], fraction=h["fraction"], project=str(resolver_ruta(h["project"])),
                    name=f"{name}_humo", exist_ok=True)
    return args


def ruta_pesos_preentrenados(ruta: str) -> str:
    """Crea la carpeta de pesos preentrenados; ultralytics descarga ahí si el archivo no existe."""
    p = resolver_ruta(ruta)
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


def callback_metadatos(configs: list[str], extra: dict):
    """Callback de inicio: guarda en la carpeta de la corrida las configs usadas y el entorno."""

    def _guardar(trainer) -> None:
        save_dir = Path(trainer.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for c in configs:
            shutil.copy2(resolver_ruta(c), save_dir / f"usada_{Path(c).name}")
        info = info_entorno() | extra
        (save_dir / "entorno.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

    return _guardar
