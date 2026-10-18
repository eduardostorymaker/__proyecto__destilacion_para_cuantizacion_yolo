"""Registro de engines TensorRT/OpenVINO con hash SHA-256 (results/tables/engines.csv).

Cada engine se construye UNA sola vez y se registra con su hash; la evaluación (Fase 6) y la eficiencia
(Fase 8) usan siempre esos mismos archivos y verifican el hash antes de usarlos. Para OpenVINO (carpeta con
.xml/.bin) el hash se calcula sobre todos los archivos de la carpeta, en orden, incluyendo sus nombres.
También se registra el entorno de ejecución: driver NVIDIA, CUDA, TensorRT y OpenVINO.

Uso:
    python -m src.quant.engines_registry --backfill    # registra las exportaciones ya existentes en exports.csv
    python -m src.quant.engines_registry --verificar   # comprueba que todos los archivos registrados no cambiaron
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata as md
import re
import subprocess
from datetime import datetime
from pathlib import Path

from src.utils.config import resolver_ruta

TABLA = "results/tables/engines.csv"
CAMPOS = ["modelo", "formato", "precision", "ruta", "sha256", "MB", "fecha_registro",
          "driver_nvidia", "cuda_driver", "cuda_torch", "tensorrt", "openvino"]


def sha256_ruta(ruta: Path) -> str:
    h = hashlib.sha256()
    archivos = sorted(f for f in ruta.rglob("*") if f.is_file()) if ruta.is_dir() else [ruta]
    for f in archivos:
        if ruta.is_dir():
            h.update(str(f.relative_to(ruta)).encode())
        with open(f, "rb") as fh:
            for bloque in iter(lambda: fh.read(1 << 20), b""):
                h.update(bloque)
    return h.hexdigest()


def info_sistema() -> dict:
    driver = cuda = ""
    try:
        driver = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                         text=True).strip().splitlines()[0]
        m = re.search(r"CUDA (?:UMD )?Version:\s*([\d.]+)", subprocess.check_output(["nvidia-smi"], text=True))
        cuda = m.group(1) if m else ""
    except (OSError, subprocess.CalledProcessError):
        pass
    try:
        import torch

        cuda_torch = torch.version.cuda or ""
    except ImportError:
        cuda_torch = ""
    ver = lambda p: next((md.version(x) for x in p if _instalado(x)), "")  # noqa: E731
    return {"driver_nvidia": driver, "cuda_driver": cuda, "cuda_torch": cuda_torch,
            "tensorrt": ver(["tensorrt-cu13", "tensorrt"]), "openvino": ver(["openvino"])}


def _instalado(paquete: str) -> bool:
    try:
        md.version(paquete)
        return True
    except md.PackageNotFoundError:
        return False


def leer() -> dict[str, dict]:
    t = resolver_ruta(TABLA)
    if not t.exists():
        return {}
    with open(t, encoding="utf-8") as f:
        return {r["ruta"]: r for r in csv.DictReader(f)}


def _relativa(ruta: Path) -> str:
    return str(Path(ruta).resolve().relative_to(resolver_ruta(".").resolve()))


def registrar(modelo: str, formato: str, precision: str, ruta: Path) -> str:
    """Registra un engine recién construido. Si ya estaba registrado con otro hash, falla (se construye una vez)."""
    ruta = Path(ruta)
    rel, h = _relativa(ruta), sha256_ruta(ruta)
    previos = leer()
    if rel in previos:
        if previos[rel]["sha256"] != h:
            raise RuntimeError(f"{rel} ya estaba registrado con otro hash: los engines se construyen una sola vez")
        return h
    t = resolver_ruta(TABLA)
    nuevo = not t.exists()
    t.parent.mkdir(parents=True, exist_ok=True)
    mb = sum(f.stat().st_size for f in ruta.rglob("*") if f.is_file()) if ruta.is_dir() else ruta.stat().st_size
    with open(t, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS)
        if nuevo:
            w.writeheader()
        w.writerow({"modelo": modelo, "formato": formato, "precision": precision, "ruta": rel, "sha256": h,
                    "MB": round(mb / 2**20, 2), "fecha_registro": datetime.now().isoformat(timespec="seconds"),
                    **info_sistema()})
    return h


def verificar(ruta: Path) -> None:
    """Falla si el archivo no está registrado o si su hash cambió."""
    rel = _relativa(Path(ruta))
    reg = leer().get(rel)
    if reg is None:
        raise RuntimeError(f"{rel} no está registrado en {TABLA}")
    if sha256_ruta(Path(ruta)) != reg["sha256"]:
        raise RuntimeError(f"{rel}: el hash no coincide con el registrado (el archivo cambió)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--backfill", action="store_true")
    g.add_argument("--verificar", action="store_true")
    a = ap.parse_args()
    if a.backfill:
        with open(resolver_ruta("results/tables/exports.csv"), encoding="utf-8") as f:
            for r in csv.DictReader(f):
                h = registrar(r["modelo"], r["formato"], r["precision"], resolver_ruta(r["ruta"]))
                print(f"{r['modelo']:<16} {r['formato']:<9} {r['precision']:<10} {h[:16]}…")
    else:
        regs = leer()
        for rel in regs:
            verificar(resolver_ruta(rel))
        print(f"{len(regs)} engines verificados: hashes sin cambios")
    print(info_sistema())


if __name__ == "__main__":
    main()
