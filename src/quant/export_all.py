"""Fase 5: exporta cada modelo a TensorRT (FP32/FP16/INT8) y OpenVINO (FP32/INT8) según D4.

- Cada precisión se exporta por separado desde best.pt (FP32), con el exportador de Ultralytics 8.4.164.
- INT8 se calibra con configs/calib.yaml (500 imágenes de TRAIN). Antes de exportar se verifica que esa lista
  esté contenida en train y no toque val ni test; durante cada INT8 se registra el mensaje de Ultralytics que
  indica de dónde tomó las imágenes de calibración.
- Para cada engine INT8 se cuenta cuántas capas quedaron en INT8, FP16 y FP32 (results/tables/engine_layers.csv).
- Reanudable: cada exportación se registra en results/tables/exports.csv al terminar; las ya hechas se saltan.

Uso (desde la raíz del proyecto, en WSL):
    python -m src.quant.export_all                          # todos los modelos de configs/quant.yaml
    python -m src.quant.export_all --modelos student_base_s0
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import time
from pathlib import Path

from ultralytics import YOLO

from src.quant.engine_layers import contar_capas
from src.utils.config import cargar_yaml, resolver_ruta
from src.utils.repro import info_entorno

CAMPOS_EXP = ["modelo", "formato", "precision", "ruta", "MB", "segundos", "calibracion"]


class CapturaLog(logging.Handler):
    """Guarda los mensajes del logger de Ultralytics durante una exportación."""

    def __init__(self):
        super().__init__()
        self.mensajes: list[str] = []

    def emit(self, record):
        self.mensajes.append(record.getMessage())


def verificar_calibracion(calib_yaml: Path) -> int:
    """Comprueba que la lista de calibración esté en train y fuera de val/test; devuelve su tamaño."""
    c = cargar_yaml(calib_yaml)
    raiz = Path(c["path"])
    listas = {k: raiz / c[k] for k in ("train", "val")}
    nombres = {k: {Path(l.strip()).name for l in open(v, encoding="utf-8") if l.strip()} for k, v in listas.items()}
    if nombres["train"] != nombres["val"]:
        raise RuntimeError("calib.yaml: 'train' y 'val' deben apuntar a la misma lista de calibración")
    calib = nombres["val"]  # el exportador de Ultralytics calibra con la clave 'val'
    splits = {s: {l.strip() for l in open(resolver_ruta(f"data/splits/{s}.txt"), encoding="utf-8") if l.strip()}
              for s in ("train", "val", "test")}
    if not calib <= splits["train"] or calib & (splits["val"] | splits["test"]):
        raise RuntimeError("La lista de calibración no está contenida en train o toca val/test")
    return len(calib)


def tamano_mb(ruta: Path) -> float:
    if ruta.is_dir():
        return sum(f.stat().st_size for f in ruta.rglob("*") if f.is_file()) / 2**20
    return ruta.stat().st_size / 2**20


def leer_hechos(tabla: Path) -> set:
    if not tabla.exists():
        return set()
    with open(tabla, encoding="utf-8") as f:
        return {(r["modelo"], r["formato"], r["precision"]) for r in csv.DictReader(f)}


def agregar_fila(tabla: Path, fila: dict) -> None:
    nuevo = not tabla.exists()
    tabla.parent.mkdir(parents=True, exist_ok=True)
    with open(tabla, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fila))
        if nuevo:
            w.writeheader()
        w.writerow(fila)


def exportar(pt: Path, destino: Path, formato: str, prec: str, ex: dict, calib: Path) -> tuple[Path, float, str]:
    destino.mkdir(parents=True, exist_ok=True)
    copia = destino / "best.pt"  # Ultralytics escribe la exportación junto al .pt de origen
    shutil.copy2(pt, copia)
    kwargs = {"format": "engine" if formato == "tensorrt" else "openvino", "imgsz": ex["imgsz"],
              "batch": ex["batch"], "dynamic": ex["dynamic"], "simplify": ex["simplify"],
              "half": prec == "fp16", "int8": prec == "int8", "verbose": False}
    if prec == "int8":
        kwargs["data"] = str(calib)
    captura = CapturaLog()
    logging.getLogger("ultralytics").addHandler(captura)
    t0 = time.perf_counter()
    try:
        salida = Path(YOLO(str(copia)).export(**kwargs))
    finally:
        logging.getLogger("ultralytics").removeHandler(captura)
    segundos = time.perf_counter() - t0
    copia.unlink()  # el origen queda en runs/students o runs/teacher
    nota = ""
    if prec == "int8":
        msjs = [m for m in captura.mensajes if "calibration" in m.lower()]
        nota = " | ".join(m.strip() for m in msjs)[:400]
        if str(calib) not in nota:
            raise RuntimeError(f"No se pudo confirmar que la calibración usó {calib}: {nota or 'sin mensajes'}")
    (destino / "entorno.json").write_text(json.dumps(info_entorno() | {"origen": str(pt), "formato": formato,
                                                     "precision": prec, "export_kwargs": kwargs}, indent=2,
                                                    ensure_ascii=False), encoding="utf-8")
    return salida, segundos, nota


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/quant.yaml")
    ap.add_argument("--modelos", nargs="*", help="subconjunto de claves de 'modelos' (por defecto, todos)")
    a = ap.parse_args()

    cfg = cargar_yaml(a.config)
    ex = cfg["exportacion"]
    calib = resolver_ruta(ex["calib_data"])
    n_calib = verificar_calibracion(calib)
    print(f"Calibración verificada: {n_calib} imágenes de train, ninguna de val/test ({calib})", flush=True)

    salida = resolver_ruta(cfg["salida"])
    t_exp, t_capas = (resolver_ruta(cfg["tablas"][k]) for k in ("exportaciones", "capas_engine"))
    hechos = leer_hechos(t_exp)
    modelos = {k: v for k, v in cfg["modelos"].items() if not a.modelos or k in a.modelos}

    for nombre, pt in modelos.items():
        pt = resolver_ruta(pt)
        if not pt.exists():
            print(f"!! {nombre}: no existe {pt}, se omite", flush=True)
            continue
        precisiones = cfg["precisiones"]["maestro" if nombre.startswith("teacher") else "alumnos"]
        for formato, lista in precisiones.items():
            for prec in lista:
                if (nombre, formato, prec) in hechos:
                    continue
                destino = salida / f"{nombre}_{formato}_{prec}"
                print(f">> {nombre} {formato} {prec}", flush=True)
                ruta, seg, nota = exportar(pt, destino, formato, prec, ex, calib)
                agregar_fila(t_exp, {"modelo": nombre, "formato": formato, "precision": prec,
                                     "ruta": str(ruta.relative_to(resolver_ruta("."))), "MB": round(tamano_mb(ruta), 2),
                                     "segundos": round(seg, 1), "calibracion": nota})
                if formato == "tensorrt" and prec == "int8":
                    agregar_fila(t_capas, {"modelo": nombre, **contar_capas(ruta)})
                print(f"   OK {ruta.name}  {tamano_mb(ruta):.1f} MB  {seg:.0f} s", flush=True)
    print("FIN", flush=True)


if __name__ == "__main__":
    main()
