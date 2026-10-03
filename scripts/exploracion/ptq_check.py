"""Exploración: verifica que la PTQ (TensorRT FP16/INT8) degrada a los alumnos base, en VALIDACIÓN.

Uso (desde la raíz del proyecto, en WSL):
    PYTHONPATH=. python scripts/exploracion/ptq_check.py
Config: configs/exploracion_ptq.yaml. No toca las corridas originales (copia best.pt antes de exportar).
Reanudable: cada resultado se agrega al CSV al obtenerse y las combinaciones ya presentes se saltan.

Variantes evaluadas por modelo:
  pytorch/fp32        best.pt con la validación por defecto (rect=True: letterbox rectangular por lote)
  pytorch/fp32_cuadrado  best.pt con rect=False (letterbox cuadrado 640×640, como el engine estático)
  tensorrt/{fp32,fp16,int8}
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from src.utils.config import cargar_yaml, resolver_ruta

cfg = cargar_yaml("configs/exploracion_ptq.yaml")
ex, ev = cfg["exportacion"], cfg["evaluacion"]
salida = resolver_ruta(cfg["salida"])
tabla = resolver_ruta(cfg["tabla"])
CAMPOS = ["modelo", "formato", "precision", "mAP50", "mAP50_95", "AP50_D40", "mAP50_sin_D10", "P", "R", "MB"]

hechos = set()
if tabla.exists():
    with open(tabla, encoding="utf-8") as f:
        hechos = {(r["modelo"], r["formato"], r["precision"]) for r in csv.DictReader(f)}
else:
    tabla.parent.mkdir(parents=True, exist_ok=True)
    with open(tabla, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=CAMPOS).writeheader()


def evaluar(ruta: Path, modelo: str, formato: str, precision: str, **extra) -> None:
    v = YOLO(str(ruta), task="detect").val(
        data=str(resolver_ruta(ev["data"])), split=ev["split"], imgsz=ev["imgsz"], batch=ev["batch"],
        conf=ev["conf"], iou=ev["iou"], plots=False, verbose=False,
        project=str(salida / "val"), name=f"{modelo}_{formato}_{precision}", exist_ok=True, **extra)
    ap50 = {v.names[i]: a for i, a in zip(v.box.ap_class_index, v.box.ap50)}
    fila = {"modelo": modelo, "formato": formato, "precision": precision,
            "mAP50": v.box.map50, "mAP50_95": v.box.map, "AP50_D40": ap50["D40"],
            "mAP50_sin_D10": float(np.mean([ap50[c] for c in ("D00", "D20", "D40")])),
            "P": v.box.mp, "R": v.box.mr, "MB": ruta.stat().st_size / 2**20}
    with open(tabla, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=CAMPOS).writerow(fila)
    print(f"RES {modelo} {formato} {precision} " + " ".join(f"{k}={x:.4f}" for k, x in fila.items()
                                                         if isinstance(x, float)), flush=True)


for nombre, pt in cfg["modelos"].items():
    d = salida / nombre
    d.mkdir(parents=True, exist_ok=True)
    pt = resolver_ruta(pt)
    if (nombre, "pytorch", "fp32") not in hechos:
        evaluar(pt, nombre, "pytorch", "fp32")
    if (nombre, "pytorch", "fp32_cuadrado") not in hechos:
        evaluar(pt, nombre, "pytorch", "fp32_cuadrado", rect=False)
    for prec in cfg["precisiones"]:
        if (nombre, "tensorrt", prec) in hechos:
            continue
        copia = d / f"{prec}.pt"  # el engine se escribe junto al .pt de origen: cada precisión en su copia
        engine = copia.with_suffix(".engine")
        if not engine.exists():
            shutil.copy2(pt, copia)
            extra = {"half": prec == "fp16", "int8": prec == "int8"}
            if prec == "int8":
                extra["data"] = str(resolver_ruta(ex["calib_data"]))  # calibración: 500 imágenes de TRAIN
            engine = Path(YOLO(str(copia)).export(format="engine", imgsz=ex["imgsz"], batch=ex["batch"],
                                                  dynamic=ex["dynamic"], simplify=ex["simplify"], verbose=False,
                                                  **extra))
        evaluar(engine, nombre, "tensorrt", prec)
print(f"TABLA {tabla}", flush=True)
