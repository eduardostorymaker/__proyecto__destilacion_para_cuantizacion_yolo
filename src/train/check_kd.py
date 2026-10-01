"""Verificación de la prueba de humo de KD (Fase 4a).

Comprueba:
  1. results.csv del alumno KD registra train/cwd (L_CWD sin λ) por época y que L_CWD baja.
  2. best.pt del alumno KD no contiene adaptadores ni maestro: mismas claves y formas de
     state_dict que best.pt del alumno base.
  3. El ONNX exportado del alumno KD tiene el mismo número de parámetros que el del alumno base.

Uso:
    python -m src.train.check_kd --kd runs/smoke/student_kd_s0_humo --base runs/smoke/student_base_s0_humo
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import numpy_helper
from ultralytics import YOLO

from src.utils.config import resolver_ruta


def leer_resultados(run: Path) -> list[dict]:
    with open(run / "results.csv", encoding="utf-8") as f:
        return [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(f)]


def params_onnx(ruta: Path) -> int:
    return int(sum(np.prod(numpy_helper.to_array(t).shape) for t in onnx.load(str(ruta)).graph.initializer))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kd", required=True)
    ap.add_argument("--base", required=True)
    a = ap.parse_args()
    run_kd, run_base = resolver_ruta(a.kd), resolver_ruta(a.base)
    ok = True

    # 1. L_CWD por época
    filas = leer_resultados(run_kd)
    if "train/cwd" not in filas[0]:
        print("FALLA: results.csv no tiene train/cwd")
        ok = False
    else:
        print("época  L_det(box+cls+dfl)  L_CWD")
        cwd = []
        for r in filas:
            det = sum(float(r[f"train/{k}_loss"]) for k in ("box", "cls", "dfl"))
            cwd.append(float(r["train/cwd"]))
            print(f"{int(float(r['epoch'])):>5}  {det:>18.4f}  {cwd[-1]:.4f}")
        finitos = all(math.isfinite(c) for c in cwd)
        baja = len(cwd) > 1 and cwd[-1] < cwd[0]
        print(f"[{'OK' if finitos and baja else 'FALLA'}] L_CWD finita y decreciente ({cwd[0]:.4f} → {cwd[-1]:.4f})")
        ok &= finitos and baja

    # 2. Checkpoints: mismas claves y formas
    sd = {}
    for nombre, run in (("kd", run_kd), ("base", run_base)):
        ck = torch.load(run / "weights" / "best.pt", map_location="cpu", weights_only=False)
        modelo = ck.get("ema") or ck["model"]  # strip_optimizer deja el modelo final en "model"
        sd[nombre] = {k: tuple(v.shape) for k, v in modelo.state_dict().items()}
        extra = [k for k in ck if k not in {"epoch", "best_fitness", "model", "ema", "updates", "modelopt", "optimizer",
                                            "scaler", "train_args", "train_metrics", "train_results", "date",
                                            "version", "license", "docs", "git"}]
        print(f"{nombre}: {len(sd[nombre])} tensores, claves extra en el checkpoint: {extra or 'ninguna'}")
    iguales = sd["kd"] == sd["base"]
    sospechosas = [k for k in sd["kd"] if "adapt" in k.lower() or "teacher" in k.lower() or "maestro" in k.lower()]
    print(f"[{'OK' if iguales and not sospechosas else 'FALLA'}] best.pt KD tiene las mismas claves/formas que el "
          f"base; claves sospechosas: {sospechosas or 'ninguna'}")
    ok &= iguales and not sospechosas

    # 3. ONNX exportado
    n = {}
    for nombre, run in (("kd", run_kd), ("base", run_base)):
        ruta = Path(YOLO(run / "weights" / "best.pt").export(format="onnx", imgsz=640, simplify=True, verbose=False))
        n[nombre] = params_onnx(ruta)
        print(f"{nombre}: ONNX {ruta.name} con {n[nombre]:,} parámetros")
    print(f"[{'OK' if n['kd'] == n['base'] else 'FALLA'}] ONNX KD y base con el mismo número de parámetros")
    ok &= n["kd"] == n["base"]

    print("\nRESULTADO:", "TODO OK" if ok else "HAY FALLAS")


if __name__ == "__main__":
    main()
