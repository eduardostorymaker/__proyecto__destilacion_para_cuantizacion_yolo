"""Fase 4b: analiza el piloto de λ y propone el λ para los alumnos KD finales.

Para cada λ de configs/kd.yaml (piloto.lambdas), lee runs/pilot/pilot_kd_l{λ}_s0:
  - results.csv: L_det (box+cls+dfl) y L_CWD (train/cwd, sin λ) por época; comprueba que L_CWD baja,
    que no hay NaN y cuánto pesa λ·L_CWD frente a L_det.
  - revalida best.pt en VALIDACIÓN (piloto.revalidacion) y elige el λ con mayor mAP@0.5:0.95.
No modifica kd.yaml: imprime la propuesta; λ se fija tras confirmación del usuario.

Uso:
    python -m src.train.select_lambda                 # corridas reales del piloto
    python -m src.train.select_lambda --humo          # corridas de la prueba de humo (runs/smoke/..._humo)
"""

from __future__ import annotations

import argparse
import csv
import math

import matplotlib
import numpy as np
from scipy.stats import linregress

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from src.utils.config import RAIZ, cargar_yaml, resolver_ruta  # noqa: E402

COLORES = ["#2a78d6", "#eb6834", "#1baf7a"]  # paleta categórica de referencia, orden fijo


def leer_curvas(run) -> dict:
    with open(run / "results.csv", encoding="utf-8") as f:
        filas = [{k.strip(): float(v) for k, v in r.items()} for r in csv.DictReader(f)]
    ep = np.array([r["epoch"] for r in filas])
    det = np.array([r["train/box_loss"] + r["train/cls_loss"] + r["train/dfl_loss"] for r in filas])
    cwd = np.array([r["train/cwd"] for r in filas])
    m5095 = np.array([r["metrics/mAP50-95(B)"] for r in filas])
    nan = any(not math.isfinite(v) for r in filas for v in r.values())
    return {"ep": ep, "det": det, "cwd": cwd, "m5095": m5095, "nan": nan}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--humo", action="store_true")
    a = ap.parse_args()

    kd = cargar_yaml("configs/kd.yaml")
    p, rv = kd["piloto"], kd["piloto"]["revalidacion"]
    base = resolver_ruta(kd["humo"]["project"] if a.humo else p["project"])
    sufijo = "_humo" if a.humo else ""

    filas, curvas = [], {}
    for lam in p["lambdas"]:
        run = base / f"pilot_kd_l{float(lam)}_s{p['seed']}{sufijo}"
        c = leer_curvas(run)
        curvas[lam] = c
        pendiente = linregress(c["ep"], c["cwd"]).slope if len(c["ep"]) > 2 else c["cwd"][-1] - c["cwd"][0]
        m = YOLO(run / "weights" / "best.pt")
        v = m.val(data=str(resolver_ruta("configs/data.yaml")), split=rv["split"], imgsz=rv["imgsz"],
                  batch=rv["batch"], conf=rv["conf"], iou=rv["iou"], plots=False, verbose=False,
                  project="/tmp/val_pilot", name=f"l{lam}{sufijo}", exist_ok=True)
        ap50 = {m.names[i]: x for i, x in zip(v.box.ap_class_index, v.box.ap50)}
        filas.append({
            "lambda": lam,
            "epocas": len(c["ep"]),
            "L_CWD_ep1": round(float(c["cwd"][0]), 4),
            "L_CWD_ult": round(float(c["cwd"][-1]), 4),
            "L_CWD_baja": bool(c["cwd"][-1] < c["cwd"][0] and pendiente < 0),
            "L_det_ult": round(float(c["det"][-1]), 4),
            "peso_lambdaCWD_sobre_Ldet_pct": round(float(100 * lam * c["cwd"][-1] / c["det"][-1]), 1),
            "nan": c["nan"],
            "val_mAP50": round(float(v.box.map50), 4),
            "val_mAP50_95": round(float(v.box.map), 4),
            "val_AP50_D40": round(float(ap50["D40"]), 4),
            "val_mAP50_sin_D10": round(float(np.mean([ap50[k] for k in ("D00", "D20", "D40")])), 4),
        })

    elegido = max(filas, key=lambda f: f["val_mAP50_95"])
    tabla = resolver_ruta(p["tabla"]).with_name(resolver_ruta(p["tabla"]).stem + sufijo + ".csv")
    tabla.parent.mkdir(parents=True, exist_ok=True)
    with open(tabla, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)

    # Figura: L_CWD por época y mAP@0.5:0.95 de validación por época, una línea por λ
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), facecolor="#fcfcfb")
    for (lam, c), col in zip(curvas.items(), COLORES):
        ax[0].plot(c["ep"], c["cwd"], color=col, lw=2, label=f"λ={lam}")
        ax[1].plot(c["ep"], lam * c["cwd"] / c["det"] * 100, color=col, lw=2, label=f"λ={lam}")
        ax[2].plot(c["ep"], c["m5095"], color=col, lw=2, label=f"λ={lam}")
    for a_, t, y in zip(ax, ("L_CWD (sin λ) por época", "Peso de λ·L_CWD sobre L_det (%)",
                             "mAP@0.5:0.95 de validación"), ("L_CWD", "%", "mAP")):
        a_.set_facecolor("#fcfcfb"); a_.grid(color="#e8e7e3", lw=0.6)
        a_.set_title(t, loc="left"); a_.set_xlabel("Época"); a_.set_ylabel(y); a_.legend(frameon=False)
    fig.tight_layout()
    figura = resolver_ruta(p["figura"]).with_name(resolver_ruta(p["figura"]).stem + sufijo + ".png")
    figura.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figura, dpi=160)

    print(f"\n{'λ':>5} {'ép':>4} {'L_CWD 1→últ':>16} {'baja':>5} {'λL_CWD/L_det':>13} {'NaN':>4} "
          f"{'mAP50':>7} {'mAP50-95':>9} {'D40':>7} {'sinD10':>7}")
    for f_ in filas:
        print(f"{f_['lambda']:>5} {f_['epocas']:>4} {f_['L_CWD_ep1']:>7.4f}→{f_['L_CWD_ult']:<8.4f} "
              f"{str(f_['L_CWD_baja']):>5} {f_['peso_lambdaCWD_sobre_Ldet_pct']:>12}% {str(f_['nan']):>4} "
              f"{f_['val_mAP50']:>7.4f} {f_['val_mAP50_95']:>9.4f} {f_['val_AP50_D40']:>7.4f} "
              f"{f_['val_mAP50_sin_D10']:>7.4f}")
    print(f"\nPropuesta (mayor mAP@0.5:0.95 de val): λ = {elegido['lambda']}")
    print(f"Tabla: {tabla.relative_to(RAIZ)} | Figura: {figura.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
