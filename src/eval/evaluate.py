"""Fase 6: evalúa en TEST todos los modelos (D6) y calcula ECE, consistencia y ΔmAP.

Para cada modelo de configs/quant.yaml se evalúa el .pt FP32 (control de consistencia) y todas sus
exportaciones de results/tables/exports.csv, siempre con letterbox cuadrado 640×640 (rect=False), batch 1,
conf 0.001 e iou 0.7. TensorRT y PyTorch corren en la GPU; OpenVINO en la CPU.

Salidas:
  runs/eval/{modelo}_{formato}_{precision}/predictions.json   predicciones por imagen (para ECE y bootstrap)
  results/metrics.csv                     una fila por modelo/semilla/formato/precisión
  results/tables/consistency_pt_vs_trt.csv  |mAP@0.5(.pt FP32) − mAP@0.5(TRT FP32)| (debe ser < 0.002)
  results/tables/delta_map.csv            Δ = FP32 − cuantizado dentro del mismo motor (TRT: fp16, int8, mopt_int8,
                                          qat_int8; OV: int8)

Reanudable: las combinaciones ya presentes en metrics.csv se saltan.

Uso (desde la raíz del proyecto, en WSL):
    python -m src.eval.evaluate                 # evalúa lo que falte y recalcula las tablas
    python -m src.eval.evaluate --solo-tablas   # solo recalcula consistencia y ΔmAP desde metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics as st
from pathlib import Path

import numpy as np
from ultralytics import YOLO
from ultralytics.utils import metrics as ul_metrics

from src.eval.ece import cargar_gt, cargar_predicciones, ece, emparejar
from src.quant.engines_registry import verificar
from src.utils.config import cargar_yaml, resolver_ruta
from src.utils.dataset import agregar_argumento, ruta_config

CLASES = ["D00", "D10", "D20", "D40"]
SIN_D10 = ["D00", "D20", "D40"]


def describir(modelo: str) -> tuple[str, int]:
    """student_kd_s1 → ('kd', 1); student_base_s0 → ('base', 0); teacher_s0 → ('maestro', 0)."""
    m = re.match(r"(student_(base|kd)|teacher)_s(\d+)", modelo)
    grupo = "maestro" if m.group(1) == "teacher" else m.group(2)
    return grupo, int(m.group(3))


def trabajos(cfg: dict) -> list[tuple[str, str, str, Path]]:
    q = cargar_yaml(cfg["quant_config"])
    lista = [(m, "pytorch", "fp32", resolver_ruta(pt)) for m, pt in q["modelos"].items()]
    with open(resolver_ruta(cfg["exportaciones"]), encoding="utf-8") as f:
        lista += [(r["modelo"], r["formato"], r["precision"], resolver_ruta(r["ruta"])) for r in csv.DictReader(f)]
    return lista


def leer_csv(ruta: Path) -> list[dict]:
    if not ruta.exists():
        return []
    with open(ruta, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def escribir_csv(ruta: Path, filas: list[dict]) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)


# Captura de los aciertos EXACTOS por imagen que calcula Ultralytics (antes del redondeo de predictions.json).
# El bootstrap (Fase 7) los usa para reproducir la mAP de Ultralytics sin desviaciones por redondeo.
_CAPTURA: list[dict] = []
_update_stats_original = ul_metrics.DetMetrics.update_stats


def _update_stats_con_captura(self, stat):
    _CAPTURA.append({k: np.asarray(stat[k]) for k in ("tp", "conf", "pred_cls", "target_cls")}
                    | {"im_name": stat["im_name"]})
    return _update_stats_original(self, stat)


ul_metrics.DetMetrics.update_stats = _update_stats_con_captura


def guardar_stats(destino: Path) -> None:
    """Guarda los aciertos exactos por imagen capturados en la última evaluación."""
    nombres = np.array([Path(c["im_name"]).stem for c in _CAPTURA])
    img = np.concatenate([np.full(len(c["conf"]), i) for i, c in enumerate(_CAPTURA)])
    np.savez_compressed(destino, nombres=nombres, img=img,
                        tp=np.concatenate([c["tp"].reshape(-1, 10) for c in _CAPTURA]).astype(bool),
                        conf=np.concatenate([c["conf"] for c in _CAPTURA]),
                        pred_cls=np.concatenate([c["pred_cls"] for c in _CAPTURA]))


def evaluar(cfg: dict, modelo: str, formato: str, prec: str, ruta: Path, gt: dict) -> dict:
    nombre = f"{modelo}_{formato}_{prec}"
    _CAPTURA.clear()
    v = YOLO(str(ruta), task="detect").val(
        data=str(resolver_ruta(cfg["data"])), split=cfg["split"], imgsz=cfg["imgsz"], batch=cfg["batch"],
        conf=cfg["conf"], iou=cfg["iou"], rect=cfg["rect"], save_json=True, plots=False, verbose=False,
        device="cpu" if formato == "openvino" else 0,
        project=str(resolver_ruta(cfg["salida"])), name=nombre, exist_ok=True)
    pred_json = Path(v.save_dir) / "predictions.json"
    guardar_stats(Path(v.save_dir) / "stats_por_imagen.npz")
    ap50 = {v.names[i]: float(a) for i, a in zip(v.box.ap_class_index, v.box.ap50)}
    ap = {v.names[i]: float(a) for i, a in zip(v.box.ap_class_index, v.box.ap)}
    e = cfg["ece"]
    confs, aciertos = emparejar(gt, cargar_predicciones(pred_json), e["conf_min"], e["iou_min"])
    grupo, semilla = describir(modelo)
    return {
        "modelo": modelo, "grupo": grupo, "semilla": semilla, "formato": formato, "precision": prec,
        "mAP50": round(float(v.box.map50), 5), "mAP50_95": round(float(v.box.map), 5),
        **{f"AP50_{c}": round(ap50.get(c, float("nan")), 5) for c in CLASES},
        **{f"AP50_95_{c}": round(ap.get(c, float("nan")), 5) for c in CLASES},
        "mAP50_sin_D10": round(float(np.mean([ap50[c] for c in SIN_D10])), 5),
        "mAP50_95_sin_D10": round(float(np.mean([ap[c] for c in SIN_D10])), 5),
        "P": round(float(v.box.mp), 5), "R": round(float(v.box.mr), 5),
        "ECE": round(ece(confs, aciertos, e["intervalos"], tuple(e["rango"])), 5),
        "n_det_ece": len(confs), "acierto_medio_ece": round(float(aciertos.mean()), 5) if len(confs) else "",
        "conf_media_ece": round(float(confs.mean()), 5) if len(confs) else "",
        "predicciones": str(pred_json.relative_to(resolver_ruta("."))),
    }


def tablas(cfg: dict) -> None:
    filas = leer_csv(resolver_ruta(cfg["tablas"]["metricas"]))
    idx = {(r["modelo"], r["formato"], r["precision"]): r for r in filas}
    modelos = sorted({r["modelo"] for r in filas})
    f = lambda r, k: float(r[k])  # noqa: E731

    # Control de consistencia: .pt FP32 (rect=False) frente a TRT FP32
    cons = []
    for m in modelos:
        pt, trt = idx.get((m, "pytorch", "fp32")), idx.get((m, "tensorrt", "fp32"))
        if pt and trt:
            d = f(pt, "mAP50") - f(trt, "mAP50")
            cons.append({"modelo": m, "mAP50_pt": pt["mAP50"], "mAP50_trt_fp32": trt["mAP50"],
                         "dif": round(d, 5), "ok": abs(d) < cfg["consistencia_max_dmap50"]})
    if cons:
        escribir_csv(resolver_ruta(cfg["tablas"]["consistencia"]), cons)

    # Δ dentro del mismo motor: FP32 − cuantizado (positivo = pérdida); para ECE, cuantizado − FP32
    deltas = []
    metricas = ["mAP50", "mAP50_95", "AP50_D40", "mAP50_sin_D10", "P", "R"]
    for m in modelos:
        for motor, precs in (("tensorrt", ["fp16", "int8", "mopt_int8", "qat_int8"]), ("openvino", ["int8"])):
            ref = idx.get((m, motor, "fp32"))
            for p in precs:
                q = idx.get((m, motor, p))
                if ref and q:
                    grupo, semilla = describir(m)
                    deltas.append({"modelo": m, "grupo": grupo, "semilla": semilla, "motor": motor, "precision": p,
                                   **{f"d_{k}": round(f(ref, k) - f(q, k), 5) for k in metricas},
                                   "ECE_fp32": ref["ECE"], "ECE_cuant": q["ECE"],
                                   "d_ECE_cuant_menos_fp32": round(f(q, "ECE") - f(ref, "ECE"), 5)})
    if deltas:
        escribir_csv(resolver_ruta(cfg["tablas"]["deltas"]), deltas)

    # Resumen en consola (descriptivo; la inferencia es la Fase 7)
    print("\n=== Consistencia .pt FP32 vs TRT FP32 (|Δ mAP@0.5| < "
          f"{cfg['consistencia_max_dmap50']}) ===")
    for c in cons:
        print(f"  {c['modelo']:<16} pt {c['mAP50_pt']}  trt {c['mAP50_trt_fp32']}  Δ {c['dif']:+.5f}  "
              f"{'OK' if c['ok'] else '⚠️  FUERA DE TOLERANCIA'}")
    print("\n=== Δ = FP32 − cuantizado (media ± d.e. por grupo) ===")
    for motor, p in (("tensorrt", "fp16"), ("tensorrt", "int8"), ("tensorrt", "mopt_int8"), ("tensorrt", "qat_int8"),
                     ("openvino", "int8")):
        for g in ("base", "kd", "maestro"):
            sel = [d for d in deltas if d["motor"] == motor and d["precision"] == p and d["grupo"] == g]
            if not sel:
                continue
            txt = []
            for k in ("d_mAP50", "d_mAP50_95", "d_AP50_D40", "d_ECE_cuant_menos_fp32"):
                xs = [d[k] for d in sel]
                txt.append(f"{k} {st.mean(xs):+.4f}" + (f"±{st.stdev(xs):.4f}" if len(xs) > 1 else ""))
            print(f"  {motor:<8} {p:<4} {g:<7} (n={len(sel)})  " + "  ".join(txt))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="por defecto, eval.yaml del dataset")
    agregar_argumento(ap)
    ap.add_argument("--solo-tablas", action="store_true")
    ap.add_argument("--solo-stats", action="store_true",
                    help="reevalúa los modelos ya evaluados solo para guardar stats_por_imagen.npz (sin tocar metrics.csv)")
    a = ap.parse_args()
    a.config = a.config or ruta_config("eval.yaml", a.dataset)
    cfg = cargar_yaml(a.config)

    if a.solo_stats:
        d = cargar_yaml(cfg["data"])
        raiz = Path(d["path"])
        gt = cargar_gt(raiz / d[cfg["split"]], raiz / d[cfg["split"]].replace("images", "labels"))
        previas = {(r["modelo"], r["formato"], r["precision"]): r for r in leer_csv(resolver_ruta(cfg["tablas"]["metricas"]))}
        for modelo, formato, prec, ruta in trabajos(cfg):
            npz = resolver_ruta(cfg["salida"]) / f"{modelo}_{formato}_{prec}" / "stats_por_imagen.npz"
            r = previas.get((modelo, formato, prec))
            if r is None or npz.exists():
                continue
            if formato != "pytorch":
                verificar(ruta)
            fila = evaluar(cfg, modelo, formato, prec, ruta, gt)
            iguales = abs(fila["mAP50"] - float(r["mAP50"])) < 1e-9 and abs(fila["mAP50_95"] - float(r["mAP50_95"])) < 1e-9
            print(f"   {modelo} {formato} {prec}: stats guardados; mAP igual a metrics.csv: {iguales}", flush=True)
            if not iguales:
                raise RuntimeError("La reevaluación no reproduce metrics.csv: revisar antes de seguir")
        print("FIN", flush=True)
        return

    if not a.solo_tablas:
        d = cargar_yaml(cfg["data"])
        raiz = Path(d["path"])
        gt = cargar_gt(raiz / d[cfg["split"]], raiz / d[cfg["split"]].replace("images", "labels"))
        print(f"Ground truth de {cfg['split']}: {len(gt)} imágenes, "
              f"{sum(len(v) for v in gt.values())} cajas", flush=True)
        tabla = resolver_ruta(cfg["tablas"]["metricas"])
        hechos = {(r["modelo"], r["formato"], r["precision"]) for r in leer_csv(tabla)}
        for modelo, formato, prec, ruta in trabajos(cfg):
            if (modelo, formato, prec) in hechos:
                continue
            print(f">> {modelo} {formato} {prec}", flush=True)
            if formato != "pytorch":
                verificar(ruta)  # se evalúa siempre el engine registrado (hash SHA-256 sin cambios)
            fila = evaluar(cfg, modelo, formato, prec, ruta, gt)
            nuevo = not tabla.exists()
            tabla.parent.mkdir(parents=True, exist_ok=True)
            with open(tabla, "a", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(fila))
                if nuevo:
                    w.writeheader()
                w.writerow(fila)
            print(f"   mAP50 {fila['mAP50']}  mAP50-95 {fila['mAP50_95']}  D40 {fila['AP50_D40']}  "
                  f"ECE {fila['ECE']}", flush=True)
    tablas(cfg)
    print("FIN", flush=True)


if __name__ == "__main__":
    main()
