"""Fase 7: estadística preespecificada (D7) con bootstrap pareado de imágenes de test.

Implementación:
  - Para cada modelo evaluado en la Fase 6 se toman sus predicciones de test (predictions.json) y se emparejan,
    imagen por imagen, con las cajas reales usando el MISMO algoritmo de Ultralytics (match_predictions, 10
    umbrales de IoU 0.50:0.95). El emparejamiento solo depende de la imagen, así que se calcula una vez.
  - En cada réplica b se sortean (con semilla fija) 1 166 índices de imagen con reemplazo, los MISMOS para todos
    los modelos. Cada imagen aporta sus detecciones y sus cajas reales tantas veces como salió sorteada, y la AP
    se calcula con la misma `ap_per_class` de Ultralytics (AP interpolada en 101 puntos, tipo COCO). Las clases
    sin instancias en la réplica quedan fuera de la mAP de esa réplica en todos los modelos por igual.
  - El ECE de cada réplica usa los pares (confianza, acierto) por imagen de src/eval/ece.py, repetidos igual.
  - La réplica 0 es la muestra original (sin remuestreo): debe reproducir la mAP de Ultralytics (< 0.001).

Contrastes (D > 0 favorece al alumno KD en H1 y H2; cada diferencia se calcula por semilla, emparejando base_s
con kd_s, y se promedia sobre las 3 semillas):
  H1a  dmAP50_base − dmAP50_kd, PTQ INT8         (bilateral)
  H1b  dmAP50_base − dmAP50_kd, QAT INT8         (bilateral)
  H2a  ECE_base − ECE_kd, PTQ INT8               (bilateral)
  H2b  ECE_base − ECE_kd, QAT INT8               (bilateral)
  H3   mAP50(KD + PTQ INT8) − mAP50(base + QAT INT8); no inferioridad con margen δ (unilateral)

Uso (desde la raíz del proyecto, en WSL):
    python -m src.eval.bootstrap
    python -m src.eval.bootstrap --dataset japan
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from ultralytics.utils.metrics import ap_per_class, box_iou

from src.eval.ece import cargar_gt, cargar_predicciones, ece, emparejar
from src.utils.config import cargar_yaml, resolver_ruta
from src.utils.dataset import agregar_argumento, ruta_config

IOUV = np.linspace(0.5, 0.95, 10)
D10, D40 = 1, 3


def match_ultralytics(pred_cls: np.ndarray, true_cls: np.ndarray, iou: np.ndarray) -> np.ndarray:
    """Copia de BaseValidator.match_predictions (Ultralytics 8.4.164, use_scipy=False), en numpy."""
    correct = np.zeros((pred_cls.shape[0], len(IOUV)), dtype=bool)
    iou = iou * (true_cls[:, None] == pred_cls)
    for i, thr in enumerate(IOUV):
        matches = np.array(np.nonzero(iou >= thr)).T
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return correct


class Modelo:
    """Predicciones de un modelo, emparejadas una vez por imagen, listas para remuestrear."""

    def __init__(self, pred_json: Path, gt: dict, nombres: list[str], ece_cfg: dict):
        import torch

        preds = cargar_predicciones(pred_json)
        npz = pred_json.with_name("stats_por_imagen.npz")
        if npz.exists():
            # Aciertos EXACTOS de Ultralytics (sin el redondeo de predictions.json): reproducen su mAP
            z = np.load(npz)
            pos = {n: i for i, n in enumerate(nombres)}
            self.img = np.array([pos[str(z["nombres"][k])] for k in z["img"]], dtype=int)
            self.conf, self.cls, self.tp = z["conf"], z["pred_cls"], z["tp"]
            self.fuente = "stats_por_imagen.npz"
        else:
            self._emparejar_desde_json(preds, gt, nombres)
            self.fuente = "predictions.json"
        self._preparar_ece(preds, gt, nombres, ece_cfg)

    def _emparejar_desde_json(self, preds: dict, gt: dict, nombres: list[str]) -> None:
        import torch

        img_idx, conf, cls, tp = [], [], [], []
        for i, n in enumerate(nombres):
            p, g = preds.get(n), gt[n]
            if p is None or len(p) == 0:
                continue
            if len(g):
                iou = box_iou(torch.tensor(g[:, 1:5]), torch.tensor(p[:, 2:6])).numpy()
                tp.append(match_ultralytics(p[:, 0], g[:, 0], iou))
            else:
                tp.append(np.zeros((len(p), len(IOUV)), dtype=bool))
            img_idx.append(np.full(len(p), i)); conf.append(p[:, 1]); cls.append(p[:, 0])
        self.img, self.conf = np.concatenate(img_idx), np.concatenate(conf)
        self.cls, self.tp = np.concatenate(cls), np.concatenate(tp)

    def _preparar_ece(self, preds: dict, gt: dict, nombres: list[str], ece_cfg: dict) -> None:
        # ECE: pares (confianza, acierto) por imagen, con el emparejamiento de D6 (sobre predictions.json)
        ei, ec, ea = [], [], []
        for i, n in enumerate(nombres):
            c, a = emparejar(gt, preds, ece_cfg["conf_min"], ece_cfg["iou_min"], imagenes=[n])
            ei.append(np.full(len(c), i)); ec.append(c); ea.append(a)
        self.e_img, self.e_conf, self.e_ok = np.concatenate(ei), np.concatenate(ec), np.concatenate(ea)

    def metricas(self, conteo: np.ndarray, gt_img: np.ndarray, gt_cls: np.ndarray, ece_cfg: dict) -> dict:
        r = conteo[self.img]
        tp, conf, cls = np.repeat(self.tp, r, axis=0), np.repeat(self.conf, r), np.repeat(self.cls, r)
        target = np.repeat(gt_cls, conteo[gt_img])
        _, _, p, rec, _, ap, clases, *_ = ap_per_class(tp, conf, cls, target)
        ap50 = dict(zip(clases.astype(int), ap[:, 0]))
        sin_d10 = [k for k in ap50 if k != D10]
        re = conteo[self.e_img]
        return {"mAP50": float(ap[:, 0].mean()), "mAP50_95": float(ap.mean()),
                "AP50_D40": float(ap50.get(D40, np.nan)),
                "mAP50_sin_D10": float(np.mean([ap50[k] for k in sin_d10])),
                "mAP50_95_sin_D10": float(ap[[k != D10 for k in clases.astype(int)]].mean()),
                "P": float(p.mean()), "R": float(rec.mean()),
                "ECE": ece(np.repeat(self.e_conf, re), np.repeat(self.e_ok, re), ece_cfg["intervalos"],
                           tuple(ece_cfg["rango"])),
                "n_clases": len(clases)}


def holm(p: list[float]) -> list[float]:
    m, orden = len(p), np.argsort(p)
    ajust, previo = [0.0] * m, 0.0
    for rango, i in enumerate(orden):
        previo = max(previo, min(1.0, (m - rango) * p[i]))
        ajust[i] = previo
    return ajust


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    agregar_argumento(ap)
    a = ap.parse_args()
    cfg = cargar_yaml(a.config or ruta_config("stats.yaml", a.dataset))
    ev = cargar_yaml(cfg["eval_config"])
    sal = {k: resolver_ruta(v) for k, v in cfg["salidas"].items()}
    t0 = time.perf_counter()

    # Ground truth de test
    d = cargar_yaml(ev["data"])
    raiz = Path(d["path"])
    gt = cargar_gt(raiz / d[ev["split"]], raiz / d[ev["split"]].replace("images", "labels"))
    nombres = sorted(gt)
    n_img = len(nombres)
    gt_img = np.concatenate([np.full(len(gt[n]), i) for i, n in enumerate(nombres)])
    gt_cls = np.concatenate([gt[n][:, 0] for n in nombres])
    print(f"Test: {n_img} imágenes, {len(gt_cls)} cajas", flush=True)

    # Modelos de la Fase 6
    with open(resolver_ruta(ev["tablas"]["metricas"]), encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    modelos = {(r["modelo"], r["formato"], r["precision"]): Modelo(resolver_ruta(r["predicciones"]), gt, nombres, ev["ece"])
               for r in filas}
    print(f"{len(modelos)} modelos preparados ({time.perf_counter() - t0:.0f} s)", flush=True)

    # Réplicas: 0 = muestra original; 1..B = remuestreos (mismos índices para todos los modelos)
    rng = np.random.default_rng(cfg["semilla"])
    conteos = [np.ones(n_img, dtype=int)] + [np.bincount(rng.integers(0, n_img, n_img), minlength=n_img)
                                             for _ in range(cfg["B"])]
    # Verificación ANTES del bootstrap: la muestra original reproduce la mAP de Ultralytics (D7)
    original = {k: m.metricas(conteos[0], gt_img, gt_cls, ev["ece"]) for k, m in modelos.items()}
    verif, ok_global = [], True
    for r in filas:
        k = (r["modelo"], r["formato"], r["precision"])
        d50 = original[k]["mAP50"] - float(r["mAP50"])
        d95 = original[k]["mAP50_95"] - float(r["mAP50_95"])
        dece = original[k]["ECE"] - float(r["ECE"]) if r["ECE"] not in ("", "nan") else 0.0
        ok = max(abs(d50), abs(d95)) < cfg["tolerancia_reproduccion"]
        ok_global &= ok
        verif.append({"modelo": k[0], "formato": k[1], "precision": k[2], "mAP50_ultralytics": r["mAP50"],
                      "mAP50_propia": round(original[k]["mAP50"], 5), "dif_mAP50": round(d50, 6),
                      "dif_mAP50_95": round(d95, 6), "dif_ECE": round(dece, 6), "ok": ok})
    escribir(sal["verificacion"], verif)
    peor = max(verif, key=lambda v: max(abs(v["dif_mAP50"]), abs(v["dif_mAP50_95"])))
    print(f"Reproducción de Ultralytics: {'OK' if ok_global else 'FALLA'} "
          f"(peor caso {peor['modelo']} {peor['formato']} {peor['precision']}: "
          f"Δ mAP50 {peor['dif_mAP50']:+.6f}, Δ mAP50-95 {peor['dif_mAP50_95']:+.6f})", flush=True)
    if not ok_global:
        print("⚠️  La mAP propia no reproduce la de Ultralytics dentro de la tolerancia: se detiene (D7).")
        return

    vals: dict[tuple, dict[str, np.ndarray]] = {}
    for j, (k, m) in enumerate(modelos.items()):
        res = [original[k]] + [m.metricas(c, gt_img, gt_cls, ev["ece"]) for c in conteos[1:]]
        vals[k] = {met: np.array([r[met] for r in res]) for met in res[0]}
        print(f"  bootstrap {j + 1}/{len(modelos)} ({time.perf_counter() - t0:.0f} s)", flush=True)

    # Clases ausentes en alguna réplica (se excluyen en todos los modelos por igual)
    n_clases = next(iter(vals.values()))["n_clases"]
    afectadas = int((n_clases[1:] < n_clases[0]).sum())

    # Estadísticos
    S, mot = cfg["semillas"], cfg["motor"]
    v = lambda g, s, prec, met: vals[(f"student_{g}_s{s}", mot, prec)][met]  # noqa: E731
    dmap = lambda g, s, prec, met="mAP50": v(g, s, cfg["fp32"], met) - v(g, s, prec, met)  # noqa: E731
    porsemilla = {
        "H1a": [dmap("base", s, cfg["ptq_int8"]) - dmap("kd", s, cfg["ptq_int8"]) for s in S],
        "H1b": [dmap("base", s, cfg["qat_int8"]) - dmap("kd", s, cfg["qat_int8"]) for s in S],
        "H2a": [v("base", s, cfg["ptq_int8"], "ECE") - v("kd", s, cfg["ptq_int8"], "ECE") for s in S],
        "H2b": [v("base", s, cfg["qat_int8"], "ECE") - v("kd", s, cfg["qat_int8"], "ECE") for s in S],
        "H3": [v("kd", s, cfg["ptq_int8"], "mAP50") - v("base", s, cfg["qat_int8"], "mAP50") for s in S],
    }
    descripcion = {
        "H1a": "dmAP50_base − dmAP50_kd, PTQ INT8 (TensorRT)",
        "H1b": "dmAP50_base − dmAP50_kd, QAT INT8 (TensorRT)",
        "H2a": "ECE_base − ECE_kd, PTQ INT8",
        "H2b": "ECE_base − ECE_kd, QAT INT8",
        "H3": f"mAP50(KD + PTQ INT8) − mAP50(base + QAT INT8); no inferioridad, δ = {cfg['delta']}",
    }
    sal["distribuciones"].mkdir(parents=True, exist_ok=True)
    B, delta = cfg["B"], cfg["delta"]
    resultados = {}
    for h, lista in porsemilla.items():
        dist = np.mean(lista, axis=0)                  # promedio sobre semillas, réplica a réplica
        np.save(sal["distribuciones"] / f"{h}.npy", dist)
        d_obs, d_b = dist[0], dist[1:]
        if h == "H3":
            p = (1 + np.sum(d_b - d_obs >= d_obs + delta)) / (B + 1)
            nivel = cfg["ic_h3"]
            signos = f"{sum(x[0] > -delta for x in lista)}/{len(S)} semillas con D > −δ"
        else:
            p = (1 + np.sum(np.abs(d_b - d_obs) >= abs(d_obs))) / (B + 1)
            nivel = cfg["ic"]
            signos = f"{sum(x[0] > 0 for x in lista)}/{len(S)} semillas positivas"
        lo, hi = np.percentile(d_b, [100 * (1 - nivel) / 2, 100 * (1 + nivel) / 2])
        resultados[h] = {"hipotesis": h, "estadistico": descripcion[h], "D_obs": d_obs, "IC_inf": lo, "IC_sup": hi,
                         "nivel_IC": nivel, "p_crudo": p, "signos_por_semilla": signos,
                         "D_por_semilla": "; ".join(f"s{s}: {x[0]:+.4f}" for s, x in zip(S, lista))}
    p_holm = holm([resultados[h]["p_crudo"] for h in resultados])
    filas_h = []
    for (h, r), ph in zip(resultados.items(), p_holm):
        rechaza = ph < cfg["alfa"]
        if h == "H3":
            dec = ("rechaza H0: KD + PTQ no inferior a base + QAT" if rechaza
                   else "no rechaza H0: no se demuestra la no inferioridad")
            dec += f" (límite inferior del IC 90 % {'>' if r['IC_inf'] > -delta else '≤'} −δ)"
        else:
            dec = "rechaza H0" if rechaza else "no rechaza H0"
        filas_h.append({k: (round(x, 5) if isinstance(x, float) else x) for k, x in r.items()}
                       | {"p_holm": round(ph, 5), "decision": dec})
    escribir(sal["hipotesis"], [{c: f[c] for c in ("hipotesis", "estadistico", "D_obs", "IC_inf", "IC_sup", "nivel_IC",
                                                    "p_crudo", "p_holm", "decision", "signos_por_semilla",
                                                    "D_por_semilla")} for f in filas_h])

    # Exploratorios (IC 95 %, sin valor p)
    def media(lista):
        return np.mean(lista, axis=0)

    tch = lambda prec, met="mAP50": vals[("teacher_s0", mot, prec)][met]  # noqa: E731
    expl = {
        "D40: dAP50_base − dAP50_kd, PTQ INT8": media([dmap("base", s, "int8", "AP50_D40") - dmap("kd", s, "int8", "AP50_D40") for s in S]),
        "D40: dAP50_base − dAP50_kd, QAT INT8": media([dmap("base", s, "qat_int8", "AP50_D40") - dmap("kd", s, "qat_int8", "AP50_D40") for s in S]),
        "dmAP50-95_base − dmAP50-95_kd, PTQ INT8": media([dmap("base", s, "int8", "mAP50_95") - dmap("kd", s, "int8", "mAP50_95") for s in S]),
        "dmAP50-95_base − dmAP50-95_kd, QAT INT8": media([dmap("base", s, "qat_int8", "mAP50_95") - dmap("kd", s, "qat_int8", "mAP50_95") for s in S]),
        "sin D10: dmAP50_base − dmAP50_kd, PTQ INT8": media([dmap("base", s, "int8", "mAP50_sin_D10") - dmap("kd", s, "int8", "mAP50_sin_D10") for s in S]),
        "sin D10: dmAP50_base − dmAP50_kd, QAT INT8": media([dmap("base", s, "qat_int8", "mAP50_sin_D10") - dmap("kd", s, "qat_int8", "mAP50_sin_D10") for s in S]),
        "sin D10: H3 (KD + PTQ − base + QAT)": media([v("kd", s, "int8", "mAP50_sin_D10") - v("base", s, "qat_int8", "mAP50_sin_D10") for s in S]),
        "dmAP50 maestro − media dmAP50 base, PTQ INT8": (tch("fp32") - tch("int8")) - media([dmap("base", s, "int8") for s in S]),
        "dmAP50 maestro − media dmAP50 KD, PTQ INT8": (tch("fp32") - tch("int8")) - media([dmap("kd", s, "int8") for s in S]),
        "dmAP50 FP16, base (FP32 − FP16)": media([dmap("base", s, "fp16") for s in S]),
        "dmAP50 FP16, KD (FP32 − FP16)": media([dmap("kd", s, "fp16") for s in S]),
        "dmAP50 FP16, maestro (FP32 − FP16)": tch("fp32") - tch("fp16"),
        "P_base − P_kd, PTQ INT8": media([v("base", s, "int8", "P") - v("kd", s, "int8", "P") for s in S]),
        "R_base − R_kd, PTQ INT8": media([v("base", s, "int8", "R") - v("kd", s, "int8", "R") for s in S]),
        "P_base − P_kd, QAT INT8": media([v("base", s, "qat_int8", "P") - v("kd", s, "qat_int8", "P") for s in S]),
        "R_base − R_kd, QAT INT8": media([v("base", s, "qat_int8", "R") - v("kd", s, "qat_int8", "R") for s in S]),
        "dmAP50_base − dmAP50_kd, ModelOpt PTQ INT8 (control)": media([dmap("base", s, "mopt_int8") - dmap("kd", s, "mopt_int8") for s in S]),
        "dmAP50_base − dmAP50_kd, OpenVINO INT8": media([(vals[(f"student_base_s{s}", "openvino", "fp32")]["mAP50"] - vals[(f"student_base_s{s}", "openvino", "int8")]["mAP50"])
                                                         - (vals[(f"student_kd_s{s}", "openvino", "fp32")]["mAP50"] - vals[(f"student_kd_s{s}", "openvino", "int8")]["mAP50"]) for s in S]),
    }
    filas_e = []
    for i, (nombre, dist) in enumerate(expl.items()):
        np.save(sal["distribuciones"] / f"E{i + 1:02d}.npy", dist)
        lo, hi = np.nanpercentile(dist[1:], [100 * (1 - cfg["ic"]) / 2, 100 * (1 + cfg["ic"]) / 2])
        filas_e.append({"id": f"E{i + 1:02d}", "estadistico": nombre, "D_obs": round(float(dist[0]), 5),
                        "IC_inf": round(float(lo), 5), "IC_sup": round(float(hi), 5)})
    escribir(sal["exploratorios"], filas_e)

    # Descriptivo: media ± d.e. de las 3 semillas por grupo/formato/precisión (muestra original)
    grupos = defaultdict(list)
    for r in filas:
        grupos[(r["grupo"], r["formato"], r["precision"])].append(r)
    desc = []
    for (g, fmt, prec), rs in sorted(grupos.items()):
        fila = {"grupo": g, "formato": fmt, "precision": prec, "n": len(rs)}
        for met in ("mAP50", "mAP50_95", "AP50_D40", "mAP50_sin_D10", "P", "R", "ECE"):
            xs = [float(x[met]) for x in rs if x[met] not in ("", "nan")]
            fila[met] = (f"{st.mean(xs):.4f} ± {st.stdev(xs):.4f}" if len(xs) > 1 else
                         (f"{xs[0]:.4f}" if xs else "nan"))
        desc.append(fila)
    escribir(sal["descriptivo"], desc)
    (sal["distribuciones"] / "meta.json").write_text(json.dumps(
        {"B": B, "semilla": cfg["semilla"], "n_imagenes": n_img, "replicas_con_clase_ausente": afectadas,
         "nota": "réplica 0 = muestra original; 1..B = remuestreos"}, indent=2), encoding="utf-8")

    print(f"\nRéplicas con alguna clase ausente (excluida en todos los modelos): {afectadas} de {B}")
    print("\n=== Familia confirmatoria (Holm) ===")
    for f in filas_h:
        print(f"{f['hipotesis']:<4} D_obs {f['D_obs']:+.4f}  IC{int(f['nivel_IC'] * 100)} [{f['IC_inf']:+.4f}, "
              f"{f['IC_sup']:+.4f}]  p {f['p_crudo']:.4f}  p_holm {f['p_holm']:.4f}  → {f['decision']}  "
              f"({f['signos_por_semilla']})")
    print("\n=== Exploratorios (IC 95 %) ===")
    for f in filas_e:
        print(f"{f['id']} {f['estadistico']:<56} {f['D_obs']:+.4f}  [{f['IC_inf']:+.4f}, {f['IC_sup']:+.4f}]")
    print(f"\nTiempo total {time.perf_counter() - t0:.0f} s")


def escribir(ruta: Path, filas: list[dict]) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)


if __name__ == "__main__":
    main()
