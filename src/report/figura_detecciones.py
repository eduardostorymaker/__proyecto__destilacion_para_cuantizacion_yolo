"""Ejemplos de detección para la presentación (configs/report.yaml, sección `detecciones`).

No vuelve a correr ningún modelo: dibuja las predicciones de prueba ya guardadas por la Fase 6
(runs/eval/*/predictions.json, las mismas con las que se calcularon la mAP y el ECE) sobre las MISMAS imágenes en
todas las comparaciones. Una figura por comparación e imagen (p. ej., base frente a KD con PTQ INT8): GT, modelo A
y modelo B en una fila. En cada panel se anotan los aciertos (IoU ≥ 0.5 y misma clase, emparejamiento greedy
por confianza, como el ECE) y los falsos positivos de las detecciones con conf ≥ conf_min.

Salidas:
  results/figures/presentacion/detecciones_{comparacion}_{imagen}.png
  results/tables/detecciones_seleccion.csv   imágenes elegidas, regla y conteos por modelo

Uso (en WSL, donde están las imágenes de prueba): python -m src.report.figura_detecciones [--dataset india]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

from src.eval.ece import cargar_gt, cargar_predicciones, iou_uno_contra_muchos  # noqa: E402
from src.utils.config import cargar_yaml, resolver_ruta  # noqa: E402
from src.utils.dataset import agregar_argumento, ruta_config  # noqa: E402

CFG = cargar_yaml("configs/report.yaml")["detecciones"]
CLASES = ["D00", "D10", "D20", "D40"]
# Paleta categórica de las figuras de la presentación (clases = slots 1–4 en orden fijo)
COLOR_CLASE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
TINTA, TINTA_2 = "#0b0b0b", "#52514e"
NOMBRE_GRUPO = {"base": "Base", "kd": "KD", "maestro": "Maestro"}
NOMBRE_PREC = {"fp32": "FP32", "fp16": "FP16", "int8": "PTQ INT8", "mopt_int8": "PTQ-MO INT8", "qat_int8": "QAT INT8"}


def evaluar(gt: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Detecciones con conf ≥ conf_min (ordenadas por confianza) y su acierto 0/1, con las reglas del ECE."""
    p = p[p[:, 1] >= CFG["conf_min"]] if len(p) else p.reshape(0, 6)
    p = p[np.argsort(-p[:, 1], kind="stable")]
    libre = np.ones(len(gt), dtype=bool)
    ok = np.zeros(len(p), dtype=int)
    for k, det in enumerate(p):
        if libre.any():
            idx = np.flatnonzero(libre)
            ious = iou_uno_contra_muchos(det[2:6], gt[idx, 1:5])
            j = int(np.argmax(ious))
            if ious[j] >= CFG["iou_min"]:
                libre[idx[j]] = False
                ok[k] = int(det[0] == gt[idx[j], 0])
    return p, ok


def elegir_semillas(m: pd.DataFrame, motor: str) -> dict[str, int]:
    """Semilla por grupo: la de mAP@0.5 mediana en el FP32 del motor (o la fijada en el yaml)."""
    sel = {}
    for g, d in m[(m.formato == motor) & (m.precision == "fp32")].groupby("grupo"):
        if CFG["semilla"] == "mediana":
            d = d.sort_values(["mAP50", "semilla"]).reset_index(drop=True)
            sel[g] = int(d.loc[(len(d) - 1) // 2, "semilla"])
        else:
            sel[g] = int(CFG["semilla"]) if g != "maestro" else int(d.semilla.iloc[0])
    return sel


def f1_imagen(gt: np.ndarray, p: np.ndarray) -> float:
    """F1 de una imagen: 2·aciertos / (detecciones + cajas reales), con las reglas de `evaluar`."""
    det, ok = evaluar(gt, p)
    return 2 * ok.sum() / (len(det) + len(gt))


def elegir_imagenes(gt: dict, preds: dict, modelos: list[str]) -> list[tuple[str, str, float]]:
    """Una imagen por nivel de exactitud: la del percentil indicado del F1 medio de los modelos mostrados, entre las
    imágenes con al menos `min_cajas` cajas reales; sin repetir; empates por nombre. Devuelve [(imagen, regla, F1)]."""
    sel = CFG["seleccion"]
    candidatas = sorted(n for n in gt if len(gt[n]) >= sel["min_cajas"])
    f1 = {n: float(np.mean([f1_imagen(gt[n], preds[k].get(n, np.zeros((0, 6)))) for k in modelos]))
          for n in candidatas}
    if CFG["imagenes"]:
        return [(n, "manual", f1.get(n, float("nan"))) for n in CFG["imagenes"]]
    elegidas = []
    for regla, pct in sel["percentiles"].items():
        objetivo = float(np.percentile(list(f1.values()), pct))
        libres = [n for n in candidatas if n not in {e for e, _, _ in elegidas}]
        # la más cercana al percentil; min estable: ante empates gana el primer nombre en orden alfabético
        n = min(libres, key=lambda x: abs(f1[x] - objetivo))
        elegidas.append((n, regla, f1[n]))
    return elegidas


def dibujar(eje, img: Image.Image, cajas: np.ndarray, ok: np.ndarray | None, lw: float = 2.0,
            tam: float = 6.5) -> None:
    """Cajas de la GT (ok=None) o de un modelo (línea continua = acierto, discontinua = falso positivo)."""
    eje.imshow(img)
    eje.set_xticks([]), eje.set_yticks([])
    for s in eje.spines.values():
        s.set_color("#c9c8c3")
    for k, c in enumerate(cajas):
        cls, (x1, y1, x2, y2) = int(c[0]), (c[1:5] if ok is None else c[2:6])
        col = COLOR_CLASE[cls]
        acierto = ok is None or ok[k]
        eje.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=col, lw=lw,
                                ls="-" if acierto else (0, (3, 2))))
        etiqueta = CLASES[cls] if ok is None else f"{CLASES[cls]} {c[1]:.2f}"
        eje.text(x1 + 2, y1 - 4, etiqueta, fontsize=tam, color="white", va="bottom",
                 bbox=dict(boxstyle="square,pad=0.15", fc=col, ec="none"))


def figura_paper(imagenes, ext, gt, preds, claves, m, semillas, manejadores) -> None:
    """Una sola figura de página completa para el apéndice del paper: filas = imágenes (exactitud alta, media y
    baja), columnas = GT y los dos modelos de cada comparación (configs/report.yaml, detecciones.paper)."""
    cp = CFG["paper"]
    modelos = [tuple(x) for _, (_, A, B) in CFG["comparaciones"].items() for x in (A, B)]
    fig, ejes = plt.subplots(len(imagenes), 1 + len(modelos), figsize=(cp["ancho_pulgadas"], cp["alto_pulgadas"]),
                             squeeze=False)
    for i, (img_n, regla, f1_medio) in enumerate(imagenes):
        img = Image.open(ext[img_n]).convert("RGB")
        g_ = gt[img_n]
        dibujar(ejes[i, 0], img, g_, None, lw=0.8, tam=3.5)
        ejes[i, 0].set_ylabel(f"Exactitud {regla}\n{img_n}\nF1 medio {f1_medio:.2f}", fontsize=6.5, color=TINTA)
        ejes[i, 0].set_xlabel(f"{len(g_)} cajas reales", fontsize=6, color=TINTA)
        for j, (g, f, p) in enumerate(modelos, start=1):
            det, ok = evaluar(g_, preds[claves[(g, f, p)]].get(img_n, np.zeros((0, 6))))
            dibujar(ejes[i, j], img, det, ok, lw=0.8, tam=3.5)
            n_ok = int(ok.sum())
            ejes[i, j].set_xlabel(f"{n_ok} ac. · {len(ok) - n_ok} FP · {len(g_) - n_ok} no det.", fontsize=6,
                                  color=TINTA)
    ejes[0, 0].set_title("Verdad de terreno", fontsize=7, fontweight="bold", color=TINTA)
    for j, (g, f, p) in enumerate(modelos, start=1):
        fila = lambda prec: m[(m.grupo == g) & (m.semilla == semillas[g]) & (m.formato == f)  # noqa: E731
                              & (m.precision == prec)].iloc[0]
        r, r32 = fila(p), fila("fp32")
        ejes[0, j].set_title(f"{NOMBRE_GRUPO[g]} s{semillas[g]} · {NOMBRE_PREC[p]}\nmAP {r.mAP50:.3f} · "
                             f"ΔmAP {r32.mAP50 - r.mAP50:.3f}\nECE {r.ECE:.3f} · AP D40 {r.AP50_D40:.3f}",
                             fontsize=6, color=TINTA)
    fig.legend(manejadores, CLASES + ["acierto", "falso positivo"], loc="lower center", ncol=6, frameon=False,
               fontsize=6.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1), h_pad=0.6, w_pad=0.3)
    salida = resolver_ruta(cp["salida"])
    for ext_ in ("png", "pdf"):
        fig.savefig(salida.with_suffix(f".{ext_}"), dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Figura (paper): {salida}.png/.pdf")


def main() -> None:
    ap = argparse.ArgumentParser()
    agregar_argumento(ap)
    a = ap.parse_args()
    ev = cargar_yaml(ruta_config("eval.yaml", a.dataset))
    d = cargar_yaml(ev["data"])
    raiz = Path(d["path"])
    dir_img = raiz / d[ev["split"]]
    gt = cargar_gt(dir_img, raiz / d[ev["split"]].replace("images", "labels"))
    m = pd.read_csv(resolver_ruta(ev["tablas"]["metricas"]))
    semillas = elegir_semillas(m, "tensorrt")
    print("Semillas elegidas:", semillas)

    # predicciones de todos los modelos que aparecen en las comparaciones
    comp = CFG["comparaciones"]
    necesarios = {tuple(x) for _, (_, A, B) in comp.items() for x in (A, B)}
    preds, claves = {}, {}
    for g, f, p in sorted(necesarios):
        fila = m[(m.grupo == g) & (m.semilla == semillas[g]) & (m.formato == f) & (m.precision == p)]
        assert len(fila) == 1, (g, f, p)
        fila = fila.iloc[0]
        preds[fila.modelo + "_" + f + "_" + p] = cargar_predicciones(resolver_ruta(fila.predicciones))
        claves[(g, f, p)] = fila.modelo + "_" + f + "_" + p

    imagenes = elegir_imagenes(gt, preds, list(claves.values()))
    print("Imágenes:", imagenes)
    ext = {p.stem: p for p in dir_img.iterdir()}

    filas_csv = []
    salida = resolver_ruta(CFG["salida"])
    salida.mkdir(parents=True, exist_ok=True)
    manejadores = [Rectangle((0, 0), 1, 1, fill=False, ec=c, lw=2) for c in COLOR_CLASE]
    manejadores += [Rectangle((0, 0), 1, 1, fill=False, ec=TINTA_2, lw=2),
                    Rectangle((0, 0), 1, 1, fill=False, ec=TINTA_2, lw=2, ls=(0, (3, 2)))]
    # una figura por comparación e imagen: GT | modelo A | modelo B, en una fila (paneles grandes)
    for nombre, (titulo, A, B) in comp.items():
        for img_n, regla, f1_medio in imagenes:
            img = Image.open(ext[img_n]).convert("RGB")
            g_ = gt[img_n]
            fig, ejes = plt.subplots(1, 3, figsize=(13.5, 5.0))
            dibujar(ejes[0], img, g_, None)
            ejes[0].set_title(f"Verdad de terreno · {len(g_)} cajas\nMétricas: prueba completa (1 166 imágenes)\n"
                              f"ΔmAP = FP32 − cuantizado (mismo motor)", fontsize=11, color=TINTA)
            for eje, (g, f, p) in zip(ejes[1:], (A, B)):
                clave = claves[(g, f, p)]
                det, ok = evaluar(g_, preds[clave].get(img_n, np.zeros((0, 6))))
                dibujar(eje, img, det, ok)
                n_ok = int(ok.sum())
                etiqueta = f"{NOMBRE_GRUPO[g]}{'' if g == 'maestro' else f' s{semillas[g]}'} · {NOMBRE_PREC[p]}"
                # métricas del modelo en TODA la prueba (metrics.csv) y su pérdida frente al FP32 del mismo motor (D6)
                fila = lambda prec: m[(m.grupo == g) & (m.semilla == semillas[g]) & (m.formato == f)  # noqa: E731
                                      & (m.precision == prec)].iloc[0]
                r, r32 = fila(p), fila("fp32")
                eje.set_title(f"{etiqueta}\nmAP@0.5 {r.mAP50:.3f} · ΔmAP {r32.mAP50 - r.mAP50:.3f} · ECE {r.ECE:.3f}\n"
                              f"AP@0.5 baches (D40) {r.AP50_D40:.3f}\n"
                              f"Esta imagen: {n_ok} acierto{'' if n_ok == 1 else 's'} · "
                              f"{int(len(ok) - n_ok)} FP · {int(len(g_) - n_ok)} sin detectar", fontsize=11, color=TINTA)
                filas_csv.append({"comparacion": nombre, "imagen": img_n, "regla": regla,
                                  "f1_medio_imagen": round(f1_medio, 4), "modelo": clave,
                                  "mAP50_prueba": r.mAP50, "dmAP50_vs_fp32": round(r32.mAP50 - r.mAP50, 5),
                                  "ECE_prueba": r.ECE, "AP50_D40_prueba": r.AP50_D40,
                                  "f1_modelo": round(f1_imagen(g_, preds[clave].get(img_n, np.zeros((0, 6)))), 4),
                                  "n_gt": len(g_), "detecciones": len(ok), "aciertos": n_ok,
                                  "falsos_positivos": int(len(ok) - n_ok), "sin_detectar": int(len(g_) - n_ok)})
            fig.legend(manejadores, CLASES + ["acierto", "falso positivo"], loc="lower center", ncol=6,
                       frameon=False, fontsize=11)
            fig.tight_layout(rect=(0, 0.06, 1, 1))
            ruta = salida / f"detecciones_{nombre}_{img_n}.png"
            fig.savefig(ruta, dpi=CFG["dpi"], facecolor="white", bbox_inches="tight")
            plt.close(fig)
            print(f"Figura: {ruta}")

    figura_paper(imagenes, ext, gt, preds, claves, m, semillas, manejadores)
    tabla = resolver_ruta(CFG["tabla"])
    pd.DataFrame(filas_csv).to_csv(tabla, index=False)
    print(f"Tabla: {tabla}")


if __name__ == "__main__":
    main()
