"""ECE para detección (D6).

Definición:
  - Se toman las detecciones con confianza ≥ conf_min.
  - En cada imagen, las detecciones se recorren de mayor a menor confianza y cada una se empareja con la caja
    real todavía libre de mayor IoU (de cualquier clase). Si ese IoU ≥ iou_min, la caja real queda usada y la
    detección es CORRECTA si su clase coincide con la de la caja real; si no hay caja con IoU ≥ iou_min, la
    detección es INCORRECTA.
  - ECE = Σ_b (n_b / N) · |acierto_b − confianza_media_b|, con `intervalos` de igual ancho sobre `rango`.

Las cajas reales salen de las etiquetas YOLO del split; las predicciones, del predictions.json de Ultralytics
(caja xywh en píxeles de la imagen original, esquina superior izquierda; category_id empieza en 1).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def cargar_gt(dir_imagenes: Path, dir_etiquetas: Path) -> dict[str, np.ndarray]:
    """{nombre_sin_extensión: array (n, 5) con [clase, x1, y1, x2, y2] en píxeles}."""
    gt = {}
    for img in sorted(dir_imagenes.iterdir()):
        with Image.open(img) as im:
            w, h = im.size
        filas = []
        etq = dir_etiquetas / f"{img.stem}.txt"
        if etq.exists():
            for linea in etq.read_text(encoding="utf-8").splitlines():
                if linea.strip():
                    c, cx, cy, bw, bh = map(float, linea.split())
                    filas.append([c, (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
        gt[img.stem] = np.array(filas, dtype=float).reshape(-1, 5)
    return gt


def cargar_predicciones(ruta: Path) -> dict[str, np.ndarray]:
    """{imagen: array (n, 6) con [clase, conf, x1, y1, x2, y2]} desde predictions.json de Ultralytics."""
    preds = defaultdict(list)
    for d in json.loads(ruta.read_text(encoding="utf-8")):
        x, y, w, h = d["bbox"]
        preds[str(d["image_id"])].append([d["category_id"] - 1, d["score"], x, y, x + w, y + h])
    return {k: np.array(v, dtype=float) for k, v in preds.items()}


def iou_uno_contra_muchos(caja: np.ndarray, cajas: np.ndarray) -> np.ndarray:
    x1 = np.maximum(caja[0], cajas[:, 0]); y1 = np.maximum(caja[1], cajas[:, 1])
    x2 = np.minimum(caja[2], cajas[:, 2]); y2 = np.minimum(caja[3], cajas[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])  # noqa: E731
    return inter / np.maximum(area(caja) + area(cajas) - inter, 1e-12)


def emparejar(gt: dict[str, np.ndarray], preds: dict[str, np.ndarray], conf_min: float,
              iou_min: float, imagenes: list[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (confianzas, aciertos 0/1) de todas las detecciones con conf ≥ conf_min.

    `imagenes` permite repetir imágenes (bootstrap): cada aparición cuenta como una imagen más.
    """
    confs, aciertos = [], []
    for nombre in (imagenes if imagenes is not None else gt):
        p = preds.get(nombre)
        if p is None or len(p) == 0:
            continue
        p = p[p[:, 1] >= conf_min]
        p = p[np.argsort(-p[:, 1], kind="stable")]
        g = gt[nombre]
        libre = np.ones(len(g), dtype=bool)
        for det in p:
            ok = 0
            if libre.any():
                idx = np.flatnonzero(libre)
                ious = iou_uno_contra_muchos(det[2:6], g[idx, 1:5])
                j = int(np.argmax(ious))
                if ious[j] >= iou_min:
                    libre[idx[j]] = False
                    ok = int(det[0] == g[idx[j], 0])
            confs.append(det[1])
            aciertos.append(ok)
    return np.asarray(confs, dtype=float), np.asarray(aciertos, dtype=float)


def ece(confs: np.ndarray, aciertos: np.ndarray, intervalos: int, rango: tuple[float, float]) -> float:
    if len(confs) == 0:
        return float("nan")
    bordes = np.linspace(rango[0], rango[1], intervalos + 1)
    b = np.clip(np.digitize(confs, bordes[1:-1], right=True), 0, intervalos - 1)
    total = 0.0
    for k in range(intervalos):
        m = b == k
        if m.any():
            total += m.sum() / len(confs) * abs(aciertos[m].mean() - confs[m].mean())
    return float(total)
