"""Estadísticas del dataset YOLO: imágenes e instancias por clase y por split.

Lee las etiquetas ya generadas en yolo_dir (verificación independiente de prepare_dataset.py)
y escribe results/tables/dataset_stats.csv.

Uso:
    python -m src.data.dataset_stats [--config configs/data_prep.yaml]
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter

from src.utils.config import RAIZ, cargar_yaml, resolver_ruta

SPLITS = ("train", "val", "test")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data_prep.yaml")
    args = ap.parse_args()

    cfg = cargar_yaml(args.config)
    clases: list[str] = cfg["clases"]
    yolo_dir = resolver_ruta(cfg["yolo_dir"])
    salida = resolver_ruta(cfg["salidas"]["stats_csv"])

    filas = []
    total_inst = Counter()
    por_split: dict[str, tuple[int, int, Counter, Counter]] = {}
    for s in SPLITS:
        etiquetas = sorted((yolo_dir / "labels" / s).glob("*.txt"))
        n_imgs = len(list((yolo_dir / "images" / s).glob("*.jpg")))
        if n_imgs != len(etiquetas):
            raise RuntimeError(f"{s}: {n_imgs} imágenes pero {len(etiquetas)} etiquetas")
        inst, imgs, fondo = Counter(), Counter(), 0
        for e in etiquetas:
            ids = [int(l.split()[0]) for l in e.read_text(encoding="utf-8").splitlines() if l.strip()]
            if not ids:
                fondo += 1
            for i in ids:
                inst[clases[i]] += 1
            for i in set(ids):
                imgs[clases[i]] += 1
        total_inst.update(inst)
        por_split[s] = (n_imgs, fondo, inst, imgs)

    for s in SPLITS:
        n_imgs, fondo, inst, imgs = por_split[s]
        for c in clases:
            pct = 100 * inst[c] / total_inst[c] if total_inst[c] else 0.0
            filas.append([s, c, imgs[c], inst[c], f"{pct:.1f}"])
        filas.append([s, "fondo (sin daños)", fondo, 0, ""])
        filas.append([s, "TOTAL", n_imgs, sum(inst.values()), ""])

    salida.parent.mkdir(parents=True, exist_ok=True)
    with open(salida, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "clase", "imagenes", "instancias", "pct_instancias_de_la_clase"])
        w.writerows(filas)

    # Vista en consola
    print(f"{'split':<6} {'clase':<18} {'imágenes':>9} {'instancias':>11} {'% clase':>8}")
    for fila in filas:
        print(f"{fila[0]:<6} {fila[1]:<18} {fila[2]:>9} {fila[3]:>11} {fila[4]:>8}")
    print(f"\nGuardado en {salida.relative_to(RAIZ)}")

    inst_test = por_split["test"][2]
    for c, minimo in cfg.get("aviso_min_instancias_test", {}).items():
        if inst_test[c] < minimo:
            print(f"⚠️  AVISO: {c} tiene {inst_test[c]} instancias en test (< {minimo}).")


if __name__ == "__main__":
    main()
