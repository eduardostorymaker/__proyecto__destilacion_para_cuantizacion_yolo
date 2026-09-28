"""Fase 1: convierte RDD2022-India (Pascal VOC) a formato YOLO, filtra clases y particiona.

Pasos:
  1. Lee cada XML de raw_dir/annotations/xmls y convierte las cajas de las clases del estudio
     a YOLO (clase cx cy w h, normalizados). Las demás etiquetas se eliminan y se cuentan.
  2. Las imágenes sin cajas de las clases del estudio se conservan como fondo (etiqueta vacía).
  3. Partición 70/15/15 estratificada (ver configs/data_prep.yaml) con semilla fija.
  4. Copia imágenes y etiquetas a yolo_dir/{images,labels}/{train,val,test}/.
  5. Guarda listas en data/splits/, elige 500 imágenes de train para calibración INT8
     y genera configs/data.yaml y configs/calib.yaml.

Uso (desde la raíz del proyecto, en WSL):
    python -m src.data.prepare_dataset [--config configs/data_prep.yaml]
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from tqdm import tqdm

from src.utils.config import RAIZ, cargar_yaml, resolver_ruta
from src.utils.repro import info_entorno

SPLITS = ("train", "val", "test")
MARCA_GENERADO = ".generado_por_prepare_dataset"  # permite borrar yolo_dir con seguridad


@dataclass
class Anotacion:
    nombre: str                                        # nombre de la imagen, p. ej. India_000000.jpg
    lineas_yolo: list[str] = field(default_factory=list)
    clases: list[str] = field(default_factory=list)    # clase de cada caja conservada
    eliminadas: list[str] = field(default_factory=list)


def leer_xml(ruta_xml: Path, clases: list[str]) -> Anotacion:
    raiz = ET.parse(ruta_xml).getroot()
    nombre = raiz.findtext("filename") or ruta_xml.with_suffix(".jpg").name
    ancho = float(raiz.findtext("size/width"))
    alto = float(raiz.findtext("size/height"))
    if ancho <= 0 or alto <= 0:
        raise ValueError(f"Tamaño inválido en {ruta_xml}: {ancho}x{alto}")

    ann = Anotacion(nombre=nombre)
    for obj in raiz.findall("object"):
        cls = (obj.findtext("name") or "").strip()
        if cls not in clases:
            ann.eliminadas.append(cls)
            continue
        caja = obj.find("bndbox")
        x1, y1, x2, y2 = (float(caja.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        # Recorta al borde de la imagen
        x1, x2 = max(0.0, min(x1, ancho)), max(0.0, min(x2, ancho))
        y1, y2 = max(0.0, min(y1, alto)), max(0.0, min(y2, alto))
        if x2 <= x1 or y2 <= y1:
            ann.eliminadas.append(f"{cls}(caja_degenerada)")
            continue
        cx, cy = (x1 + x2) / 2 / ancho, (y1 + y2) / 2 / alto
        w, h = (x2 - x1) / ancho, (y2 - y1) / alto
        ann.lineas_yolo.append(f"{clases.index(cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        ann.clases.append(cls)
    return ann


def estrato(ann: Anotacion, orden_rareza: list[str]) -> str:
    """Clase más rara presente en la imagen, o 'fondo' si no tiene ninguna."""
    presentes = set(ann.clases)
    for cls in orden_rareza:
        if cls in presentes:
            return cls
    return "fondo"


def particionar(anns: list[Anotacion], cfg_part: dict, orden_rareza: list[str]) -> dict[str, list[str]]:
    rng = random.Random(cfg_part["semilla"])
    props = cfg_part["proporciones"]
    grupos: dict[str, list[str]] = defaultdict(list)
    for a in anns:
        clave = estrato(a, orden_rareza) if cfg_part["estratificar"] else "todo"
        grupos[clave].append(a.nombre)

    splits: dict[str, list[str]] = {s: [] for s in SPLITS}
    for clave in sorted(grupos):  # orden fijo -> reproducible
        nombres = sorted(grupos[clave])
        rng.shuffle(nombres)
        n = len(nombres)
        n_train = round(n * props["train"])
        n_val = round(n * props["val"])
        splits["train"] += nombres[:n_train]
        splits["val"] += nombres[n_train : n_train + n_val]
        splits["test"] += nombres[n_train + n_val :]
    return {s: sorted(v) for s, v in splits.items()}


def preparar_salida(yolo_dir: Path) -> None:
    if yolo_dir.exists():
        if not (yolo_dir / MARCA_GENERADO).exists():
            raise RuntimeError(
                f"{yolo_dir} existe pero no fue generado por este script; bórralo a mano o cambia yolo_dir."
            )
        shutil.rmtree(yolo_dir)
    for sub in ("images", "labels"):
        for s in SPLITS:
            (yolo_dir / sub / s).mkdir(parents=True, exist_ok=True)
    (yolo_dir / MARCA_GENERADO).touch()


def escribir_yaml(ruta: Path, contenido: dict, comentario: str) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(comentario)
        yaml.safe_dump(contenido, f, sort_keys=False, allow_unicode=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/data_prep.yaml")
    args = ap.parse_args()

    cfg = cargar_yaml(args.config)
    clases: list[str] = cfg["clases"]
    raw_dir = resolver_ruta(cfg["raw_dir"])
    yolo_dir = resolver_ruta(cfg["yolo_dir"])
    splits_dir = resolver_ruta(cfg["splits_dir"])
    sal = {k: resolver_ruta(v) for k, v in cfg["salidas"].items()}

    # 1-2. Conversión y filtrado
    xmls = sorted((raw_dir / "annotations" / "xmls").glob("*.xml"))
    if not xmls:
        raise FileNotFoundError(f"No hay XML en {raw_dir / 'annotations' / 'xmls'}")
    anns = [leer_xml(x, clases) for x in tqdm(xmls, desc="Leyendo XML")]
    faltantes = [a.nombre for a in anns if not (raw_dir / "images" / a.nombre).exists()]
    if faltantes:
        raise FileNotFoundError(f"{len(faltantes)} imágenes sin archivo, p. ej. {faltantes[:3]}")

    instancias = Counter(c for a in anns for c in a.clases)
    eliminadas = Counter(c for a in anns for c in a.eliminadas)
    fondo_por_filtrado = sum(1 for a in anns if not a.clases and a.eliminadas)
    fondo_original = sum(1 for a in anns if not a.clases and not a.eliminadas)

    # 3. Partición estratificada (rareza = menos instancias primero)
    orden_rareza = sorted(clases, key=lambda c: (instancias[c], c))
    splits = particionar(anns, cfg["particion"], orden_rareza)
    assert sum(len(v) for v in splits.values()) == len(anns)
    assert not (set(splits["train"]) & set(splits["val"]) or set(splits["train"]) & set(splits["test"])
                or set(splits["val"]) & set(splits["test"])), "Hay imágenes repetidas entre splits"

    # 4. Copia al directorio YOLO
    preparar_salida(yolo_dir)
    por_nombre = {a.nombre: a for a in anns}
    for s, nombres in splits.items():
        for nombre in tqdm(nombres, desc=f"Copiando {s}"):
            shutil.copy2(raw_dir / "images" / nombre, yolo_dir / "images" / s / nombre)
            lineas = por_nombre[nombre].lineas_yolo
            (yolo_dir / "labels" / s / Path(nombre).with_suffix(".txt")).write_text(
                "\n".join(lineas) + ("\n" if lineas else ""), encoding="utf-8"
            )

    # 5a. Listas de partición (solo nombres de archivo: portables)
    splits_dir.mkdir(parents=True, exist_ok=True)
    for s, nombres in splits.items():
        (splits_dir / f"{s}.txt").write_text("\n".join(nombres) + "\n", encoding="utf-8")

    # 5b. Calibración INT8: subconjunto SOLO de train
    cc = cfg["calibracion"]
    if cc["n_imagenes"] > len(splits["train"]):
        raise ValueError("Se piden más imágenes de calibración que las disponibles en train")
    calib = sorted(random.Random(cc["semilla"]).sample(splits["train"], cc["n_imagenes"]))
    assert set(calib) <= set(splits["train"])
    (splits_dir / "calib.txt").write_text("\n".join(calib) + "\n", encoding="utf-8")
    calib_lista = yolo_dir / "calib.txt"  # rutas absolutas, formato que lee ultralytics
    calib_lista.write_text(
        "\n".join(str(yolo_dir / "images" / "train" / n) for n in calib) + "\n", encoding="utf-8"
    )

    # 5c. YAMLs para ultralytics
    nombres_cls = dict(enumerate(clases))
    escribir_yaml(
        sal["data_yaml"],
        {"path": str(yolo_dir), "train": "images/train", "val": "images/val", "test": "images/test",
         "names": nombres_cls},
        "# Generado por src/data/prepare_dataset.py — no editar a mano.\n",
    )
    # ultralytics toma las imágenes de calibración INT8 de la clave 'val' (split por defecto del exportador);
    # train y val apuntan ambos a la lista de 500 imágenes de TRAIN para que nunca se use el val real.
    escribir_yaml(
        sal["calib_yaml"],
        {"path": str(yolo_dir), "train": "calib.txt", "val": "calib.txt", "names": nombres_cls},
        "# Generado por src/data/prepare_dataset.py — no editar a mano.\n"
        "# Calibración INT8: 500 imágenes del split TRAIN (nunca val ni test).\n",
    )

    # Registro de etiquetas eliminadas
    sal["etiquetas_eliminadas_csv"].parent.mkdir(parents=True, exist_ok=True)
    with open(sal["etiquetas_eliminadas_csv"], "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["etiqueta", "instancias_eliminadas"])
        for cls, n in sorted(eliminadas.items(), key=lambda t: -t[1]):
            w.writerow([cls, n])
        w.writerow(["TOTAL", sum(eliminadas.values())])

    resumen = {
        "xml_leidos": len(anns),
        "clases": clases,
        "orden_rareza_estratos": orden_rareza,
        "instancias_conservadas": dict(instancias),
        "instancias_eliminadas": dict(eliminadas),
        "imagenes_fondo_original": fondo_original,
        "imagenes_fondo_por_filtrado": fondo_por_filtrado,
        "imagenes_por_split": {s: len(v) for s, v in splits.items()},
        "imagenes_calibracion": len(calib),
        "yolo_dir": str(yolo_dir),
        "entorno": info_entorno(),
    }
    sal["resumen_json"].write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Resumen ===")
    print(f"XML: {len(anns)} | por split: {resumen['imagenes_por_split']}")
    print(f"Instancias conservadas: {dict(instancias)}")
    print(f"Instancias eliminadas: {sum(eliminadas.values())} {dict(eliminadas)}")
    print(f"Fondo: {fondo_original} originales + {fondo_por_filtrado} por filtrado")
    print(f"Salida YOLO: {yolo_dir}")
    print(f"YAMLs: {sal['data_yaml'].relative_to(RAIZ)}, {sal['calib_yaml'].relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
