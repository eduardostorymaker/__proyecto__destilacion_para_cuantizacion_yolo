"""Fase 1: convierte RDD2022-India (Pascal VOC) a formato YOLO con las reglas de limpieza D1.

Reglas (en este orden; cada conteo se registra en results/tables/data_audit.csv):
  1. Etiquetas de otras clases: se elimina SOLO esa caja; la imagen se conserva.
  2. Cajas fuera de la imagen: se recortan a los bordes.
  3. Cajas con ancho o alto < min_lado_px tras recortar: se eliminan.
  4. Cajas duplicadas idénticas (misma clase y coordenadas): se deja una.
  5. Imágenes corruptas o ilegibles: se elimina la imagen.
  6. Imágenes sin cajas tras la limpieza: se conservan como fondo (.txt vacío).
Ninguna imagen se elimina por otro motivo que no sea la regla 5.

Después: partición 70/15/15 por estratificación iterativa multietiqueta (4 clases + fondo),
500 imágenes de calibración INT8 estratificadas dentro de train, y generación de
configs/data.yaml y configs/calib.yaml.

Uso (desde la raíz del proyecto, en WSL):
    python -m src.data.prepare_dataset [--config configs/data_prep.yaml]
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from PIL import Image
from tqdm import tqdm

from src.utils.config import RAIZ, cargar_yaml, resolver_ruta
from src.utils.repro import info_entorno

SPLITS = ("train", "val", "test")
MARCA_GENERADO = ".generado_por_prepare_dataset"  # permite borrar yolo_dir con seguridad


@dataclass
class Imagen:
    nombre: str
    ancho: int = 0
    alto: int = 0
    corrupta: bool = False
    tam_xml_distinto: bool = False
    n_objetos_xml: int = 0                             # objetos en el XML original (cualquier etiqueta)
    cajas: list[tuple[int, float, float, float, float]] = field(default_factory=list)  # (cls, x1, y1, x2, y2)


def leer_imagen(ruta: Path) -> tuple[int, int] | None:
    """Devuelve (ancho, alto) si la imagen se decodifica completa; None si está corrupta."""
    try:
        with Image.open(ruta) as im:
            im.verify()
        with Image.open(ruta) as im:  # verify() invalida el objeto: se reabre y se decodifica
            im.load()
            return im.size
    except Exception:  # noqa: BLE001
        return None


def procesar(ruta_xml: Path, dir_imgs: Path, clases: list[str], min_lado: float,
             aud: dict[str, Counter]) -> Imagen:
    """Aplica las reglas 1-5 a una imagen y acumula los conteos en `aud` (por etiqueta original)."""
    raiz = ET.parse(ruta_xml).getroot()
    nombre = raiz.findtext("filename") or ruta_xml.with_suffix(".jpg").name
    img = Imagen(nombre=nombre)

    objetos = [((o.findtext("name") or "").strip(), o.find("bndbox")) for o in raiz.findall("object")]
    img.n_objetos_xml = len(objetos)
    for etq, _ in objetos:
        aud["original"][etq] += 1

    tam = leer_imagen(dir_imgs / nombre)
    if tam is None:  # regla 5
        img.corrupta = True
        for etq, _ in objetos:
            aud["r5_imagen_corrupta"][etq] += 1
        return img
    img.ancho, img.alto = tam
    # Se normaliza con el tamaño real de la imagen; se registra si el XML dice otra cosa
    ancho_xml, alto_xml = int(float(raiz.findtext("size/width") or 0)), int(float(raiz.findtext("size/height") or 0))
    img.tam_xml_distinto = (ancho_xml, alto_xml) != (img.ancho, img.alto)

    vistas: set[tuple] = set()
    for etq, caja in objetos:
        if etq not in clases:  # regla 1
            aud["r1_otra_clase"][etq] += 1
            continue
        x1, y1, x2, y2 = (float(caja.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        # regla 2: recorte a los bordes
        cx1, cx2 = min(max(x1, 0.0), img.ancho), min(max(x2, 0.0), img.ancho)
        cy1, cy2 = min(max(y1, 0.0), img.alto), min(max(y2, 0.0), img.alto)
        if (cx1, cy1, cx2, cy2) != (x1, y1, x2, y2):
            aud["r2_recortada"][etq] += 1
        if cx2 - cx1 < min_lado or cy2 - cy1 < min_lado:  # regla 3
            aud["r3_menor_1px"][etq] += 1
            continue
        clave = (clases.index(etq), cx1, cy1, cx2, cy2)
        if clave in vistas:  # regla 4
            aud["r4_duplicada"][etq] += 1
            continue
        vistas.add(clave)
        img.cajas.append(clave)
        aud["conservada"][etq] += 1
    return img


def lineas_yolo(img: Imagen) -> list[str]:
    out = []
    for c, x1, y1, x2, y2 in img.cajas:
        cx, cy = (x1 + x2) / 2 / img.ancho, (y1 + y2) / 2 / img.alto
        w, h = (x2 - x1) / img.ancho, (y2 - y1) / img.alto
        out.append(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return out


def matriz_presencia(imgs: list[Imagen], n_clases: int) -> np.ndarray:
    """Presencia por imagen de cada clase + columna 'fondo' (1 si no tiene cajas)."""
    y = np.zeros((len(imgs), n_clases + 1), dtype=int)
    for i, im in enumerate(imgs):
        for c, *_ in im.cajas:
            y[i, c] = 1
        y[i, n_clases] = int(not im.cajas)
    return y


def dividir(idx: np.ndarray, y: np.ndarray, tam_prueba: float | int, semilla: int) -> tuple[np.ndarray, np.ndarray]:
    """Una división estratificada iterativa de `idx` en (resto, prueba)."""
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=tam_prueba, random_state=semilla)
    a, b = next(msss.split(np.zeros(len(idx)), y[idx]))
    return idx[a], idx[b]


def particionar(imgs: list[Imagen], n_clases: int, cfg_part: dict) -> dict[str, list[str]]:
    y = matriz_presencia(imgs, n_clases)
    p, s = cfg_part["proporciones"], cfg_part["semilla"]
    todos = np.arange(len(imgs))
    train, resto = dividir(todos, y, p["val"] + p["test"], s)                 # 70 / 30
    val, test = dividir(resto, y, p["test"] / (p["val"] + p["test"]), s)      # 15 / 15
    return {k: sorted(imgs[i].nombre for i in v) for k, v in (("train", train), ("val", val), ("test", test))}


def preparar_salida(yolo_dir: Path) -> None:
    if yolo_dir.exists():
        if not (yolo_dir / MARCA_GENERADO).exists():
            raise RuntimeError(f"{yolo_dir} existe pero no fue generado por este script; bórralo a mano.")
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


def escribir_auditoria(ruta: Path, aud: dict[str, Counter], img_info: dict) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    cols = ["original", "r1_otra_clase", "r2_recortada", "r3_menor_1px", "r4_duplicada",
            "r5_imagen_corrupta", "conservada"]
    etiquetas = sorted(aud["original"], key=lambda e: -aud["original"][e])
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seccion", "concepto", *cols])
        for e in etiquetas:
            w.writerow(["cajas_por_etiqueta_original", e, *(aud[c][e] for c in cols)])
        w.writerow(["cajas_por_etiqueta_original", "TOTAL", *(sum(aud[c].values()) for c in cols)])
        for k, v in img_info.items():
            w.writerow(["imagenes", k, v])


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
    dir_imgs = raw_dir / "images"

    # Reglas 1-5
    xmls = sorted((raw_dir / "annotations" / "xmls").glob("*.xml"))
    imgs_raw = sorted(p.name for p in dir_imgs.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not xmls:
        raise FileNotFoundError(f"No hay XML en {raw_dir / 'annotations' / 'xmls'}")
    aud: dict[str, Counter] = {k: Counter() for k in (
        "original", "r1_otra_clase", "r2_recortada", "r3_menor_1px", "r4_duplicada", "r5_imagen_corrupta", "conservada")}
    todas = [procesar(x, dir_imgs, clases, cfg["limpieza"]["min_lado_px"], aud) for x in tqdm(xmls, desc="Limpieza")]
    sin_xml = sorted(set(imgs_raw) - {i.nombre for i in todas})
    if sin_xml:
        raise RuntimeError(f"{len(sin_xml)} imágenes sin XML (p. ej. {sin_xml[:3]}); decidir cómo tratarlas.")
    corruptas = [i.nombre for i in todas if i.corrupta]
    imgs = [i for i in todas if not i.corrupta]

    # Partición y copia
    splits = particionar(imgs, len(clases), cfg["particion"])
    conjuntos = [set(v) for v in splits.values()]
    assert sum(map(len, conjuntos)) == len(imgs) == len(set().union(*conjuntos)), "Splits solapados o incompletos"

    preparar_salida(yolo_dir)
    por_nombre = {i.nombre: i for i in imgs}
    for s, nombres in splits.items():
        for n in tqdm(nombres, desc=f"Copiando {s}"):
            shutil.copy2(dir_imgs / n, yolo_dir / "images" / s / n)
            lin = lineas_yolo(por_nombre[n])
            (yolo_dir / "labels" / s / Path(n).with_suffix(".txt")).write_text(
                "\n".join(lin) + ("\n" if lin else ""), encoding="utf-8")

    splits_dir.mkdir(parents=True, exist_ok=True)
    for s, nombres in splits.items():
        (splits_dir / f"{s}.txt").write_text("\n".join(nombres) + "\n", encoding="utf-8")

    # Calibración INT8: 500 imágenes de TRAIN, estratificadas por clase
    cc = cfg["calibracion"]
    train_imgs = [por_nombre[n] for n in splits["train"]]
    y_train = matriz_presencia(train_imgs, len(clases))
    _, idx_cal = dividir(np.arange(len(train_imgs)), y_train, cc["n_imagenes"], cc["semilla"])
    # La estratificación iterativa no garantiza el tamaño exacto; se ajusta quitando o agregando
    # imágenes de FONDO al azar (con semilla), lo que no altera la proporción de clases.
    rng = np.random.default_rng(cc["semilla"])
    es_fondo = y_train[:, -1] == 1
    ajuste_calib = len(idx_cal) - cc["n_imagenes"]
    if ajuste_calib > 0:
        quitar = rng.choice(idx_cal[es_fondo[idx_cal]], ajuste_calib, replace=False)
        idx_cal = np.setdiff1d(idx_cal, quitar)
    elif ajuste_calib < 0:
        libres = np.setdiff1d(np.flatnonzero(es_fondo), idx_cal)
        idx_cal = np.union1d(idx_cal, rng.choice(libres, -ajuste_calib, replace=False))
    calib = sorted(train_imgs[i].nombre for i in idx_cal)
    assert len(calib) == cc["n_imagenes"] and set(calib) <= set(splits["train"])
    (splits_dir / "calib.txt").write_text("\n".join(calib) + "\n", encoding="utf-8")
    (yolo_dir / "calib.txt").write_text(
        "\n".join(str(yolo_dir / "images" / "train" / n) for n in calib) + "\n", encoding="utf-8")

    nombres_cls = dict(enumerate(clases))
    escribir_yaml(sal["data_yaml"],
                  {"path": str(yolo_dir), "train": "images/train", "val": "images/val", "test": "images/test",
                   "names": nombres_cls},
                  "# Generado por src/data/prepare_dataset.py — no editar a mano.\n")
    # El exportador de ultralytics toma la calibración INT8 de la clave 'val'; ambas claves apuntan
    # a las 500 imágenes de TRAIN para que nunca se calibre con val ni test.
    escribir_yaml(sal["calib_yaml"],
                  {"path": str(yolo_dir), "train": "calib.txt", "val": "calib.txt", "names": nombres_cls},
                  "# Generado por src/data/prepare_dataset.py — no editar a mano.\n"
                  "# Calibración INT8: 500 imágenes del split TRAIN (nunca val ni test).\n")

    # Auditoría
    n_fondo = sum(1 for i in imgs if not i.cajas)
    n_yolo = sum(len(list((yolo_dir / "images" / s).iterdir())) for s in SPLITS)
    img_info = {
        "imagenes_raw": len(imgs_raw),
        "xml_raw": len(xmls),
        "imagenes_corruptas_eliminadas_r5": len(corruptas),
        "imagenes_en_data_yolo": n_yolo,
        "imagenes_fondo_r6": n_fondo,
        "pct_fondo": round(100 * n_fondo / len(imgs), 2),
        "imagenes_fondo_sin_anotaciones_en_xml": sum(1 for i in imgs if i.n_objetos_xml == 0),
        "imagenes_con_tamano_xml_distinto": sum(i.tam_xml_distinto for i in imgs),
        **{f"imagenes_{s}": len(v) for s, v in splits.items()},
        "imagenes_calibracion": len(calib),
        "calibracion_ajuste_fondo": -ajuste_calib,  # imágenes de fondo agregadas (+) o quitadas (−)
        **{f"calibracion_imagenes_con_{c}": int(y_train[idx_cal, k].sum()) for k, c in enumerate(clases)},
    }
    assert n_yolo == len(imgs_raw) - len(corruptas), "data/yolo no coincide con raw menos corruptas"
    escribir_auditoria(sal["auditoria_csv"], aud, img_info)

    resumen = {"clases": clases, "auditoria_cajas": {k: dict(v) for k, v in aud.items()},
               "auditoria_imagenes": img_info, "imagenes_corruptas": corruptas,
               "yolo_dir": str(yolo_dir), "entorno": info_entorno()}
    sal["resumen_json"].write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Auditoría ===")
    with open(sal["auditoria_csv"], encoding="utf-8") as f:
        for fila in csv.reader(f):
            print("  ".join(f"{x:>18}" for x in fila))
    print(f"\nSalida YOLO: {yolo_dir}")
    print(f"YAMLs: {sal['data_yaml'].relative_to(RAIZ)}, {sal['calib_yaml'].relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
