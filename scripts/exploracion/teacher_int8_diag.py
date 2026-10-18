"""Exploración: ¿por qué colapsan las confianzas del maestro YOLO11m en TensorRT INT8? (en VALIDACIÓN)

1. Exporta el maestro a OpenVINO FP32/INT8 (NNCF) y evalúa en val TRT FP32/INT8 y OV FP32/INT8:
   ¿el colapso es propio de TensorRT o del modelo?
2. Compara la precisión de las capas de la cabeza (model.23) en los engines INT8 del maestro y de un alumno.
3. Mide los rangos de activación de cada convolución con las imágenes de calibración, en el maestro y en un
   alumno: la calibración MinMax fija la escala con el máximo absoluto, así que un cociente
   máx/percentil 99.99 grande indica valores extremos que dejan poca resolución INT8 al resto.

Uso: PYTHONPATH=. python scripts/exploracion/teacher_int8_diag.py
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from src.quant.engine_layers import leer_engine_ultralytics, precision_de
from src.utils.config import cargar_yaml, resolver_ruta

SALIDA = resolver_ruta("runs/exploracion/teacher_int8")
TABLA = resolver_ruta("results/tables/exploracion_teacher_int8.csv")
MAESTRO = resolver_ruta("runs/teacher/teacher_s0/weights/best.pt")
ALUMNO = resolver_ruta("runs/students/student_base_s0/weights/best.pt")
CALIB = resolver_ruta("configs/calib.yaml")
DATA = resolver_ruta("configs/data.yaml")
N_IMG_ACT = 100  # imágenes de calibración para medir rangos de activación


def evaluar(ruta: Path, nombre: str, cpu: bool) -> dict:
    v = YOLO(str(ruta), task="detect").val(data=str(DATA), split="val", imgsz=640, batch=1, conf=0.001, iou=0.7,
                                          rect=False, save_json=True, plots=False, verbose=False,
                                          device="cpu" if cpu else 0, project=str(SALIDA / "val"), name=nombre,
                                          exist_ok=True)
    s = np.array([d["score"] for d in json.loads((Path(v.save_dir) / "predictions.json").read_text())])
    fila = {"variante": nombre, "mAP50": round(float(v.box.map50), 4), "mAP50_95": round(float(v.box.map), 4),
            "conf_max": round(float(s.max()), 4), "conf_p99": round(float(np.quantile(s, 0.99)), 4),
            "n_conf_ge_0.25": int((s >= 0.25).sum())}
    print("RES", fila, flush=True)
    return fila


def exportar_ov(prec: str) -> Path:
    d = SALIDA / f"teacher_s0_openvino_{prec}"
    d.mkdir(parents=True, exist_ok=True)
    copia = d / "best.pt"
    shutil.copy2(MAESTRO, copia)
    extra = {"int8": True, "data": str(CALIB)} if prec == "int8" else {}
    return Path(YOLO(str(copia)).export(format="openvino", imgsz=640, batch=1, dynamic=False, verbose=False, **extra))


def capas_cabeza(engine: Path) -> list[tuple[str, str]]:
    import tensorrt as trt

    rt = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    e = rt.deserialize_cuda_engine(leer_engine_ultralytics(engine))
    capas = json.loads(e.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))["Layers"]
    cabeza = max(int(c["Name"].split("/model.")[1].split("/")[0]) for c in capas if "/model." in c["Name"])
    return [(c["Name"][:90], precision_de((c.get("Outputs") or [{}])[0].get("Format/Datatype", "")))
            for c in capas if f"/model.{cabeza}/" in c["Name"] and c.get("LayerType") != "Reformat"]


def rangos_activacion(pt: Path, imagenes: list[Path]) -> list[dict]:
    modelo = YOLO(str(pt)).model.float().cuda().eval()
    stats: dict[str, list] = {}

    def gancho(nombre):
        def f(_, __, salida):
            x = salida.detach().float().abs().flatten()
            stats.setdefault(nombre, []).append((x.max().item(), torch.quantile(x[:: max(1, x.numel() // 200000)],
                                                                                0.9999).item()))
        return f

    for n, m in modelo.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            m.register_forward_hook(gancho(n))
    with torch.no_grad():
        for img in imagenes:
            im = np.asarray(Image.open(img).convert("RGB").resize((640, 640)), dtype=np.float32) / 255
            modelo(torch.from_numpy(im).permute(2, 0, 1)[None].cuda())
    filas = []
    for n, v in stats.items():
        mx = max(a for a, _ in v)
        p = float(np.mean([b for _, b in v]))
        filas.append({"capa": n, "max_abs": round(mx, 2), "p99_99": round(p, 2), "cociente": round(mx / max(p, 1e-6), 1)})
    return sorted(filas, key=lambda f: -f["cociente"])


def main() -> None:
    SALIDA.mkdir(parents=True, exist_ok=True)
    # 1. TensorRT (engines de la Fase 5) vs OpenVINO, en validación
    filas = [evaluar(resolver_ruta("runs/exports/teacher_s0_tensorrt_fp32/best.engine"), "teacher_trt_fp32", False),
             evaluar(resolver_ruta("runs/exports/teacher_s0_tensorrt_int8/best.engine"), "teacher_trt_int8", False),
             evaluar(exportar_ov("fp32"), "teacher_ov_fp32", True),
             evaluar(exportar_ov("int8"), "teacher_ov_int8", True),
             evaluar(resolver_ruta("runs/exports/student_base_s0_tensorrt_int8/best.engine"), "base_s0_trt_int8", False)]
    TABLA.parent.mkdir(parents=True, exist_ok=True)
    with open(TABLA, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)

    # 2. Precisión de las capas de la cabeza
    for nombre, eng in (("maestro", "teacher_s0"), ("alumno base_s0", "student_base_s0")):
        cab = capas_cabeza(resolver_ruta(f"runs/exports/{eng}_tensorrt_int8/best.engine"))
        print(f"\nCABEZA {nombre}: {len(cab)} capas; INT8={sum(p == 'INT8' for _, p in cab)} "
              f"FP16={sum(p == 'FP16' for _, p in cab)} FP32={sum(p == 'FP32' for _, p in cab)}")
        for n, p in cab:
            if "cv3" in n:  # rama de clasificación
                print(f"   {p:<5} {n}")

    # 3. Rangos de activación con imágenes de calibración
    calib = cargar_yaml(CALIB)
    lista = Path(calib["path"]) / calib["train"]
    imgs = [Path(l.strip()) for l in open(lista, encoding="utf-8") if l.strip()][:N_IMG_ACT]
    for nombre, pt in (("maestro", MAESTRO), ("alumno base_s0", ALUMNO)):
        r = rangos_activacion(pt, imgs)
        print(f"\nACTIVACIONES {nombre}: {len(r)} convoluciones; cociente máx/p99.99 mediano "
              f"{np.median([x['cociente'] for x in r]):.1f}; top 8:")
        for x in r[:8]:
            print(f"   {x['capa']:<35} max {x['max_abs']:>9}  p99.99 {x['p99_99']:>8}  cociente {x['cociente']}")
        cab = [x for x in r if x["capa"].startswith("model.23.cv3")]
        print(f"   rama de clasificación (model.23.cv3): cociente máx {max(x['cociente'] for x in cab)}; "
              f"max_abs máx {max(x['max_abs'] for x in cab)}")


if __name__ == "__main__":
    main()
