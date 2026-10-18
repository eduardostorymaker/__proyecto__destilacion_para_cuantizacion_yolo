"""Fase 5b: QAT INT8 con NVIDIA ModelOpt (D5), sobre la QAT integrada en Ultralytics 8.4.164.

Para cada alumno (desde su best.pt FP32):
  1. Calibración: se insertan los cuantizadores de ModelOpt (mtq.INT8_DEFAULT_CFG, algoritmo "max") y se
     calibran con las 500 imágenes de configs/calib.yaml, sin aumentación y con letterbox cuadrado.
  2. Control "ModelOpt PTQ": ese modelo solo calibrado se guarda (runs/qat/mopt_*/weights/best.pt) y se exporta
     a TensorRT INT8 sin ajuste fino.
  3. Ajuste fino QAT: train(quantize=8) de Ultralytics con SOLO la pérdida de detección (también para los
     alumnos KD), 30 épocas, lr0 0.001, sin warmup ni mosaico; el resto igual a configs/train.yaml. La
     calibración interna de Ultralytics se reemplaza por la del paso 1 (mismas imágenes y mismo método).
  4. Exportación del best.pt QAT: ONNX con nodos Q/DQ → TensorRT INT8 (rangos ya incluidos; sin recalibrar).

Las exportaciones se agregan a results/tables/exports.csv (precision mopt_int8 y qat_int8) y sus capas a
results/tables/engine_layers.csv. Reanudable: lo ya registrado se salta.

Uso (desde la raíz del proyecto, en WSL):
    python -m src.quant.qat                               # los 6 alumnos
    python -m src.quant.qat --alumnos student_base_s0     # uno o varios
    python -m src.quant.qat --alumnos student_base_s0 --humo   # prueba de humo (1 época, 5 % de train)
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import time
from pathlib import Path

import torch
import ultralytics.engine.trainer as ul_trainer
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_dataloader
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import qat_state, strip_qat

from src.quant.engine_layers import contar_capas
from src.quant.export_all import agregar_fila, leer_hechos, tamano_mb, verificar_calibracion
from src.train.common import argumentos_entrenamiento, callback_metadatos
from src.utils.config import cargar_yaml, resolver_ruta
from src.utils.repro import fijar_semillas, info_entorno


def calibrar(model: torch.nn.Module, cal: dict) -> torch.nn.Module:
    """Inserta los cuantizadores de ModelOpt y los calibra con las imágenes de calib.yaml (sin aumentación).

    Replica `ultralytics.utils.torch_utils.prepare_qat` (misma config de ModelOpt y mismas capas en float),
    cambiando solo los datos de calibración.
    """
    import modelopt.torch.quantization as mtq

    data = check_det_dataset(str(resolver_ruta(cal["data"])))
    hyp = get_cfg(overrides={"imgsz": cal["imgsz"]})
    dataset = YOLODataset(img_path=data["val"], imgsz=cal["imgsz"], batch_size=cal["batch"], augment=False,
                          hyp=hyp, rect=False, stride=32, data=data, task="detect")
    loader = build_dataloader(dataset, cal["batch"], workers=0, shuffle=False)
    device = next(model.parameters()).device

    def forward_loop(m):
        for lote in loader:
            m(lote["img"].to(device).float() / 255)

    LOGGER.info(f"QAT: calibrando ModelOpt (INT8_DEFAULT_CFG, algoritmo "
                f"'{mtq.INT8_DEFAULT_CFG.get('algorithm')}') con {len(dataset)} imágenes de {cal['data']}")
    entrenando = model.training
    model.eval()
    with torch.no_grad():
        model = mtq.quantize(model, mtq.INT8_DEFAULT_CFG, forward_loop)
        cabeza = f"model.{len(model.model) - 1}."
        mtq.disable_quantizer(model, lambda n: n.startswith(cabeza) and (".conv." not in n or ".dfl." in n))
    model.train(entrenando)
    return model


def guardar_control(pt: Path, destino: Path, cal: dict) -> Path:
    """Modelo 'ModelOpt PTQ': best.pt calibrado, sin ajuste fino, en formato de checkpoint de Ultralytics."""
    modelo, ck = load_checkpoint(str(pt))
    modelo = calibrar(modelo.float().cuda(), cal)
    estado = qat_state(modelo)
    copia = copy.deepcopy(modelo).half()
    strip_qat(copia)
    ruta = destino / "weights" / "best.pt"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: ck[k] for k in ("date", "version", "license", "docs", "train_args") if k in ck}
               | {"epoch": -1, "best_fitness": None, "model": copia, "ema": None, "updates": None,
                  "optimizer": None, "modelopt": estado}, ruta)
    return ruta


def exportar_qdq(pt: Path, destino: Path, ex: dict, calib: Path) -> tuple[Path, float]:
    """Exporta un modelo con cuantizadores de ModelOpt: ONNX con Q/DQ → TensorRT INT8 (sin recalibrar)."""
    destino.mkdir(parents=True, exist_ok=True)
    copia = destino / "best.pt"
    shutil.copy2(pt, copia)
    t0 = time.perf_counter()
    salida = Path(YOLO(str(copia)).export(format="engine", imgsz=ex["imgsz"], batch=ex["batch"],
                                          dynamic=ex["dynamic"], simplify=ex["simplify"], int8=True,
                                          data=str(calib), verbose=False))
    seg = time.perf_counter() - t0
    copia.unlink()
    return salida, seg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/qat.yaml")
    ap.add_argument("--alumnos", nargs="*")
    ap.add_argument("--humo", action="store_true")
    a = ap.parse_args()

    cfg, ctrain = cargar_yaml(a.config), cargar_yaml("configs/train.yaml")
    cal, ex = cfg["calibracion"], cfg["exportacion"]
    calib = resolver_ruta(cal["data"])
    print(f"Calibración verificada: {verificar_calibracion(calib)} imágenes de train, ninguna de val/test",
          flush=True)

    # La QAT de Ultralytics llama a prepare_qat(model, loader, preprocess); se sustituye por la calibración de D5
    ul_trainer.prepare_qat = lambda model, dataloader, preprocess, batches=8: calibrar(model, cal)

    proyecto = resolver_ruta(cfg["humo"]["project"] if a.humo else cfg["project"])
    sufijo = "_humo" if a.humo else ""
    t_exp, t_capas = (resolver_ruta(cfg["tablas"][k]) for k in ("exportaciones", "capas_engine"))
    hechos = set() if a.humo else leer_hechos(t_exp)
    salida_exp = resolver_ruta("runs/exports") if not a.humo else proyecto / "exports"

    for nombre, pt in cfg["alumnos"].items():
        if a.alumnos and nombre not in a.alumnos:
            continue
        pt = resolver_ruta(pt)
        grupo, seed = nombre.split("_")[1], int(nombre.rsplit("_s", 1)[1])
        fijar_semillas(seed)

        # 1-2. Control "ModelOpt PTQ" (sin ajuste fino)
        if (nombre, "tensorrt", "mopt_int8") not in hechos:
            print(f">> {nombre}: control ModelOpt PTQ", flush=True)
            ctrl = guardar_control(pt, proyecto / f"mopt_{grupo}_s{seed}{sufijo}", cal)
            eng, seg = exportar_qdq(ctrl, salida_exp / f"{nombre}_tensorrt_mopt_int8", ex, calib)
            if not a.humo:
                agregar_fila(t_exp, {"modelo": nombre, "formato": "tensorrt", "precision": "mopt_int8",
                                     "ruta": str(eng.relative_to(resolver_ruta("."))), "MB": round(tamano_mb(eng), 2),
                                     "segundos": round(seg, 1), "calibracion": f"ModelOpt max, {cal['data']}"})
                agregar_fila(t_capas, {"modelo": f"{nombre}_mopt", **contar_capas(eng)})
            print(f"   OK {eng}  {tamano_mb(eng):.1f} MB", flush=True)

        # 3-4. Ajuste fino QAT y exportación
        if (nombre, "tensorrt", "qat_int8") not in hechos:
            print(f">> {nombre}: ajuste fino QAT", flush=True)
            kw = argumentos_entrenamiento(ctrain, "alumno", seed, f"qat_{grupo}_s{seed}", a.humo)
            kw.update({k: v for k, v in cfg["ajuste"].items()}, project=str(proyecto),
                      name=f"qat_{grupo}_s{seed}{sufijo}", exist_ok=True)  # si se cortó, se rehace este alumno
            if a.humo:
                kw.update(epochs=cfg["humo"]["epochs"], fraction=cfg["humo"]["fraction"])
            modelo = YOLO(str(pt))
            modelo.add_callback("on_pretrain_routine_start",
                                callback_metadatos([a.config, "configs/train.yaml"],
                                                   {"rol": "qat", "origen": str(pt), "humo": a.humo}))
            t0 = time.perf_counter()
            modelo.train(**kw)
            horas = (time.perf_counter() - t0) / 3600
            best = Path(modelo.trainer.best)
            eng, seg = exportar_qdq(best, salida_exp / f"{nombre}_tensorrt_qat_int8", ex, calib)
            (eng.parent / "entorno.json").write_text(json.dumps(info_entorno() | {"origen": str(best),
                                                     "horas_ajuste": round(horas, 2)}, indent=2), encoding="utf-8")
            if not a.humo:
                agregar_fila(t_exp, {"modelo": nombre, "formato": "tensorrt", "precision": "qat_int8",
                                     "ruta": str(eng.relative_to(resolver_ruta("."))), "MB": round(tamano_mb(eng), 2),
                                     "segundos": round(seg, 1), "calibracion": f"ModelOpt max + QAT, {cal['data']}"})
                agregar_fila(t_capas, {"modelo": f"{nombre}_qat", **contar_capas(eng)})
            print(f"   OK {eng}  {tamano_mb(eng):.1f} MB  ajuste {horas:.2f} h", flush=True)
    print("FIN", flush=True)


if __name__ == "__main__":
    main()
