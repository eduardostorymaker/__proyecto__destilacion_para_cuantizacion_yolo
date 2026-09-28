"""Verifica el entorno del proyecto: Python, CUDA/PyTorch, TensorRT, OpenVINO y ultralytics.

Uso:
    python scripts/check_env.py

Escribe un resumen en results/env_check.json y termina con código 1 si falta algo crítico.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.utils.repro import info_entorno  # noqa: E402

PY_ESPERADO = (3, 13)
TRT_MAYOR_ESPERADO = 10  # ver comentario en requirements.txt

resultados: list[tuple[str, bool, str, bool]] = []  # (nombre, ok, detalle, crítico)


def chequeo(nombre: str, critico: bool = True):
    """Decorador: ejecuta la función, captura excepciones y registra el resultado."""

    def envoltura(fn):
        try:
            ok, detalle = fn()
        except ModuleNotFoundError as e:
            # Paquete no instalado: basta con el mensaje, sin traza
            ok, detalle = False, f"no instalado ({e.name})"
        except Exception as e:  # noqa: BLE001
            ok, detalle = False, f"{type(e).__name__}: {e}"
            traceback.print_exc()
        resultados.append((nombre, ok, detalle, critico))
        return fn

    return envoltura


@chequeo("Python 3.13")
def _python():
    v = sys.version_info
    return (v.major, v.minor) == PY_ESPERADO, sys.version.split()[0]


@chequeo("PyTorch + CUDA")
def _torch():
    import torch

    if not torch.cuda.is_available():
        return False, f"torch {torch.__version__}, CUDA NO disponible"
    # Operación real en GPU para confirmar que el driver funciona desde WSL
    x = torch.randn(512, 512, device="cuda")
    _ = (x @ x).sum().item()
    nombre = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 2**30
    return True, f"torch {torch.__version__}, CUDA {torch.version.cuda}, {nombre} ({mem:.1f} GB)"


@chequeo("TensorRT")
def _tensorrt():
    import tensorrt as trt

    mayor = int(trt.__version__.split(".")[0])
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)  # falla si TensorRT no ve la GPU
    detalle = f"TensorRT {trt.__version__}"
    for flag in ("platform_has_fast_fp16", "platform_has_fast_int8"):
        if hasattr(builder, flag):
            detalle += f", {flag.split('_')[-1].upper()}={getattr(builder, flag)}"
    if mayor != TRT_MAYOR_ESPERADO:
        detalle += f" (se esperaba la versión mayor {TRT_MAYOR_ESPERADO})"
    return mayor == TRT_MAYOR_ESPERADO, detalle


@chequeo("OpenVINO")
def _openvino():
    import openvino as ov

    core = ov.Core()
    dispositivos = core.available_devices
    cpu = core.get_property("CPU", "FULL_DEVICE_NAME") if "CPU" in dispositivos else "sin CPU"
    return "CPU" in dispositivos, f"OpenVINO {ov.__version__}, dispositivos={dispositivos}, CPU={cpu}"


@chequeo("NNCF (INT8 OpenVINO)")
def _nncf():
    import nncf

    return True, f"nncf {nncf.__version__}"


@chequeo("ultralytics")
def _ultralytics():
    import ultralytics

    return True, f"ultralytics {ultralytics.__version__}"


@chequeo("ONNX")
def _onnx():
    import onnx
    import onnxslim

    return True, f"onnx {onnx.__version__}, onnxslim {onnxslim.__version__}"


@chequeo("pycocotools / scipy / pandas")
def _cientificas():
    import pandas
    import pycocotools  # noqa: F401
    import scipy

    return True, f"scipy {scipy.__version__}, pandas {pandas.__version__}"


@chequeo("RAM visible (>= 20 GB recomendado)", critico=False)
def _ram():
    import psutil

    gb = psutil.virtual_memory().total / 2**30
    return gb >= 20, f"{gb:.1f} GB (en WSL se ajusta con memory= en %UserProfile%\\.wslconfig)"


@chequeo("Proyecto fuera de /mnt (E/S rápida)", critico=False)
def _ruta():
    en_mnt = str(RAIZ).startswith("/mnt/")
    return not en_mnt, f"{RAIZ}" + (" -> lectura de datos lenta en WSL" if en_mnt else "")


def main() -> int:
    ancho = max(len(r[0]) for r in resultados)
    print("\n=== Verificación del entorno ===")
    for nombre, ok, detalle, critico in resultados:
        marca = "OK  " if ok else ("FALLA" if critico else "AVISO")
        print(f"[{marca:5}] {nombre:<{ancho}}  {detalle}")

    fallas = [r[0] for r in resultados if not r[1] and r[3]]
    salida = RAIZ / "results" / "env_check.json"
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(
        json.dumps(
            {
                "entorno": info_entorno(),
                "chequeos": [
                    {"nombre": n, "ok": ok, "detalle": d, "critico": c} for n, ok, d, c in resultados
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nResumen guardado en {salida.relative_to(RAIZ)}")
    if fallas:
        print(f"Fallas críticas: {', '.join(fallas)}")
        return 1
    print("Entorno listo.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
