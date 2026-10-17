"""Inspector de engines TensorRT: cuenta en qué precisión quedó cada capa (D4).

La precisión de una capa se toma del tipo de dato de su primera salida (campo "Format/Datatype" del
inspector de TensorRT). Los engines INT8 de Ultralytics se construyen con ProfilingVerbosity.DETAILED, así
que el inspector entrega esa información por capa.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def leer_engine_ultralytics(ruta: Path) -> bytes:
    """Ultralytics antepone al engine 4 bytes de longitud + metadatos JSON; devuelve solo el engine."""
    with open(ruta, "rb") as f:
        n = int.from_bytes(f.read(4), "little")
        f.read(n)
        return f.read()


def precision_de(tipo: str) -> str:
    t = tipo.lower()
    if "int8" in t:
        return "INT8"
    if "half" in t or "fp16" in t:
        return "FP16"
    if "float" in t or "fp32" in t:
        return "FP32"
    return "OTRA"  # Int32, Bool, etc. (índices, formas)


def contar_capas(ruta: Path) -> dict:
    """Número y porcentaje de capas por precisión; aparte, las capas de cómputo (sin Reformat) y las convoluciones."""
    import tensorrt as trt

    runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    engine = runtime.deserialize_cuda_engine(leer_engine_ultralytics(ruta))
    info = json.loads(engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))
    capas = info["Layers"]
    if capas and not isinstance(capas[0], dict):
        raise RuntimeError(f"{ruta}: el inspector no entrega detalle por capa (¿engine sin ProfilingVerbosity.DETAILED?)")

    def prec(capa: dict) -> str:
        salidas = capa.get("Outputs") or [{}]
        return precision_de(salidas[0].get("Format/Datatype", ""))

    todas = Counter(prec(c) for c in capas)
    computo = Counter(prec(c) for c in capas if c.get("LayerType") != "Reformat")
    convs = [c for c in capas if c.get("ParameterType") == "Convolution"]
    pesos_conv = Counter(precision_de(c.get("Weights", {}).get("Type", "")) for c in convs)

    fila = {"capas_total": len(capas), "capas_computo": sum(computo.values()), "convoluciones": len(convs)}
    for p in ("INT8", "FP16", "FP32", "OTRA"):
        fila[f"capas_{p}"] = todas[p]
        fila[f"pct_capas_{p}"] = round(100 * todas[p] / max(len(capas), 1), 1)
        fila[f"computo_{p}"] = computo[p]
        fila[f"pct_computo_{p}"] = round(100 * computo[p] / max(sum(computo.values()), 1), 1)
    for p in ("INT8", "FP16", "FP32"):
        fila[f"conv_pesos_{p}"] = pesos_conv[p]
    fila["capas_FP32_nombres"] = "; ".join(c["Name"] for c in capas
                                           if c.get("LayerType") != "Reformat" and prec(c) == "FP32")
    return fila
