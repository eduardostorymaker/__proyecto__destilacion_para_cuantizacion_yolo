"""Fase 1b: pruebas de tiempo (tiempo por época, VRAM máxima, pérdidas) y estimación del proyecto.

Uso (cada caso en su propio proceso para medir la VRAM máxima limpia):
    python -m src.train.timing_test --caso maestro_amp
    python -m src.train.timing_test --forward-maestro
    python -m src.train.timing_test --resumen
o todo junto con scripts/fase1b_tiempos.sh.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch

from src.train.common import argumentos_entrenamiento, callback_metadatos, ruta_pesos_preentrenados
from src.utils.config import RAIZ, cargar_yaml, resolver_ruta
from src.utils.repro import fijar_semillas

CFG_TRAIN = "configs/train.yaml"
CFG_TIMING = "configs/timing.yaml"


def correr_caso(caso_id: str) -> None:
    from ultralytics import YOLO

    cfg, ct = cargar_yaml(CFG_TRAIN), cargar_yaml(CFG_TIMING)
    caso = next(c for c in ct["casos"] if c["id"] == caso_id)
    rol = caso["rol"]
    seed = cfg[rol].get("seed", cfg[rol].get("semillas", [0])[0])
    fijar_semillas(seed)

    kwargs = argumentos_entrenamiento(cfg, rol, seed, caso_id, humo=False)
    kwargs.update(epochs=ct["epocas"], amp=caso["amp"], project=str(resolver_ruta(ct["project"])), exist_ok=True)

    registro: list[dict] = []
    t: dict[str, float] = {}

    def inicio(tr):
        torch.cuda.reset_peak_memory_stats()
        t["ini"] = time.perf_counter()

    def fin_train(tr):
        torch.cuda.synchronize()
        t["train"] = time.perf_counter()

    def fin_epoca(tr):
        torch.cuda.synchronize()
        fin = time.perf_counter()
        registro.append({
            "epoca": tr.epoch + 1,
            "s_train": t["train"] - t["ini"],
            "s_val": fin - t["train"],
            "s_total": fin - t["ini"],
            "vram_reservada_gb": torch.cuda.max_memory_reserved() / 2**30,
            "vram_asignada_gb": torch.cuda.max_memory_allocated() / 2**30,
        })

    model = YOLO(ruta_pesos_preentrenados(cfg[rol]["modelo"]))
    model.add_callback("on_pretrain_routine_start", callback_metadatos([CFG_TRAIN, CFG_TIMING], {"caso": caso_id}))
    model.add_callback("on_train_epoch_start", inicio)
    model.add_callback("on_train_epoch_end", fin_train)
    model.add_callback("on_fit_epoch_end", fin_epoca)
    model.train(**kwargs)

    salida = Path(model.trainer.save_dir) / "tiempos.json"
    salida.write_text(json.dumps({"caso": caso, "batch": kwargs["batch"], "registro": registro}, indent=2),
                      encoding="utf-8")
    print(f"Tiempos guardados en {salida}")


def forward_maestro() -> None:
    """Tiempo del paso hacia adelante del maestro (eval, sin gradientes, autocast) por batch."""
    from ultralytics import YOLO

    cfg, ct = cargar_yaml(CFG_TRAIN), cargar_yaml(CFG_TIMING)
    fm = ct["forward_maestro"]
    batch, imgsz = cfg["comun"]["batch"], cfg["comun"]["imgsz"]
    red = YOLO(ruta_pesos_preentrenados(fm["modelo"])).model.cuda().eval()
    x = torch.rand(batch, 3, imgsz, imgsz, device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for _ in range(fm["calentamiento"]):
            red(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(fm["iteraciones"]):
            red(x)
        torch.cuda.synchronize()
    s_batch = (time.perf_counter() - t0) / fm["iteraciones"]
    salida = resolver_ruta(ct["project"]) / "forward_maestro.json"
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(json.dumps({"batch": batch, "s_por_batch": s_batch}, indent=2), encoding="utf-8")
    print(f"Forward maestro: {s_batch * 1000:.1f} ms por batch de {batch}")


def perdidas_results_csv(ruta: Path) -> dict:
    """Pérdidas por época desde results.csv de Ultralytics: detecta NaN/inf y da la última época."""
    with open(ruta, encoding="utf-8") as f:
        filas = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(f)]
    claves = [k for k in filas[0] if "loss" in k]
    valores = [float(r[k]) for r in filas for k in claves]
    return {"nan": any(not math.isfinite(v) for v in valores),
            "perdidas_ultima_epoca": json.dumps({k: round(float(filas[-1][k]), 3) for k in claves})}


def resumen() -> None:
    cfg, ct = cargar_yaml(CFG_TRAIN), cargar_yaml(CFG_TIMING)
    base = resolver_ruta(ct["project"])
    filas, por_caso = [], {}
    for c in ct["casos"]:
        ruta = base / c["id"] / "tiempos.json"
        if not ruta.exists():  # caso interrumpido (p. ej. maestro_fp32): se registra a mano en LOG.md
            print(f"(sin datos para {c['id']}, se omite)")
            continue
        datos = json.loads(ruta.read_text(encoding="utf-8"))
        # Ultralytics repite on_fit_epoch_end en la validación final con best.pt: se descarta
        reg = [r for r in datos["registro"] if r["epoca"] <= ct["epocas"]]
        sin_primera = reg[1:] or reg
        media = lambda k: sum(r[k] for r in sin_primera) / len(sin_primera)  # noqa: E731
        fila = {
            "caso": c["id"], "amp": c["amp"], "batch": datos["batch"],
            "s_epoca_total": round(media("s_total"), 1), "s_epoca_train": round(media("s_train"), 1),
            "s_epoca_val": round(media("s_val"), 1),
            "vram_max_reservada_gb": round(max(r["vram_reservada_gb"] for r in reg), 2),
            "vram_max_asignada_gb": round(max(r["vram_asignada_gb"] for r in reg), 2),
            **perdidas_results_csv(base / c["id"] / "results.csv"),
        }
        filas.append(fila)
        por_caso[c["id"]] = fila

    fwd = json.loads((base / "forward_maestro.json").read_text(encoding="utf-8"))
    n_train = sum(1 for _ in open(resolver_ruta("data/splits/train.txt"), encoding="utf-8"))
    batches_epoca = math.ceil(n_train / cfg["comun"]["batch"])
    extra_kd = fwd["s_por_batch"] * batches_epoca  # s por época

    salida = RAIZ / "results" / "tables" / "timing.csv"
    salida.parent.mkdir(parents=True, exist_ok=True)
    with open(salida, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)

    print(f"{'caso':<14}{'amp':>6}{'s/época':>10}{'train':>8}{'val':>7}{'VRAM res.':>11}{'NaN':>6}")
    for r in filas:
        print(f"{r['caso']:<14}{str(r['amp']):>6}{r['s_epoca_total']:>10}{r['s_epoca_train']:>8}"
              f"{r['s_epoca_val']:>7}{r['vram_max_reservada_gb']:>10} G{str(r['nan']):>6}")
    print(f"\nForward maestro: {fwd['s_por_batch'] * 1000:.1f} ms/batch × {batches_epoca} batches"
          f" = +{extra_kd:.0f} s por época KD (sin contar la CWD ni los adaptadores)")

    e = ct["estimacion"]
    alumno = por_caso["alumno_amp"]["s_epoca_total"]
    for modo in (m for m in ("maestro_amp", "maestro_fp32") if m in por_caso):
        bloques = {
            "maestro": por_caso[modo]["s_epoca_total"] * e["maestro"]["epocas"] * e["maestro"]["corridas"],
            "alumnos_base": alumno * e["alumnos_base"]["epocas"] * e["alumnos_base"]["corridas"],
            "alumnos_kd": (alumno + extra_kd) * e["alumnos_kd"]["epocas"] * e["alumnos_kd"]["corridas"],
            "piloto_lambda": (alumno + extra_kd) * e["piloto_lambda"]["epocas"] * e["piloto_lambda"]["corridas"],
            "qat": alumno * e["factor_qat"] * e["qat"]["epocas"] * e["qat"]["corridas"],
        }
        total = sum(bloques.values())
        print(f"\nEstimación con maestro {modo} (alumnos con AMP):")
        for k, v in bloques.items():
            print(f"  {k:<14} {v / 3600:6.1f} h")
        print(f"  {'TOTAL':<14} {total / 3600:6.1f} h  (~{total / 86400:.1f} días de GPU)")
    print(f"\nGuardado en {salida.relative_to(RAIZ)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--caso")
    g.add_argument("--forward-maestro", action="store_true")
    g.add_argument("--resumen", action="store_true")
    a = ap.parse_args()
    if a.caso:
        correr_caso(a.caso)
    elif a.forward_maestro:
        forward_maestro()
    else:
        resumen()


if __name__ == "__main__":
    main()
