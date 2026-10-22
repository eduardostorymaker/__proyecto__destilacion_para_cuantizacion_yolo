"""Fase 8: eficiencia (D8). Latencia y FPS con batch 1 en GPU (TensorRT) y CPU (OpenVINO).

- Solo se miden los engines registrados en results/tables/engines.csv; antes de medir se verifica su hash.
- Por modelo: 3 repeticiones, cada una con 50 calentamientos + 500 medidas sobre imágenes de test precargadas
  en memoria (no se mide la lectura de disco). El predictor de Ultralytics separa preproceso, inferencia y
  postproceso (NMS); la latencia principal es la inferencia y FPS = 1000 / latencia media de inferencia.
- Sistema sin otra carga: antes de empezar se comprueba que no haya procesos en la GPU y que la carga de CPU sea
  baja; si no, se aborta. Durante cada repetición se muestrean los relojes de la GPU (SM, memoria, estado de
  energía, potencia, temperatura).

Uso (desde la raíz del proyecto, en WSL, con el equipo sin otra carga):
    python -m src.bench.benchmark
    python -m src.bench.benchmark --dataset japan
    python -m src.bench.benchmark --modelos student_base_s0     # subconjunto
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

from src.quant.engines_registry import info_sistema, verificar
from src.utils.config import cargar_yaml, resolver_ruta
from src.utils.dataset import agregar_argumento, ruta_config

CAMPOS_GPU = "clocks.sm,clocks.mem,pstate,power.draw,temperature.gpu,utilization.gpu"


def consulta_gpu() -> dict:
    out = subprocess.check_output(["nvidia-smi", f"--query-gpu={CAMPOS_GPU}", "--format=csv,noheader,nounits"],
                                  text=True).strip().splitlines()[0]
    v = [x.strip() for x in out.split(",")]
    num = lambda x: float(x) if x.replace(".", "", 1).isdigit() else float("nan")  # noqa: E731
    return {"sm_mhz": num(v[0]), "mem_mhz": num(v[1]), "pstate": v[2], "potencia_w": num(v[3]),
            "temp_c": num(v[4]), "util_pct": num(v[5])}


def procesos_gpu() -> list[str]:
    out = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                                  text=True).strip()
    return [l for l in out.splitlines() if l.strip()]


class MuestreoRelojes(threading.Thread):
    """Muestrea los relojes de la GPU en segundo plano mientras se mide."""

    def __init__(self, periodo: float):
        super().__init__(daemon=True)
        self.periodo, self.muestras, self._fin = periodo, [], threading.Event()

    def run(self):
        while not self._fin.is_set():
            try:
                self.muestras.append(consulta_gpu())
            except (OSError, subprocess.CalledProcessError):
                pass
            self._fin.wait(self.periodo)

    def detener(self) -> dict:
        self._fin.set()
        self.join()
        if not self.muestras:
            return {}
        sm = [m["sm_mhz"] for m in self.muestras]
        return {"sm_mhz_media": round(float(np.nanmean(sm)), 0), "sm_mhz_min": np.nanmin(sm),
                "mem_mhz_media": round(float(np.nanmean([m["mem_mhz"] for m in self.muestras])), 0),
                "pstates": "/".join(sorted({m["pstate"] for m in self.muestras})),
                "potencia_w_media": round(float(np.nanmean([m["potencia_w"] for m in self.muestras])), 1),
                "temp_c_max": np.nanmax([m["temp_c"] for m in self.muestras])}


def comprobar_sin_carga(cfg: dict) -> dict:
    import psutil

    procs = procesos_gpu()
    g = consulta_gpu()
    cpu = psutil.cpu_percent(interval=2.0)
    print(f"Estado inicial: procesos en GPU {len(procs)}, uso GPU {g['util_pct']} %, CPU {cpu} %, "
          f"SM {g['sm_mhz']} MHz, memoria {g['mem_mhz']} MHz, {g['pstate']}", flush=True)
    if procs or g["util_pct"] > cfg["max_util_gpu_pct"] or cpu > cfg["max_cpu_pct"]:
        raise RuntimeError(f"El sistema no está libre (procesos GPU: {procs}, uso GPU {g['util_pct']} %, "
                           f"CPU {cpu} %). D8 exige medir sin otra carga.")
    return {f"inicio_{k}": v for k, v in g.items()} | {"inicio_cpu_pct": cpu}


def medir(ruta: Path, formato: str, imagenes: list[np.ndarray], cfg: dict) -> tuple[list[dict], dict]:
    from ultralytics import YOLO

    modelo = YOLO(str(ruta), task="detect")
    dispositivo = "cpu" if formato == "openvino" else 0
    kw = dict(imgsz=cfg["imgsz"], batch=cfg["batch"], conf=cfg["conf"], iou=cfg["iou"], device=dispositivo,
              verbose=False, save=False)
    reps, n = [], len(imagenes)
    for r in range(cfg["repeticiones"]):
        for i in range(cfg["calentamiento"]):
            modelo.predict(imagenes[i % n], **kw)
        relojes = MuestreoRelojes(cfg["muestreo_relojes_s"])
        relojes.start()
        t = {"preproceso": [], "inferencia": [], "postproceso": []}
        for i in range(cfg["medidas"]):
            s = modelo.predict(imagenes[(cfg["calentamiento"] + i) % n], **kw)[0].speed
            t["preproceso"].append(s["preprocess"]); t["inferencia"].append(s["inference"])
            t["postproceso"].append(s["postprocess"])
        reloj = relojes.detener() if formato != "openvino" else {}
        fila = {"repeticion": r + 1}
        for k, xs in t.items():
            fila[f"{k}_ms_mediana"] = round(float(np.median(xs)), 4)
            fila[f"{k}_ms_media"] = round(float(np.mean(xs)), 4)
        fila["fps"] = round(1000 / fila["inferencia_ms_media"], 1)
        reps.append(fila | reloj)
    return reps, {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--modelos", nargs="*")
    agregar_argumento(ap)
    a = ap.parse_args()
    cfg = cargar_yaml(a.config or ruta_config("bench.yaml", a.dataset))
    sal = {k: resolver_ruta(v) for k, v in cfg["salidas"].items()}

    with open(resolver_ruta(cfg["engines"]), encoding="utf-8") as f:
        engines = [r for r in csv.DictReader(f) if r["ruta"].startswith(cfg["prefijo_rutas"])
                   and (not a.modelos or r["modelo"] in a.modelos)]
    engines.sort(key=lambda r: (r["formato"] != "tensorrt", r["modelo"], r["precision"]))  # GPU primero
    for r in engines:
        verificar(resolver_ruta(r["ruta"]))
    sistema = info_sistema()
    print(f"{len(engines)} engines registrados y verificados. Entorno: {sistema}", flush=True)

    d = cargar_yaml(cfg["data"])
    carpeta = Path(d["path"]) / d[cfg["split"]]
    archivos = sorted(carpeta.iterdir())[: cfg["calentamiento"] + cfg["medidas"]]
    imagenes = [np.asarray(Image.open(p).convert("RGB"))[:, :, ::-1].copy() for p in archivos]  # BGR, como cv2
    print(f"{len(imagenes)} imágenes de {cfg['split']} precargadas en memoria", flush=True)

    inicio = comprobar_sin_carga(cfg)
    filas_rep, resumen = [], []
    for r in engines:
        ruta = resolver_ruta(r["ruta"])
        t0 = time.perf_counter()
        reps, _ = medir(ruta, r["formato"], imagenes, cfg)
        base = {"modelo": r["modelo"], "formato": r["formato"], "precision": r["precision"], "MB": r["MB"]}
        filas_rep += [base | x for x in reps]
        inf = [x["inferencia_ms_mediana"] for x in reps]
        inf_media = [x["inferencia_ms_media"] for x in reps]
        fila = base | {
            "dispositivo": "CPU i7-12700KF" if r["formato"] == "openvino" else "GPU RTX 4060",
            "inferencia_ms_mediana": round(st.mean(inf), 4), "inferencia_ms_mediana_de": round(st.stdev(inf), 4),
            "inferencia_ms_media": round(st.mean(inf_media), 4), "inferencia_ms_media_de": round(st.stdev(inf_media), 4),
            "preproceso_ms_mediana": round(st.mean(x["preproceso_ms_mediana"] for x in reps), 4),
            "postproceso_ms_mediana": round(st.mean(x["postproceso_ms_mediana"] for x in reps), 4),
            "fps": round(1000 / st.mean(inf_media), 1),
            "sm_mhz_media": round(st.mean(x.get("sm_mhz_media", float("nan")) for x in reps), 0) if r["formato"] != "openvino" else "",
            "repeticiones": len(reps),
        } | sistema | inicio
        resumen.append(fila)
        print(f"{r['modelo']:<16} {r['formato']:<9} {r['precision']:<10} inferencia {fila['inferencia_ms_mediana']:.3f} ms "
              f"(d.e. entre reps {fila['inferencia_ms_mediana_de']:.3f})  FPS {fila['fps']:>7}  "
              f"pre {fila['preproceso_ms_mediana']:.2f}  post {fila['postproceso_ms_mediana']:.2f}  "
              f"{r['MB']} MB  ({time.perf_counter() - t0:.0f} s)", flush=True)
        escribir(sal["crudo"], filas_rep)
        escribir(sal["resumen"], resumen)
    print("FIN", flush=True)


def escribir(ruta: Path, filas: list[dict]) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    campos = list(dict.fromkeys(k for f in filas for k in f))
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(filas)


if __name__ == "__main__":
    main()
