r"""Fase 8: eficiencia (D8). Latencia y FPS con batch 1 en GPU (TensorRT) y CPU (OpenVINO).

- Solo se miden los engines registrados en results/tables/engines.csv; antes de medir se verifica su hash.
- Por modelo: 3 repeticiones, cada una con 50 calentamientos + 500 medidas sobre imágenes de test precargadas
  en memoria (no se mide la lectura de disco). El predictor de Ultralytics separa preproceso, inferencia y
  postproceso (NMS); la latencia principal es la inferencia y FPS = 1000 / latencia media de inferencia.
- Sistema sin otra carga: antes de empezar se comprueba que no haya procesos en la GPU y que la carga de CPU sea
  baja; si no, se aborta. Durante cada repetición se muestrean los relojes de la GPU (SM, memoria, estado de
  energía, potencia, temperatura).
- Reloj de la GPU fijado (scripts/benchmark.ps1, nvidia-smi -lgc) para que el ahorro de energía del driver no
  cambie la latencia entre modelos; cada repetición de GPU se valida con la mediana del reloj SM (reloj fijado ±
  tolerancia) y, si no cumple, se repite (máx. max_reintentos).

Uso: en PowerShell como administrador (fija el reloj, mide y lo restablece):
    powershell -ExecutionPolicy Bypass -File scripts\benchmark.ps1 [--dataset japan] [--reanudar]
Directo en WSL (el reloj debe estar fijado antes):
    python -m src.bench.benchmark
    python -m src.bench.benchmark --dataset japan
    python -m src.bench.benchmark --modelos student_base_s0     # subconjunto
"""

from __future__ import annotations

import argparse
import csv
import os
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

_NVML = None


def consulta_gpu() -> dict:
    """Estado de la GPU con NVML dentro del proceso (no lanza nvidia-smi: así el muestreo casi no usa CPU)."""
    global _NVML
    import pynvml as nv

    if _NVML is None:
        nv.nvmlInit()
        _NVML = nv.nvmlDeviceGetHandleByIndex(0)
    h = _NVML
    try:
        potencia = nv.nvmlDeviceGetPowerUsage(h) / 1000
    except nv.NVMLError:  # en WSL la potencia no está disponible
        potencia = float("nan")
    return {"sm_mhz": float(nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM)),
            "mem_mhz": float(nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_MEM)),
            "pstate": f"P{nv.nvmlDeviceGetPerformanceState(h)}", "potencia_w": potencia,
            "temp_c": float(nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU)),
            "util_pct": float(nv.nvmlDeviceGetUtilizationRates(h).gpu),
            "sm_max_mhz": float(nv.nvmlDeviceGetMaxClockInfo(h, nv.NVML_CLOCK_SM))}


def procesos_gpu() -> list[str]:
    out = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                                  text=True).strip()
    propio = str(os.getpid())  # el engine anterior de este mismo proceso puede seguir en memoria
    return [l for l in out.splitlines() if l.strip() and l.split(",")[0].strip() != propio]


class MuestreoRelojes(threading.Thread):
    """Muestrea los relojes de la GPU en segundo plano mientras se mide."""

    def __init__(self, periodo: float):
        super().__init__(daemon=True)
        self.periodo, self.muestras, self._fin = periodo, [], threading.Event()

    def run(self):
        while not self._fin.is_set():
            self.muestras.append(consulta_gpu())
            self._fin.wait(self.periodo)

    def detener(self) -> dict:
        self._fin.set()
        self.join()
        if not self.muestras:
            return {}
        sm = [m["sm_mhz"] for m in self.muestras]
        return {"sm_mhz_mediana": float(np.nanmedian(sm)), "sm_mhz_media": round(float(np.nanmean(sm)), 0),
                "sm_mhz_min": np.nanmin(sm), "muestras_reloj": len(sm),
                "mem_mhz_media": round(float(np.nanmean([m["mem_mhz"] for m in self.muestras])), 0),
                "pstates": "/".join(sorted({m["pstate"] for m in self.muestras})),
                # en WSL NVML no reporta la potencia: queda vacía
                "potencia_w_media": round(float(np.nanmean(pot)), 1) if not np.isnan(pot := [m["potencia_w"] for m in self.muestras]).all() else "",
                "temp_c_max": np.nanmax([m["temp_c"] for m in self.muestras])}


def cpu_windows() -> float:
    """Carga de CPU de Windows (%); desde WSL, psutil solo ve la carga de la máquina virtual. Mediana de 5 lecturas
    en una sola llamada (cada lectura es instantánea y el arranque de powershell.exe produce un pico)."""
    orden = ("Start-Sleep -Milliseconds 500; 1..5 | ForEach-Object { "
             "(Get-CimInstance Win32_Processor | Measure-Object LoadPercentage -Average).Average; "
             "Start-Sleep -Milliseconds 500 }")
    try:
        out = subprocess.check_output(["powershell.exe", "-NoProfile", "-Command", orden], text=True, timeout=60)
        return float(np.median([float(x.replace(",", ".")) for x in out.split()]))
    except (OSError, subprocess.SubprocessError, ValueError):
        return float("nan")


def comprobar_sin_carga(cfg: dict, etiqueta: str) -> dict:
    """Aborta si hay otra carga. La carga de GPU es la carga efectiva (uso × reloj SM / reloj SM máximo),
    promediada en varias lecturas: en reposo (P8, ~210 MHz) el escritorio de Windows (WDDM) marca ~10 % de "uso"
    relativo a ese reloj bajo, que es < 1 % del cómputo disponible. Los procesos del escritorio de Windows no son
    visibles desde WSL."""
    import psutil

    procs = procesos_gpu()
    muestras = []
    for _ in range(cfg["muestras_carga"]):
        muestras.append(consulta_gpu())
        time.sleep(cfg["periodo_carga_s"])
    util = float(np.mean([m["util_pct"] * m["sm_mhz"] / m["sm_max_mhz"] for m in muestras]))
    cpu = psutil.cpu_percent(interval=1.0)
    cpu_win = cpu_windows()
    g = muestras[-1]
    print(f"[{etiqueta}] procesos en GPU {len(procs)}, carga GPU efectiva {util:.1f} %, CPU WSL {cpu} %, "
          f"CPU Windows {cpu_win} %, SM {g['sm_mhz']} MHz, memoria {g['mem_mhz']} MHz, {g['pstate']}", flush=True)
    if (procs or util > cfg["max_util_gpu_pct"] or cpu > cfg["max_cpu_pct"]
            or cpu_win > cfg["max_cpu_windows_pct"]):
        raise RuntimeError(f"El sistema no está libre (procesos GPU: {procs}, carga GPU efectiva {util:.1f} %, CPU WSL {cpu} %, "
                           f"CPU Windows {cpu_win} %). D8 exige medir sin otra carga.")
    return {"previo_carga_gpu_efectiva_pct": round(util, 1), "previo_cpu_wsl_pct": cpu, "previo_cpu_windows_pct": cpu_win,
            "previo_sm_mhz": g["sm_mhz"], "previo_mem_mhz": g["mem_mhz"], "previo_pstate": g["pstate"]}


def medir(ruta: Path, formato: str, imagenes: list[np.ndarray], cfg: dict) -> tuple[list[dict], dict]:
    from ultralytics import YOLO

    modelo = YOLO(str(ruta), task="detect")
    dispositivo = "cpu" if formato == "openvino" else 0
    kw = dict(imgsz=cfg["imgsz"], batch=cfg["batch"], conf=cfg["conf"], iou=cfg["iou"], device=dispositivo,
              verbose=False, save=False)
    reps, n = [], len(imagenes)
    for r in range(cfg["repeticiones"]):
        for intento in range(cfg["max_reintentos"] + 1):
            fila, valida = repeticion(modelo, formato, imagenes, kw, cfg)
            if valida:
                break
            print(f"  repetición {r + 1}: reloj SM mediano {fila['sm_mhz_mediana']:.0f} MHz fuera de "
                  f"{cfg['reloj_fijo_sm_mhz']} ± {cfg['tolerancia_reloj_mhz']} MHz; se repite", flush=True)
        else:
            raise RuntimeError(f"El reloj de la GPU no se mantuvo en {cfg['reloj_fijo_sm_mhz']} MHz tras "
                               f"{cfg['max_reintentos']} reintentos. ¿Se fijó con scripts/benchmark.ps1?")
        reps.append({"repeticion": r + 1, "reintentos": intento} | fila)
    return reps, {}


def repeticion(modelo, formato: str, imagenes: list[np.ndarray], kw: dict, cfg: dict) -> tuple[dict, bool]:
    """Una repetición completa (calentamiento + medidas). En GPU es válida solo si la mediana del reloj SM
    durante las medidas está en el reloj fijado ± tolerancia (el driver puede bajar el reloj con batch 1)."""
    n = len(imagenes)
    for i in range(cfg["calentamiento"]):
        modelo.predict(imagenes[i % n], **kw)
    # Relojes de la GPU solo para los engines de GPU. (Antes el hilo se iniciaba también en CPU y nunca se
    # detenía: los hilos acumulados cargaban la CPU y degradaban la latencia de OpenVINO.)
    relojes = MuestreoRelojes(cfg["muestreo_relojes_s"]) if formato != "openvino" else None
    if relojes:
        relojes.start()
    t = {"preproceso": [], "inferencia": [], "postproceso": []}
    try:
        for i in range(cfg["medidas"]):
            s = modelo.predict(imagenes[(cfg["calentamiento"] + i) % n], **kw)[0].speed
            t["preproceso"].append(s["preprocess"]); t["inferencia"].append(s["inference"])
            t["postproceso"].append(s["postprocess"])
    finally:
        reloj = relojes.detener() if relojes else {}
    fila = {}
    for k, xs in t.items():
        fila[f"{k}_ms_mediana"] = round(float(np.median(xs)), 4)
        fila[f"{k}_ms_media"] = round(float(np.mean(xs)), 4)
    fila["fps"] = round(1000 / fila["inferencia_ms_media"], 1)
    valida = (not relojes or
              abs(reloj.get("sm_mhz_mediana", 0) - cfg["reloj_fijo_sm_mhz"]) <= cfg["tolerancia_reloj_mhz"])
    return fila | reloj, valida


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--modelos", nargs="*")
    ap.add_argument("--reanudar", action="store_true", help="no repite los modelos que ya están en efficiency.csv")
    ap.add_argument("--repeticiones", type=int, help="solo para la prueba de humo (D8 fija 3)")
    ap.add_argument("--medidas", type=int, help="solo para la prueba de humo (D8 fija 500)")
    agregar_argumento(ap)
    a = ap.parse_args()
    cfg = cargar_yaml(a.config or ruta_config("bench.yaml", a.dataset))
    sal = {k: resolver_ruta(v) for k, v in cfg["salidas"].items()}
    if a.repeticiones or a.medidas:  # prueba de humo: no escribe en results/
        cfg["repeticiones"] = a.repeticiones or cfg["repeticiones"]
        cfg["medidas"] = a.medidas or cfg["medidas"]
        sal = {k: resolver_ruta(f"runs/smoke/bench_{k}.csv") for k in sal}

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

    filas_rep, resumen = [], []
    if a.reanudar and sal["resumen"].exists():  # conserva los modelos ya medidos (p. ej., tras un aborto por carga)
        with open(sal["resumen"], encoding="utf-8") as f:
            resumen = list(csv.DictReader(f))
        with open(sal["crudo"], encoding="utf-8") as f:
            filas_rep = list(csv.DictReader(f))
        hechos = {(x["modelo"], x["formato"], x["precision"]) for x in resumen}
        engines = [r for r in engines if (r["modelo"], r["formato"], r["precision"]) not in hechos]
        print(f"Reanudando: {len(hechos)} ya medidos, faltan {len(engines)}", flush=True)
    for r in engines:
        ruta = resolver_ruta(r["ruta"])
        inicio = comprobar_sin_carga(cfg, f"{r['modelo']} {r['formato']} {r['precision']}")
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
            "sm_mhz_media": round(st.mean(float(x.get("sm_mhz_media", "nan")) for x in reps), 0) if r["formato"] != "openvino" else "",
            "reloj_fijo_sm_mhz": cfg["reloj_fijo_sm_mhz"] if r["formato"] != "openvino" else "",
            "repeticiones": len(reps), "reintentos": sum(x["reintentos"] for x in reps),
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
