"""(Uso: PYTHONPATH=. python scripts/exploracion/explora_teacher_l.py)
Exploración: maestro YOLO11l con la config D2 (3 épocas completas) + costo de forward para KD."""
import json, math, time, csv
import torch
from ultralytics import YOLO
from src.train.common import argumentos_entrenamiento, ruta_pesos_preentrenados
from src.train.cwd import canales_detect
from src.utils.config import cargar_yaml
from src.utils.repro import fijar_semillas

cfg = cargar_yaml("configs/train.yaml")
fijar_semillas(0)
pesos = ruta_pesos_preentrenados("runs/pretrained/yolo11l.pt")
kw = argumentos_entrenamiento(cfg, "maestro", 0, "teacher_l_3ep", humo=False)
kw.update(epochs=3, project="runs/exploracion", exist_ok=True)

reg, t = [], {}
def ini(tr): torch.cuda.reset_peak_memory_stats(); t["i"] = time.perf_counter()
def ftr(tr): torch.cuda.synchronize(); t["t"] = time.perf_counter()
def fin(tr):
    torch.cuda.synchronize(); f = time.perf_counter()
    reg.append({"ep": tr.epoch + 1, "train": t["t"] - t["i"], "val": f - t["t"],
                "vram": torch.cuda.max_memory_reserved() / 2**30})
m = YOLO(pesos)
info = m.info(verbose=False)
m.add_callback("on_train_epoch_start", ini); m.add_callback("on_train_epoch_end", ftr); m.add_callback("on_fit_epoch_end", fin)
m.train(**kw)
reg = [r for r in reg if r["ep"] <= 3]
rows = [{k.strip(): float(v) for k, v in r.items()} for r in csv.DictReader(open(f"{m.trainer.save_dir}/results.csv"))]
nan = any(not math.isfinite(v) for r in rows for k, v in r.items() if "loss" in k)

red = YOLO(pesos).model.cuda().eval()
x = torch.rand(8, 3, 640, 640, device="cuda")
with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
    for _ in range(20): red(x)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(100): red(x)
    torch.cuda.synchronize()
fwd = (time.perf_counter() - t0) / 100
print("RESULTADO", json.dumps({"params_GFLOPs": info, "epocas": reg, "nan": nan,
      "mAP50_por_epoca": [r["metrics/mAP50(B)"] for r in rows], "fwd_ms_batch8": fwd * 1000,
      "canales_detect": canales_detect(red)}))
