"""Figuras de apoyo para la presentación del primer avance (configs/report.yaml, sección `presentacion`).

  1. diagrama_flujo.png        Flujo de la metodología (datos → modelos → cuantización → evaluación).
  2. fiabilidad.png            Diagrama de fiabilidad (confianza vs. acierto) de base y KD en FP32, PTQ INT8 y QAT INT8,
                               con las mismas predicciones y parámetros del ECE (D6), 3 semillas juntas.
  3. map_vs_latencia.png       mAP@0.5 frente a la latencia de inferencia de cada método (GPU y CPU).
  4. distribucion_clases.png   Instancias por clase en India y Japón (con las de prueba).
  5. activaciones_maestro.png  Cociente máx/p99.99 de las activaciones por convolución, maestro vs. alumno base_s0
                               (causa del colapso del maestro con MinMax). Recalcula los rangos con 100 imágenes de
                               calibración y los guarda en results/tables/activaciones_maestro_alumno.csv.
Solo lee resultados ya calculados, salvo el pase hacia adelante de la figura 5 (GPU si está disponible).

Uso (en WSL): python -m src.report.figuras_presentacion [--solo 1 2 ...]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from src.eval.ece import cargar_gt, cargar_predicciones, emparejar  # noqa: E402
from src.utils.config import cargar_yaml, resolver_ruta  # noqa: E402
from src.utils.dataset import ruta_config  # noqa: E402

CFG = cargar_yaml("configs/report.yaml")
P = CFG["presentacion"]
SALIDA = resolver_ruta(P["salida"])
# Paleta categórica (dataviz): base = slot 1, KD = slot 2, maestro = slot 3; clases = slots 1–4 en orden fijo
AZUL, NARANJA, AQUA, AMARILLO = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
COLOR_GRUPO = {"base": AZUL, "kd": NARANJA, "maestro": AQUA}
TINTA, TINTA_2, REJILLA = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": TINTA_2, "axes.labelcolor": TINTA, "xtick.color": TINTA_2,
                     "ytick.color": TINTA_2})


def guardar(fig, nombre: str) -> None:
    SALIDA.mkdir(parents=True, exist_ok=True)
    fig.savefig(SALIDA / nombre, dpi=P["dpi"], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Figura: {SALIDA / nombre}")


def rejilla(eje, eje_y: bool = True) -> None:
    eje.grid(axis="y" if eje_y else "both", color=REJILLA, lw=0.7)
    eje.set_axisbelow(True)


# ---------------------------------------------------------------- 1. diagrama de flujo

def caja(eje, x, y, w, h, texto, color, negrita=False, tam=9.5):
    eje.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                 fc=color, ec=TINTA_2, lw=1.0))
    eje.text(x, y, texto, ha="center", va="center", fontsize=tam, color=TINTA,
             fontweight="bold" if negrita else "normal", linespacing=1.3)


def flecha(eje, x1, y1, x2, y2, estilo="-|>", ls="-"):
    eje.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=estilo, mutation_scale=12, lw=1.2,
                                  color=TINTA_2, ls=ls, shrinkA=2, shrinkB=2))


def diagrama_flujo() -> None:
    fig, eje = plt.subplots(figsize=(14, 7))
    eje.set_xlim(0, 14), eje.set_ylim(0, 7), eje.axis("off")
    claro = {"datos": "#e8eef7", "maestro": "#e3f4ec", "base": "#dce9f9", "kd": "#fbe3d9", "cuant": "#f3f2ee",
             "eval": "#fdf1d6"}
    # Columnas: títulos
    for x, t in ((1.3, "1. Datos"), (4.4, "2. Modelos (FP32)"), (8.1, "3. Cuantización"), (11.9, "4. Evaluación")):
        eje.text(x, 6.7, t, ha="center", fontsize=12, fontweight="bold", color=TINTA)
    # 1. Datos
    caja(eje, 1.3, 4.6, 2.3, 1.5, "RDD2022\nIndia · Japón\n4 clases", claro["datos"], True)
    caja(eje, 1.3, 2.6, 2.3, 1.6, "Limpieza + auditoría\nPartición 70/15/15\nestratificada (semilla 42)",
         claro["datos"], tam=8.5)
    caja(eje, 1.3, 0.9, 2.3, 0.9, "500 imágenes de\ncalibración (train)", claro["datos"], tam=8.5)
    flecha(eje, 1.3, 3.85, 1.3, 3.4), flecha(eje, 1.3, 1.8, 1.3, 1.35)
    # 2. Modelos
    caja(eje, 4.4, 5.5, 2.6, 0.95, "Maestro YOLO11m\n(congelado en la KD)", claro["maestro"], True)
    caja(eje, 4.4, 3.7, 2.6, 1.05, "Alumno BASE YOLO11n\nL = L_det · 3 semillas", claro["base"], True)
    caja(eje, 4.4, 1.9, 2.6, 1.25, "Alumno KD YOLO11n\nL = L_det + λ·L_CWD\nλ = 1.0 (piloto) · 3 semillas",
         claro["kd"], True)
    flecha(eje, 2.45, 2.6, 3.1, 3.7), flecha(eje, 2.45, 2.6, 3.1, 1.9), flecha(eje, 2.45, 4.6, 3.1, 5.5)
    eje.add_patch(FancyArrowPatch((5.7, 5.5), (5.7, 2.2), connectionstyle="arc3,rad=-0.45", arrowstyle="-|>",
                                  mutation_scale=12, lw=1.2, color=AQUA, ls="--"))
    eje.text(4.4, 4.62, "CWD en P3–P5\n(maestro → alumno KD)", fontsize=8.5, color=AQUA, ha="center",
             va="center")
    # 3. Cuantización (5 rutas desde el best.pt FP32)
    rutas = [("TensorRT FP16", "media precisión"), ("TensorRT INT8 · PTQ", "MinMax (uso práctico)"),
             ("OpenVINO INT8 · PTQ", "NNCF, CPU"), ("ModelOpt INT8 · PTQ-MO", "control sin ajuste fino"),
             ("ModelOpt INT8 · QAT", "30 épocas de ajuste fino")]
    ys = np.linspace(5.6, 1.0, len(rutas))
    for (t, s), y in zip(rutas, ys):
        caja(eje, 8.1, y, 3.0, 0.8, f"{t}\n{s}", claro["cuant"], tam=8.5)
    for y in ys:
        flecha(eje, 5.75, 2.8, 6.55, y)
    flecha(eje, 2.45, 0.9, 6.55, ys[-1] - 0.25, ls=":")
    eje.text(4.4, 0.55, "calibración", fontsize=8.5, color=TINTA_2, ha="center")
    eje.text(8.1, 6.2, "desde el mismo best.pt FP32 (también el maestro)", fontsize=8.5, color=TINTA_2,
             ha="center")
    # 4. Evaluación
    caja(eje, 11.9, 5.2, 3.0, 1.3, "Prueba (test)\nmAP@0.5 · mAP@0.5:0.95\nAP por clase · P · R · ECE",
         claro["eval"], True, tam=9)
    caja(eje, 11.9, 3.2, 3.0, 1.3, "Estadística\nbootstrap pareado (B = 1 000)\nHolm · P1–P5 → H1–H5",
         claro["eval"], tam=9)
    caja(eje, 11.9, 1.3, 3.0, 1.2, "Eficiencia\nlatencia · FPS · tamaño\nGPU y CPU", claro["eval"], tam=9)
    for y in ys:
        flecha(eje, 9.6, y, 10.4, 5.2)
    flecha(eje, 11.9, 4.55, 11.9, 3.85), flecha(eje, 11.9, 2.55, 11.9, 1.9)
    eje.text(7.0, 0.1, "Única diferencia entre alumnos: la pérdida de destilación. El flujo completo se repite en "
             "Japón con el mismo código.", ha="center", fontsize=9.5, color=TINTA_2, style="italic")
    guardar(fig, "diagrama_flujo.png")


# ---------------------------------------------------------------- 2. diagrama de fiabilidad

def fiabilidad() -> None:
    ev = cargar_yaml(ruta_config("eval.yaml"))
    d = cargar_yaml(ev["data"])
    raiz = Path(d["path"])
    gt = cargar_gt(raiz / d[ev["split"]], raiz / d[ev["split"]].replace("images", "labels"))
    m = pd.read_csv(resolver_ruta(ev["tablas"]["metricas"]))
    e = ev["ece"]
    bordes = np.linspace(e["rango"][0], e["rango"][1], e["intervalos"] + 1)
    nombres = {"fp32": "TensorRT FP32", **CFG["metodos"]}
    metodos = P["fiabilidad"]["metodos"]
    fig, ejes = plt.subplots(1, len(metodos), figsize=(4.2 * len(metodos), 4.4), sharey=True)
    for eje, prec in zip(ejes, metodos):
        eje.plot([0, 1], [0, 1], color="#b5b4ae", lw=1, ls="--", label="calibración perfecta")
        for grupo in ("base", "kd"):
            filas = m[(m["grupo"] == grupo) & (m["formato"] == CFG["motor_gpu"]) & (m["precision"] == prec)]
            confs, aciertos = [], []
            for ruta in filas["predicciones"]:
                c, a = emparejar(gt, cargar_predicciones(resolver_ruta(ruta)), e["conf_min"], e["iou_min"])
                confs.append(c), aciertos.append(a)
            c, a = np.concatenate(confs), np.concatenate(aciertos)
            idx = np.clip(np.digitize(c, bordes[1:-1]), 0, len(bordes) - 2)
            xs, ys = [], []
            for b in range(len(bordes) - 1):
                sel = idx == b
                if sel.sum() >= P["fiabilidad"]["min_detecciones_intervalo"]:
                    xs.append(c[sel].mean()), ys.append(a[sel].mean())
            etiqueta = f"{CFG['grupos'][grupo]} (ECE {filas['ECE'].mean():.3f})"
            eje.plot(xs, ys, "o-", color=COLOR_GRUPO[grupo], lw=2, ms=6, label=etiqueta,
                     markeredgecolor="white", markeredgewidth=1)
        eje.set_title(nombres[prec], fontsize=11, color=TINTA)
        eje.set_xlim(e["conf_min"] - 0.02, 1), eje.set_ylim(0, 1)
        eje.set_xlabel("Confianza media del intervalo")
        rejilla(eje, eje_y=False)
        eje.legend(frameon=False, fontsize=8.5, loc="upper left")
    ejes[0].set_ylabel("Fracción de detecciones correctas")
    fig.suptitle("Calibración de la confianza en India (3 semillas juntas; conf ≥ 0.25, IoU ≥ 0.5, 15 intervalos)",
                 fontsize=11, color=TINTA_2, y=1.02)
    fig.tight_layout()
    guardar(fig, "fiabilidad.png")


# ---------------------------------------------------------------- 3. mAP vs. latencia

def map_vs_latencia() -> None:
    m = pd.read_csv(resolver_ruta(cargar_yaml(ruta_config("eval.yaml"))["tablas"]["metricas"]))
    ef = pd.read_csv(resolver_ruta(cargar_yaml(ruta_config("bench.yaml"))["salidas"]["resumen"]))
    ef = ef[ef["modelo"].str.startswith("student_")].assign(grupo=lambda x: x["modelo"].str.split("_").str[1])
    pasos = {CFG["motor_gpu"]: ["fp32", "fp16", *[p for p in CFG["metodos"] if p != "fp16"]],
             CFG["motor_cpu"]: ["fp32", "int8"]}
    etq = {"fp32": "FP32", "fp16": "FP16", "int8": "PTQ INT8", "mopt_int8": "PTQ-MO", "qat_int8": "QAT"}
    # desplazamiento de cada etiqueta (puntos) para que no se pisen QAT, PTQ-MO y FP16, muy juntos en latencia
    desp = {"fp32": (8, -14), "fp16": (9, 4), "int8": (8, -14), "mopt_int8": (9, -24), "qat_int8": (-34, -4)}
    fig, ejes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for eje, (motor, precs), titulo in zip(ejes, pasos.items(), ("GPU · RTX 4060 (TensorRT)",
                                                                 "CPU · i7-12700KF (OpenVINO)")):
        for grupo in ("base", "kd"):
            for prec in precs:
                q = m[(m["grupo"] == grupo) & (m["formato"] == motor) & (m["precision"] == prec)]["mAP50"]
                lat = ef[(ef["grupo"] == grupo) & (ef["formato"] == motor) & (ef["precision"] == prec)]
                if q.empty or lat.empty:
                    continue
                x = lat["inferencia_ms_media"].mean()
                eje.errorbar(x, q.mean(), yerr=q.std(ddof=1), fmt="o", ms=8, color=COLOR_GRUPO[grupo], capsize=3,
                             lw=1.3, markeredgecolor="white", markeredgewidth=1.2,
                             label=CFG["grupos"][grupo] if prec == precs[0] else None)
                if grupo == "base":
                    eje.annotate(etq[prec], (x, q.mean()), xytext=desp[prec], textcoords="offset points",
                                 fontsize=8.5, color=TINTA_2)
        eje.set_title(titulo, fontsize=11, color=TINTA)
        eje.set_xlabel("Latencia media de inferencia (ms, batch 1)")
        rejilla(eje, eje_y=False)
    ejes[0].set_ylabel("mAP@0.5 en prueba (media ± d.e.)")
    ejes[0].legend(frameon=False, loc="lower right")
    fig.suptitle("Precisión frente a velocidad de los alumnos en India (más arriba y a la izquierda es mejor)",
                 fontsize=11, color=TINTA_2, y=1.02)
    fig.tight_layout()
    guardar(fig, "map_vs_latencia.png")


# ---------------------------------------------------------------- 4. distribución de clases

def distribucion_clases() -> None:
    clases = ["D00", "D10", "D20", "D40"]
    nombres = ["D00\ngrieta longitudinal", "D10\ngrieta transversal", "D20\npiel de cocodrilo", "D40\nbache"]
    colores = [AZUL, NARANJA, AQUA, AMARILLO]
    fig, ejes = plt.subplots(1, 2, figsize=(12, 4.6))
    for eje, ds, etq in zip(ejes, CFG["datasets"], ("India", "Japón")):
        st = pd.read_csv(resolver_ruta(cargar_yaml(ruta_config("data_prep.yaml", ds))["salidas"]["stats_csv"]))
        total = [st[st["clase"] == c]["instancias"].sum() for c in clases]
        prueba = [st[(st["clase"] == c) & (st["split"] == "test")]["instancias"].sum() for c in clases]
        imgs = st[st["clase"] == "TOTAL"]["imagenes"].sum()
        fondo = st[st["clase"].str.startswith("fondo")]["imagenes"].sum()
        barras = eje.bar(range(4), total, color=colores, width=0.62, edgecolor="white", linewidth=2)
        for b, t, p in zip(barras, total, prueba):
            eje.text(b.get_x() + b.get_width() / 2, t, f"{t:,}".replace(",", " ") + f"\n(prueba: {p})",
                     ha="center", va="bottom", fontsize=8.5, color=TINTA)
        eje.set_xticks(range(4), nombres, fontsize=8.5)
        eje.set_ylim(0, max(total) * 1.22)
        miles = f"{imgs:,}".replace(",", " ")
        eje.set_title(f"{etq}: {miles} imágenes, {100 * fondo / imgs:.1f} % de fondo", fontsize=11, color=TINTA)
        rejilla(eje)
    ejes[0].set_ylabel("Instancias tras la limpieza")
    fig.tight_layout()
    guardar(fig, "distribucion_clases.png")


# ---------------------------------------------------------------- 5. activaciones del maestro

def rangos_activacion(pt: Path, imagenes: list[Path], cuantil: float) -> list[dict]:
    import torch
    from PIL import Image
    from ultralytics import YOLO

    disp = "cuda" if torch.cuda.is_available() else "cpu"
    modelo = YOLO(str(pt)).model.float().to(disp).eval()
    stats: dict[str, list] = {}

    def gancho(nombre):
        def f(_, __, salida):
            x = salida.detach().float().abs().flatten()
            stats.setdefault(nombre, []).append((x.max().item(),
                                                 torch.quantile(x[:: max(1, x.numel() // 200000)], cuantil).item()))
        return f

    for n, mod in modelo.named_modules():
        if isinstance(mod, torch.nn.Conv2d):
            mod.register_forward_hook(gancho(n))
    with torch.no_grad():
        for img in imagenes:
            im = np.asarray(Image.open(img).convert("RGB").resize((640, 640)), dtype=np.float32) / 255
            modelo(torch.from_numpy(im).permute(2, 0, 1)[None].to(disp))
    filas = []
    for i, (n, v) in enumerate(stats.items()):  # orden de ejecución = orden de la red
        mx, p = max(a for a, _ in v), float(np.mean([b for _, b in v]))
        filas.append({"orden": i, "capa": n, "max_abs": round(mx, 3), "p99_99": round(p, 3),
                      "cociente": round(mx / max(p, 1e-6), 2)})
    return filas


def activaciones() -> None:
    a = P["activaciones"]
    calib = cargar_yaml(a["calib"])
    lista = Path(calib["path"]) / calib["train"]
    imgs = [Path(x.strip()) for x in open(lista, encoding="utf-8") if x.strip()][: a["n_imagenes"]]
    datos = {"maestro": rangos_activacion(resolver_ruta(a["maestro"]), imgs, a["cuantil"]),
             "alumno base_s0": rangos_activacion(resolver_ruta(a["alumno"]), imgs, a["cuantil"])}
    tabla = resolver_ruta(a["tabla"])
    with open(tabla, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["modelo", "orden", "capa", "max_abs", "p99_99", "cociente"])
        w.writeheader()
        for modelo, filas in datos.items():
            w.writerows({"modelo": modelo, **r} for r in filas)
    print(f"Tabla: {tabla}")

    fig, eje = plt.subplots(figsize=(12, 4.6))
    for modelo, color in (("alumno base_s0", AZUL), ("maestro", AQUA)):
        filas = datos[modelo]
        n = len(filas)
        x = np.array([r["orden"] for r in filas]) / (n - 1)  # posición relativa en la red (0 = entrada, 1 = salida)
        y = np.array([r["cociente"] for r in filas])
        eje.plot(x, y, "o", ms=3.5, color=color, alpha=0.85,
                 label=f"{'Maestro YOLO11m' if modelo == 'maestro' else 'Alumno YOLO11n (base, s0)'} "
                       f"(máx. {y.max():.1f})")
        cab = [r for r in filas if r["capa"].startswith(a["capa_cabeza"])]
        if modelo == "maestro":
            eje.axvspan(min(r["orden"] for r in cab) / (n - 1), 1.0, color="#f3f2ee", zorder=0,
                        label="cabeza de detección (zona sombreada)")
            top = max(filas, key=lambda r: r["cociente"])
            eje.annotate(f"{top['capa']}\nmáx {top['max_abs']:.0f} vs p99.99 {top['p99_99']:.1f}",
                         (top["orden"] / (n - 1), top["cociente"]), xytext=(-300, -30), textcoords="offset points",
                         fontsize=8.5, color=TINTA, arrowprops={"arrowstyle": "->", "color": TINTA_2})
    eje.axhline(1, color="#b5b4ae", lw=0.8)
    eje.set_yscale("log")
    eje.set_xlabel("Posición de la convolución en la red (0 = entrada, 1 = salida)")
    eje.set_ylabel("máx |activación| / percentil 99.99")
    eje.set_title("Valores extremos en las activaciones: con MinMax, la escala INT8 la fija el máximo",
                  fontsize=11, color=TINTA)
    rejilla(eje)
    eje.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    guardar(fig, "activaciones_maestro.png")


FIGURAS = {1: diagrama_flujo, 2: fiabilidad, 3: map_vs_latencia, 4: distribucion_clases, 5: activaciones}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo", type=int, nargs="*", choices=list(FIGURAS))
    for k in ap.parse_args().solo or FIGURAS:
        FIGURAS[k]()


if __name__ == "__main__":
    main()
