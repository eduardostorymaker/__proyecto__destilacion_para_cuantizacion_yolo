"""Fase 9 / J9: tablas III, V, VI, VII y VIII del paper (paper/main.tex), figura de mAP vs. método de cuantización
y figura de activaciones del maestro frente al alumno (desde activaciones_maestro_alumno.csv).

Solo lee resultados ya calculados (no evalúa nada ni usa la GPU):
  - Tabla III (partición):        dataset_stats.csv de cada dataset (Fase 1 / J1).
  - Tabla V (FP32):               metrics.csv, TensorRT FP32 (mismo motor que ΔmAP, D6).
  - Tabla VI (cuantización, GPU): delta_map.csv (ΔmAP@0.5) y metrics.csv (ECE del modelo cuantizado).
  - Tabla VII (contrastes):       hypothesis_tests.csv (D7), p ajustado con Holm.
  - Tabla VIII (eficiencia):      efficiency.csv de India (D8), los 6 alumnos juntos.
Media ± d.e. (ddof = 1) de las 3 semillas; el maestro tiene una sola corrida y se reporta sin d.e.
Cada tabla se guarda en CSV, Markdown y como filas LaTeX listas para pegar entre \\midrule y \\bottomrule.

Uso: python -m src.report.paper_tables
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.utils.config import cargar_yaml, resolver_ruta  # noqa: E402
from src.utils.dataset import ruta_config  # noqa: E402

CFG = cargar_yaml("configs/report.yaml")
DEC = CFG["decimales"]
VACIO = "—"


# ---------------------------------------------------------------- lectura

def leer(ruta: str | Path) -> pd.DataFrame | None:
    p = resolver_ruta(ruta)
    return pd.read_csv(p) if p.exists() else None


def entradas(dataset: str) -> dict[str, pd.DataFrame | None]:
    ev = cargar_yaml(ruta_config("eval.yaml", dataset))["tablas"]
    st = cargar_yaml(ruta_config("stats.yaml", dataset))["salidas"]
    be = cargar_yaml(ruta_config("bench.yaml", dataset))["salidas"]
    dp = cargar_yaml(ruta_config("data_prep.yaml", dataset))["salidas"]
    return {"metricas": leer(ev["metricas"]), "deltas": leer(ev["deltas"]), "hipotesis": leer(st["hipotesis"]),
            "eficiencia": leer(be["resumen"]), "particion": leer(dp["stats_csv"])}


# ---------------------------------------------------------------- formato

def num(x: float, d: int, signo: bool = False) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return VACIO
    s = f"{x:+.{d}f}" if signo else f"{x:.{d}f}"
    return s.replace("-", "−")


def media_de(valores: pd.Series, d: int) -> str:
    v = valores.dropna()
    if v.empty:
        return VACIO
    if len(v) == 1:
        return num(float(v.iloc[0]), d)
    return f"{num(v.mean(), d)} ± {num(v.std(ddof=1), d)}"


def entero(n: float) -> str:
    return f"{int(n):,}".replace(",", " ")


def guardar(nombre: str, tabla: pd.DataFrame, nota: str) -> None:
    out = resolver_ruta(CFG["salidas"]["tablas"])
    out.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(out / f"{nombre}.csv", index=False, encoding="utf-8")
    cab = "| " + " | ".join(tabla.columns) + " |"
    sep = "|" + "|".join("---" for _ in tabla.columns) + "|"
    filas = ["| " + " | ".join(map(str, f)) + " |" for f in tabla.itertuples(index=False)]
    (out / f"{nombre}.md").write_text("\n".join([cab, sep, *filas, "", nota, ""]), encoding="utf-8")
    tex = []
    for f in tabla.itertuples(index=False):
        celdas = [str(c).replace("±", "$\\pm$").replace("−", "$-$").replace("%", "\\%") for c in f]
        celdas = [c.replace(" ", "\\,") if c.replace(" ", "").isdigit() else c for c in celdas]  # miles: 5\,396
        tex.append(" & ".join(celdas) + " \\\\")
    (out / f"{nombre}.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"\n== {nombre}\n" + "\n".join([cab, sep, *filas]))


# ---------------------------------------------------------------- tablas

def tabla_particion(datos: dict) -> pd.DataFrame:
    nombres = {"train": "entren.", "val": "valid.", "test": "prueba"}
    filas = []
    for ds, etq in CFG["datasets"].items():
        p = datos[ds]["particion"]
        bloques = [*nombres, "total"]
        for split in bloques:
            if p is None:
                filas.append([f"{etq}: {nombres.get(split, split)}", *[VACIO] * 6])
                continue
            q = p if split == "total" else p[p["split"] == split]
            img = q[q["clase"] == "TOTAL"]["imagenes"].sum()
            inst = [q[q["clase"] == c]["instancias"].sum() for c in ("D00", "D10", "D20", "D40")]
            fondo = q[q["clase"].str.startswith("fondo")]["imagenes"].sum()
            filas.append([f"{etq}: {nombres.get(split, split)}", entero(img), *map(entero, inst), entero(fondo)])
    return pd.DataFrame(filas, columns=["Partición", "Imágenes", "D00", "D10", "D20", "D40", "Fondo"])


def tabla_fp32(datos: dict) -> pd.DataFrame:
    filas = []
    for ds, etq in CFG["datasets"].items():
        m = datos[ds]["metricas"]
        for grupo, nombre in CFG["grupos"].items():
            fila = [f"{nombre} ({etq})"]
            q = None if m is None else m[(m["grupo"] == grupo) & (m["formato"] == CFG["motor_gpu"])
                                         & (m["precision"] == "fp32")]
            for col, d in (("mAP50", DEC["map"]), ("mAP50_95", DEC["map"]), ("AP50_D40", DEC["map"]),
                           ("P", DEC["map"]), ("ECE", DEC["ece"])):
                fila.append(VACIO if q is None or q.empty else media_de(q[col], d))
            filas.append(fila)
    return pd.DataFrame(filas, columns=["Modelo", "mAP@0.5", "mAP@0.5:0.95", "AP D40", "Prec.", "ECE"])


def tabla_cuantizacion(datos: dict) -> pd.DataFrame:
    filas = []
    for grupo, nombre in CFG["grupos"].items():
        for prec, metodo in CFG["metodos"].items():
            fila = [nombre, metodo]
            hay = False
            for ds, etq in CFG["datasets"].items():
                de, m = datos[ds]["deltas"], datos[ds]["metricas"]
                qd = None if de is None else de[(de["grupo"] == grupo) & (de["motor"] == CFG["motor_gpu"])
                                                & (de["precision"] == prec)]
                qm = None if m is None else m[(m["grupo"] == grupo) & (m["formato"] == CFG["motor_gpu"])
                                              & (m["precision"] == prec)]
                fila.append(VACIO if qd is None or qd.empty else media_de(qd["d_mAP50"], DEC["map"]))
                fila.append(VACIO if qm is None or qm.empty else media_de(qm["ECE"], DEC["ece"]))
                hay |= qd is not None and not qd.empty
            if hay or grupo != "maestro":  # el maestro solo se cuantizó con PTQ (D4)
                filas.append(fila)
    cols = ["Modelo", "Método"]
    for etq in CFG["datasets"].values():
        cols += [f"ΔmAP ({etq})", f"ECE ({etq})"]
    return pd.DataFrame(filas, columns=cols)


def tabla_contrastes(datos: dict) -> pd.DataFrame:
    hip = {ds: datos[ds]["hipotesis"] for ds in CFG["datasets"]}
    base = next((h for h in hip.values() if h is not None), None)
    if base is None:
        raise SystemExit("No hay hypothesis_tests.csv en ningún dataset (Fase 7 / J7).")
    filas = []
    for _, r in base.iterrows():
        confirmatoria = r["tipo"] == "confirmatoria"
        fila, decisiones = [r["pregunta"], r["hipotesis"], "C" if confirmatoria else "E"], []
        for ds in CFG["datasets"]:
            h = hip[ds]
            q = None if h is None else h[h["hipotesis"] == r["hipotesis"]]
            if q is None or q.empty:
                fila += [VACIO, VACIO]
                decisiones.append(None)
                continue
            x = q.iloc[0]
            fila.append(f"{num(x['D_obs'], DEC['map'], True)} [{num(x['IC_inf'], DEC['map'], True)}, "
                        f"{num(x['IC_sup'], DEC['map'], True)}]")
            fila.append(num(x["p_holm"], DEC["p"]) if confirmatoria else VACIO)
            # confirmatoria: rechaza H0 tras Holm; exploratoria: el IC no contiene el 0
            efecto = (x["decision"] == "rechaza H0") if confirmatoria else (x["IC_inf"] > 0 or x["IC_sup"] < 0)
            decisiones.append((efecto, math.copysign(1, x["D_obs"])))
        if any(d is None for d in decisiones):
            fila.append(VACIO)
        else:
            fila.append("sí" if all(d[0] for d in decisiones) and len({d[1] for d in decisiones}) == 1 else "no")
        filas.append(fila)
    cols = ["P", "Hip.", "Tipo"]
    for etq in CFG["datasets"].values():
        cols += [f"D [IC] {etq}", f"p {etq}"]
    return pd.DataFrame(filas, columns=[*cols, "Rep."])


def tabla_eficiencia(datos: dict) -> pd.DataFrame:
    e = datos[CFG["eficiencia_dataset"]]["eficiencia"]
    if e is None:
        raise SystemExit("No hay efficiency.csv (Fase 8).")
    e = e[e["modelo"].str.startswith("student_")]
    filas = []
    for motor, plataforma in CFG["plataformas"].items():
        for prec in ["fp32", *CFG["metodos"]]:
            q = e[(e["formato"] == motor) & (e["precision"] == prec)]
            if q.empty:
                continue
            etiqueta = "FP32" if prec == "fp32" else CFG["metodos"][prec].replace("PTQ FP16", "FP16")
            filas.append([plataforma, etiqueta, num(q["MB"].mean(), DEC["mb"]),
                          media_de(q["inferencia_ms_media"], DEC["ms"]),
                          f"{1000 / q['inferencia_ms_media'].mean():.0f}"])
    return pd.DataFrame(filas, columns=["Plataforma", "Precisión", "Tamaño (MB)", "Latencia (ms)", "FPS"])


# ---------------------------------------------------------------- figura

def figura(dataset: str, m: pd.DataFrame) -> None:
    """mAP de ambos alumnos por método de cuantización (media ± d.e. y cada semilla)."""
    pasos = [(CFG["motor_gpu"], "fp32", "FP32")] + [(CFG["motor_gpu"], p, e) for p, e in CFG["metodos"].items()]
    pasos += [(CFG["motor_cpu"], "fp32", "FP32"), (CFG["motor_cpu"], "int8", CFG["metodos"]["int8"])]
    colores = {"base": "#2a78d6", "kd": "#eb6834"}  # slots 1 y 2 de la paleta categórica (dataviz)
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, ejes = plt.subplots(2, 1, figsize=(3.5, 4.4), sharex=True)
    for eje, (col, titulo) in zip(ejes, (("mAP50", "mAP@0.5"), ("mAP50_95", "mAP@0.5:0.95"))):
        for k, grupo in enumerate(("base", "kd")):
            xs, medias, des = [], [], []
            for i, (motor, prec, _) in enumerate(pasos):
                q = m[(m["grupo"] == grupo) & (m["formato"] == motor) & (m["precision"] == prec)][col].dropna()
                if q.empty:
                    continue
                x = i + (k - 0.5) * 0.28
                eje.scatter([x] * len(q), q, s=9, facecolors="none", edgecolors=colores[grupo], linewidths=0.7,
                            alpha=0.6, zorder=2)
                xs.append(x), medias.append(q.mean()), des.append(q.std(ddof=1) if len(q) > 1 else 0)
            eje.errorbar(xs, medias, yerr=des, fmt="o", ms=4, color=colores[grupo], capsize=2, lw=1.2, zorder=3,
                         label=CFG["grupos"][grupo])
        eje.axvline(len(pasos) - 2.5, color="#b5b4ae", lw=0.8, ls="--")
        eje.set_ylabel(titulo)
        eje.grid(axis="y", color="#e4e3df", lw=0.6)
        eje.set_axisbelow(True)
    ejes[0].legend(frameon=False, loc="lower left", ncol=2)
    ymax = ejes[1].get_ylim()[1]
    for x, texto in ((len(pasos) / 2 - 1.75, "GPU · TensorRT"), (len(pasos) - 1.5, "CPU · OpenVINO")):
        ejes[0].text(x, ejes[0].get_ylim()[1], texto, ha="center", va="bottom", fontsize=7, color="#52514e")
    ejes[1].set_ylim(top=ymax)
    ejes[1].set_xticks(range(len(pasos)), [e for _, _, e in pasos], rotation=35, ha="right")
    fig.tight_layout()
    salida = resolver_ruta(CFG["salidas"]["figuras"][dataset])
    salida.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(salida.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura: {salida}.png/.pdf")


def figura_activaciones() -> None:
    """Cociente máx/p99.99 de las activaciones por convolución, maestro frente a alumno (India).

    Lee la tabla que calcula src.report.figuras_presentacion (figura 5); no vuelve a pasar imágenes por la red.
    """
    a = CFG["presentacion"]["activaciones"]
    d = leer(a["tabla"])
    if d is None:
        return
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, eje = plt.subplots(figsize=(3.5, 2.1))
    colores = {"alumno": "#2a78d6", "maestro": "#1baf7a"}  # slots 1 y 3 de la paleta (como en la presentación)
    etiquetas = {"alumno": "Alumno YOLO11n (base, s0)", "maestro": "Maestro YOLO11m"}
    for modelo in ("alumno", "maestro"):
        q = d[d["modelo"].str.startswith(modelo)].sort_values("orden")  # "maestro" y "alumno base_s0"
        x = q["orden"] / (len(q) - 1)
        if modelo == "maestro":
            cab = x[q["capa"].str.startswith(a["capa_cabeza"] + ".").values]
            eje.axvspan(cab.min(), cab.max(), color="#f1f0ec", lw=0, zorder=0, label="Cabeza de detección")
        eje.scatter(x, q["cociente"], s=5, color=colores[modelo], zorder=2,
                    label=f"{etiquetas[modelo]} (máx. {q['cociente'].max():.1f})")
    eje.set_yscale("log")
    eje.axhline(1, color="#b5b4ae", lw=0.6)
    eje.set_xlabel("Posición de la convolución (0 = entrada, 1 = salida)")
    eje.set_ylabel("máx |a| / p99.99")
    eje.legend(frameon=False, loc="upper left", fontsize=6.5)
    fig.tight_layout()
    salida = resolver_ruta(CFG["salidas"]["figura_activaciones"])
    salida.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(salida.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura: {salida}.png/.pdf")


# ---------------------------------------------------------------- main

def main() -> None:
    datos = {ds: entradas(ds) for ds in CFG["datasets"]}
    guardar("tabla_III_particion", tabla_particion(datos),
            "Columnas de clase: instancias; Fondo: imágenes sin daños.")
    guardar("tabla_V_fp32", tabla_fp32(datos),
            "TensorRT FP32, batch 1, test. Media ± d.e. de 3 semillas; el maestro tiene una corrida.")
    guardar("tabla_VI_cuantizacion", tabla_cuantizacion(datos),
            "ΔmAP en mAP@0.5 respecto al TensorRT FP32 del mismo modelo; ECE del modelo cuantizado. "
            "Media ± d.e. de 3 semillas. PTQ-MO: PTQ con ModelOpt (control sin ajuste fino de la QAT). "
            "ECE en blanco: ninguna detección con confianza ≥ 0.25 (ECE no definido).")
    guardar("tabla_VII_contrastes", tabla_contrastes(datos),
            "D > 0 favorece al alumno destilado. C: confirmatoria (p ajustado con Holm dentro de cada subconjunto); "
            "E: exploratoria (solo IC). IC 95 %, salvo H5 (IC 90 %, no inferioridad con δ = 0.010). "
            "Rep.: C significativa, o E con IC sin el 0, y del mismo signo en ambos subconjuntos.")
    guardar("tabla_VIII_eficiencia", tabla_eficiencia(datos),
            "Batch 1, 640 × 640, latencia de inferencia (media ± d.e. entre los 6 alumnos, cada uno con 3 "
            "repeticiones de 500 medidas); FPS = 1000 / latencia media; reloj SM de la GPU fijado en 2505 MHz.")
    for ds in CFG["datasets"]:
        if datos[ds]["metricas"] is not None:
            figura(ds, datos[ds]["metricas"])
    figura_activaciones()


if __name__ == "__main__":
    main()
