"""Figura del pipeline para el paper (configs/report.yaml, sección `pipeline`).

Diagrama de los seis bloques del estudio, a lo ancho de la página (figure* de IEEE):
  1 Datos → 2 Entrenamiento → 3 Cuantización → 4 Evaluación → 5 Estadística, y 3 → 6 Eficiencia.
Las flechas punteadas de arriba son los conjuntos que el bloque 1 entrega directamente a 3 (calibración) y a 4
(prueba). Los textos están escritos aquí: conteos de diseño (iguales en India y Japón) y tamaños tomados de
data_audit.csv y dataset_stats.csv de cada subconjunto; no se muestra ningún resultado. Si cambia el diseño,
actualizar los textos.

Salidas: results/figures/pipeline.pdf (vectorial, para LaTeX) y results/figures/pipeline.png.

Uso: python -m src.report.figura_pipeline
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from src.utils.config import cargar_yaml, resolver_ruta  # noqa: E402

CFG = cargar_yaml("configs/report.yaml")["pipeline"]
TINTA, TINTA_2, FONDO = "#1e242b", "#52514e", "#f3f4f6"
# Un color por bloque (mismo orden que la presentación); el cuerpo de cada bloque va en gris claro
COLORES = ["#38434f", "#2a78d6", "#eb6834", "#1baf7a", "#c4453a", "#c99700"]
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})
TAM, TAM_T = 6.3, 7.4   # cuerpo y encabezados, en puntos (tamaño final en el paper)


def rect(eje, x, y, w, h, fc, ec=None, lw=0.0, r=0.04, ls="-"):
    eje.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc,
                                 ec=ec or fc, lw=lw, ls=ls))


def flecha(eje, x1, y1, x2, y2, color=TINTA_2, ls="-", lw=0.9):
    eje.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=7, lw=lw, color=color,
                                  ls=ls, shrinkA=0, shrinkB=0))


def texto(eje, x, y, t, tam=TAM, color=TINTA, negrita=False, ha="left", va="top", **kw):
    eje.text(x, y, t, fontsize=tam, color=color, fontweight="bold" if negrita else "normal", ha=ha, va=va,
             linespacing=1.25, **kw)


def bloque(eje, x, y, w, h, i, titulo, pie):
    """Bloque con encabezado de color y pie con lo que entrega al siguiente."""
    rect(eje, x, y, w, h, FONDO)
    rect(eje, x, y + h - 0.24, w, 0.24, COLORES[i])
    texto(eje, x + 0.06, y + h - 0.12, f"{i + 1}  {titulo}", TAM_T, "white", True, va="center")
    rect(eje, x + 0.04, y + 0.04, w - 0.08, 0.22, "white", COLORES[i], 0.7, r=0.03)
    texto(eje, x + w / 2, y + 0.15, pie, TAM, TINTA, True, ha="center", va="center")


def nodo(eje, x, y, w, h, t, i, relleno=False):
    rect(eje, x, y, w, h, COLORES[i] if relleno else "white", COLORES[i], 0.7, r=0.03)
    texto(eje, x + w / 2, y + h / 2, t, TAM, "white" if relleno else TINTA, ha="center", va="center")


def main() -> None:
    W, H = CFG["ancho_pulgadas"], CFG["alto_pulgadas"]
    fig = plt.figure(figsize=(W, H))
    eje = fig.add_axes((0, 0, 1, 1))
    eje.set_xlim(0, W), eje.set_ylim(0, H), eje.axis("off")

    anchos, g = [1.08, 1.42, 1.56, 1.13, 1.13], 0.2
    x0 = (W - sum(anchos) - g * (len(anchos) - 1)) / 2
    xs = [x0 + sum(anchos[:i]) + g * i for i in range(len(anchos))]
    yb, hb = 0.86, 2.0                     # fila principal
    top = yb + hb

    # 1 Datos
    x, w = xs[0], anchos[0]
    bloque(eje, x, yb, w, hb, 0, "Datos", "→ train/val a 2")
    texto(eje, x + 0.07, top - 0.32, "RDD2022 (IN · JP)\n7 706 · 10 506 img.\n\nLimpieza: 6 reglas,\n"
          "0 imágenes quitadas\n4 clases: D00, D10,\nD20, D40\n\nPartición 70/15/15\nestratificada (s. 42)")

    # 2 Entrenamiento
    x, w = xs[1], anchos[1]
    bloque(eje, x, yb, w, hb, 1, "Entrenamiento", "→ 7 best.pt FP32")
    nodo(eje, x + 0.08, top - 0.6, w - 0.16, 0.3, "Maestro YOLO11m\n(congelado en la KD)", 1, True)
    wn = (w - 0.22) / 2
    nodo(eje, x + 0.08, yb + 0.6, wn, 0.56, "Base ×3\nYOLO11n\nL_det", 1)
    nodo(eje, x + 0.14 + wn, yb + 0.6, wn, 0.56, "KD ×3\nYOLO11n\nL_det +\nλ·L_CWD", 1)
    xk = x + 0.14 + wn + wn / 2
    flecha(eje, xk, top - 0.6, xk, yb + 1.16, COLORES[2], lw=1.0)
    texto(eje, xk - 0.05, (top - 0.6 + yb + 1.16) / 2, "CWD en P3–P5", TAM, COLORES[2], True, ha="right",
          va="center")
    texto(eje, x + 0.08, yb + 0.53, "COCO · 300 épocas · semillas\n0–2 · λ = 1 (piloto en val)", va="top")

    # 3 Cuantización
    x, w = xs[2], anchos[2]
    bloque(eje, x, yb, w, hb, 2, "Cuantización", "→ 45 modelos + SHA-256")
    texto(eje, x + 0.08, top - 0.3, "por separado, desde cada best.pt:", TAM, TINTA_2)
    filas = [("TensorRT · GPU", "FP32 · FP16 · INT8 PTQ   = 21"), ("OpenVINO · CPU", "FP32 · INT8 (NNCF)   = 12"),
             ("ModelOpt · GPU · INT8 (Q/DQ)", "PTQ-MO 6 + QAT 6   = 12")]
    for k, (a, b) in enumerate(filas):
        yy = top - 0.8 - k * 0.44
        nodo(eje, x + 0.08, yy, w - 0.16, 0.36, f"{a}\n{b}", 2)

    # 4 Evaluación
    x, w = xs[3], anchos[3]
    bloque(eje, x, yb, w, hb, 3, "Evaluación", "→ predicciones a 5")
    texto(eje, x + 0.07, top - 0.32, "52 evaluaciones\n(45 + 7 .pt control)\n\nPrueba · batch 1\n640 · rect=False\n"
          "conf 0.001 · IoU 0.7\n\nmAP, AP por clase,\nP, R, ECE\nΔmAP en el mismo\nmotor")

    # 5 Estadística
    x, w = xs[4], anchos[4]
    bloque(eje, x, yb, w, hb, 4, "Estadística", "→ tablas del paper")
    texto(eje, x + 0.07, top - 0.32, "Bootstrap pareado\nB = 1 000 (s. 42)\nbase_s vs. kd_s\n\nP1–P5 → H1–H5\n"
          "Confirm.: H3a, H3b,\nH4a, H4b, H5\n(Holm, α = 0.05;\nH5: δ = 0.010)\nExplor.: H1, H2a/b,\nH4c/d (IC 95 %)")

    # flechas entre bloques
    for i in range(4):
        flecha(eje, xs[i] + anchos[i] + 0.02, yb + hb / 2, xs[i + 1] - 0.02, yb + hb / 2)

    # conjuntos que 1 entrega directamente (punteadas, por arriba)
    c1 = xs[0] + anchos[0] / 2
    for dx, destino, alto, etq in ((-0.2, 2, 0.17, "calibración: 500 imágenes de train"),
                                   (0.2, 3, 0.36, "prueba: IN 1 166 · JP 1 565 imágenes")):
        xd = xs[destino] + anchos[destino] / 2 + (0.15 if destino == 3 else 0)
        yy = top + alto
        eje.plot([c1 + dx, c1 + dx, xd], [top, yy, yy], color=TINTA_2, lw=0.8, ls=(0, (3, 2)))
        flecha(eje, xd, yy, xd, top + 0.01, ls=(0, (3, 2)), lw=0.8)
        texto(eje, (c1 + xd) / 2 + 0.3, yy + 0.03, etq, TAM, TINTA_2, ha="center", va="bottom")

    # 6 Eficiencia (alimentado por 3)
    x6, w6, y6, h6 = xs[2], xs[3] + anchos[3] - xs[2], 0.08, 0.58
    rect(eje, x6, y6, w6, h6, FONDO)
    rect(eje, x6, y6, 0.95, h6, COLORES[5])
    texto(eje, x6 + 0.06, y6 + h6 / 2, "6  Eficiencia", TAM_T, "white", True, va="center")
    texto(eje, x6 + 1.03, y6 + h6 / 2, "45 modelos × 3 repeticiones de 50 calentamientos + 500 medidas\n"
          "Reloj de la GPU fijado en 2 505 MHz · latencia de inferencia, FPS y MB\n"
          "RTX 4060 (TensorRT) e i7-12700KF (OpenVINO)", va="center")
    flecha(eje, xs[2] + 0.35, yb - 0.01, xs[2] + 0.35, y6 + h6 + 0.01)

    # nota de la réplica
    xn, wn_ = xs[0], xs[1] + anchos[1] - xs[0]
    rect(eje, xn, y6, wn_, h6, "white", TINTA_2, 0.6, ls=(0, (3, 2)))
    texto(eje, xn + 0.07, y6 + h6 / 2, "IN y JP: mismo protocolo y mismo código.\nPor subconjunto: maestro + 6 alumnos + piloto\n"
          "de λ (3) + 6 QAT · 45 modelos · 52 evaluaciones.", TAM, TINTA_2, va="center")

    base = resolver_ruta(CFG["salida"])
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".png"), dpi=CFG["dpi_png"], facecolor="white")
    plt.close(fig)
    print(f"Figura: {base.with_suffix('.pdf')} y .png")


if __name__ == "__main__":
    main()
