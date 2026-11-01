"""Presupuesto de cómputo: horas de GPU por fase → results/tables/compute_budget.csv.

Las horas de entrenamiento salen de la columna `time` (segundos acumulados) de la última fila de cada
results.csv; las de exportación, de results/tables/exports.csv. Lo que no tiene medición directa en un archivo
se marca como "estimado" con su origen (LOG.md). Volver a ejecutar tras cada fase para actualizarlo.

Uso: python -m src.report.compute_budget
"""

from __future__ import annotations

import csv
from pathlib import Path

from src.utils.config import resolver_ruta

# (fase, concepto, patrón de carpetas con results.csv)
CORRIDAS = [
    ("1b — Pruebas de tiempo", "maestro AMP 3 ép. + alumno AMP 3 ép.", "runs/timing/*"),
    ("2 — Maestros", "YOLO11m teacher_s0", "runs/teacher/teacher_s0"),
    ("2 — Maestros", "YOLO11l teacher_l_s0 (candidato)", "runs/teacher/teacher_l_s0"),
    ("2 — Maestros", "YOLO11m diseño v2 (obsoleto, detenido en la época 47)", "runs/obsoletos/teacher_s0_v2_3clases_150ep"),
    ("3 — Alumnos base", "student_base_s0..s2", "runs/students/student_base_s*"),
    ("4b — Piloto de λ", "pilot_kd_l{0.5,1.0,2.0}_s0", "runs/pilot/pilot_kd_*"),
    ("4b — Alumnos KD", "student_kd_s0..s2", "runs/students/student_kd_s*"),
    ("5b — QAT", "qat_{base,kd}_s0..s2 (ajuste fino)", "runs/qat/qat_*"),
    ("Exploración", "maestro YOLO11l, 3 épocas", "runs/exploracion/teacher_l_3ep"),
    ("J2 — Maestro (Japón)", "YOLO11m teacher_s0", "runs/japan/teacher/teacher_s0"),
    ("J3 — Alumnos base (Japón)", "student_base_s0..s2", "runs/japan/students/student_base_s*"),
    ("J4b — Piloto de λ (Japón)", "pilot_kd_l{0.5,1.0,2.0}_s0", "runs/japan/pilot/pilot_kd_*"),
    ("J4c — Alumnos KD (Japón)", "student_kd_s0..s2", "runs/japan/students/student_kd_s*"),
]

# Sin results.csv: estimaciones a partir de LOG.md
ESTIMADOS = [
    ("1b — Pruebas de tiempo", "maestro FP32 (detenido en la época 2) + forward del maestro", 0.5,
     "LOG 2026-09-28: época 1 de 947 s y época 2 al 79 %"),
    ("6 — Evaluación", "40 evaluaciones en test (7 .pt + 33 exportaciones)", 0.3,
     "LOG 2026-10-18: ~10 s por evaluación en GPU, más lento en CPU"),
    ("Exploración", "ptq_check (9 engines + 15 evaluaciones en val)", 1.0, "LOG 2026-10-03"),
    ("Exploración", "diagnóstico del maestro INT8 (OV + evaluaciones + activaciones)", 0.2, "LOG 2026-10-18"),
    ("Pruebas de humo", "maestros, alumnos, KD, piloto, QAT", 0.3, "suma aproximada de las pruebas de humo"),
    ("8 — Eficiencia", "45 engines × 3 repeticiones + pruebas de humo y diagnósticos", 0.5,
     "LOG 2026-10-22: ~15 min la medición completa"),
]


def horas_results(patron: str) -> tuple[float, int]:
    total, n = 0.0, 0
    for d in sorted(resolver_ruta(".").glob(patron)):
        f = d / "results.csv"
        if f.exists():
            with open(f, encoding="utf-8") as fh:
                filas = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(fh)]
            if filas:
                total += float(filas[-1]["time"]) / 3600
                n += 1
    return total, n


def main() -> None:
    filas = []
    for fase, concepto, patron in CORRIDAS:
        h, n = horas_results(patron)
        if n:
            filas.append({"fase": fase, "concepto": concepto, "corridas": n, "horas_gpu": round(h, 2),
                          "fuente": f"medido ({patron}/results.csv)"})
    exp = resolver_ruta("results/tables/exports.csv")
    if exp.exists():
        with open(exp, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        for etiqueta, sel in (("5 — Exportación PTQ", lambda r: r["precision"] in ("fp32", "fp16", "int8")),
                              ("5b — Exportación QAT y control", lambda r: r["precision"] in ("mopt_int8", "qat_int8"))):
            s = [r for r in rows if sel(r)]
            if s:
                filas.append({"fase": etiqueta, "concepto": f"{len(s)} exportaciones", "corridas": len(s),
                              "horas_gpu": round(sum(float(r["segundos"]) for r in s) / 3600, 2),
                              "fuente": "medido (results/tables/exports.csv)"})
    for fase, concepto, h, origen in ESTIMADOS:
        filas.append({"fase": fase, "concepto": concepto, "corridas": "", "horas_gpu": h, "fuente": f"estimado ({origen})"})
    total = round(sum(float(f["horas_gpu"]) for f in filas), 1)
    filas.append({"fase": "TOTAL", "concepto": "", "corridas": "", "horas_gpu": total, "fuente": ""})

    salida = resolver_ruta("results/tables/compute_budget.csv")
    with open(salida, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    for f in filas:
        print(f"{f['fase']:<32} {f['concepto']:<58} {f['horas_gpu']:>7} h  {f['fuente'][:40]}")
    print(f"\nGuardado en {salida}")


if __name__ == "__main__":
    main()
