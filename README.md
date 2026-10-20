# Destilación de conocimiento vs. pérdida por cuantización en YOLO (detección de daños viales)

Código del paper (IEEE, en preparación) que estudia si la **destilación de conocimiento (KD)** hace que un
detector YOLO ligero pierda menos precisión al **cuantizarse a INT8**, usando el subconjunto India de
RDD2022.

> El diseño experimental completo (las decisiones que no se cambian sin justificarlo) está en
> [CLAUDE.md](CLAUDE.md). La bitácora con fechas, comandos y resultados de cada fase está en
> [results/LOG.md](results/LOG.md). Este README es la guía de uso: qué hace cada fase y cómo se ejecuta.

---

## 1. Preguntas de investigación

| | Pregunta | Cómo se mide |
|---|---|---|
| **P1** | ¿La KD reduce la pérdida de mAP por cuantización INT8 en un YOLO11n? | ΔmAP@0.5 = mAP(FP32) − mAP(INT8), alumno base vs. alumno KD |
| **P2** | ¿El alumno KD conserva mejor calibración (ECE) tras cuantizar? | ECE del alumno base vs. alumno KD en INT8 |
| **P3** | ¿Un alumno KD con PTQ alcanza a un alumno base con QAT? | mAP@0.5(KD + PTQ INT8) vs. mAP@0.5(base + QAT INT8), no inferioridad con δ = 0.010 |

**Idea general del experimento**

```
                   ┌──────────────── maestro YOLO11m (congelado) ───────────────┐
                   │                                                             ▼
datos RDD2022-India ──► alumno YOLO11n BASE (solo L_det)          alumno YOLO11n KD (L_det + λ·L_CWD)
                              │                                                 │
                              ▼                                                 ▼
                 FP32 · FP16 · INT8 (PTQ) · INT8 (QAT)        FP32 · FP16 · INT8 (PTQ) · INT8 (QAT)
                              │                                                 │
                              └──────────► evaluación en test, ECE, bootstrap ◄─┘
```

Los dos alumnos usan exactamente los mismos hiperparámetros; **la única diferencia es la pérdida de
destilación CWD**. Cada alumno se entrena con 3 semillas (0, 1, 2) para medir la variabilidad del
entrenamiento.

---

## 2. Estado del proyecto

| Fase | Qué es | Estado |
|---|---|---|
| 0 | Entorno y estructura | ✅ |
| 1 | Preparación de datos y auditoría | ✅ |
| 1b | Pruebas de tiempo (AMP vs. FP32) | ✅ |
| 2 | Maestro YOLO11m | ✅ |
| 2b | Segundo maestro candidato YOLO11l y selección | ✅ (se usa el 11m) |
| 3 | Alumnos base (3 semillas) | ✅ |
| 4a | Implementación de CWD | ✅ |
| 4b | Piloto de λ (λ = 1.0) y alumnos KD (3 semillas) | ✅ |
| 5 | Cuantización PTQ (TensorRT y OpenVINO) | ✅ |
| 5b | QAT con NVIDIA ModelOpt | ✅ |
| 6 | Evaluación en test (mAP, ECE) | ✅ PTQ (la QAT se evalúa con el mismo código) |
| 7 | Estadística (bootstrap, Holm) | ✅ India |
| 8 | Eficiencia (latencia, FPS, tamaño) | pendiente |
| 9 | Tablas y figuras del paper | pendiente |
| J1 | Réplica en Japón: datos y auditoría | ✅ |
| J2–J9 | Réplica en Japón: maestro, alumnos, cuantización, evaluación, estadística | ⏳ J2 hecho; J3 siguiente (ver checklist) |

### Resultados hasta ahora (validación)

Todos se obtuvieron en el split de **validación**; el split test no se ha usado y queda reservado para la
Fase 6. Métricas de best.pt revalidado con imgsz 640, batch 8, conf 0.001 e iou 0.7 (media ± d.e. de 3
semillas para los alumnos):

| Modelo | mAP@0.5 | mAP@0.5:0.95 | AP@0.5 D40 | mAP@0.5 sin D10 |
|---|---|---|---|---|
| Maestro YOLO11m | 0.345 | 0.147 | 0.437 | 0.449 |
| Alumnos base YOLO11n | 0.342 ± 0.022 | 0.145 ± 0.006 | 0.397 ± 0.005 | 0.424 ± 0.020 |
| Alumnos KD YOLO11n (λ = 1.0) | 0.347 ± 0.008 | 0.147 ± 0.001 | 0.411 ± 0.016 | 0.437 ± 0.012 |

- La KD mejora levemente al alumno en FP32 (sobre todo en D40) y reduce mucho la variabilidad entre
  semillas (d.e. de mAP@0.5:0.95: 0.001 frente a 0.006).
- La PTQ INT8 degrada a los alumnos base en 3/3 semillas: −3.1 puntos de mAP@0.5 (−9.5 %) y −4.3 en D40,
  con más precisión y menos recall (−10.9 %). FP16 no pierde precisión (exploración `ptq_check`).

### Resultados en test (Fases 5, 5b y 6; descriptivos, la inferencia es la Fase 7)

Todos los modelos con letterbox cuadrado 640×640 y batch 1; pérdida = FP32 − cuantizado dentro de TensorRT
(media ± d.e. de 3 semillas):

| | mAP@0.5 FP32 | Pérdida PTQ INT8 | Pérdida control ModelOpt PTQ | Pérdida QAT INT8 | ECE PTQ INT8 | ECE QAT INT8 |
|---|---|---|---|---|---|---|
| Alumnos base | 0.310 ± 0.020 | 0.035 ± 0.009 | 0.006 ± 0.009 | 0.006 ± 0.003 | 0.096 | 0.077 |
| Alumnos KD | 0.316 ± 0.005 | 0.031 ± 0.007 | 0.013 ± 0.003 | 0.011 ± 0.009 | 0.113 | 0.085 |
| Maestro YOLO11m | 0.329 | 0.178 (colapso de confianzas) | — | — | — | — |

- Con la PTQ de TensorRT (Ultralytics), el alumno KD pierde algo menos (2/3 semillas); con QAT, algo más.
- Casi toda la ganancia de la QAT viene del esquema de cuantización de ModelOpt (Q/DQ explícito, pesos por
  canal): el control sin ajuste fino ya pierde casi lo mismo que la QAT.
- FP16 no pierde nada, y OpenVINO INT8 (NNCF) casi no pierde (≤ 0.3 puntos).
- El maestro colapsa con la PTQ de TensorRT por valores extremos en las activaciones de su cabeza (ver
  "Exploraciones").

---

## 3. Entorno

| Elemento | Valor |
|---|---|
| GPU / CPU / RAM | NVIDIA RTX 4060 8 GB · Intel i7-12700KF · 32 GB (24 GB asignados a WSL) |
| Sistema | WSL2 (Ubuntu 26.04) sobre Windows 11 |
| Python | 3.13, en el venv `~/.venvs/yolokd` (disco de Linux) |
| Librerías clave | PyTorch 2.14 (CUDA 13.0), ultralytics **8.4.164**, TensorRT **10.16**, OpenVINO 2026.4, NNCF 3.4 |

Dos versiones están fijadas a propósito:

- **TensorRT 10, no 11.** La rama 10 conserva la cuantización INT8 implícita con calibrador; la 11 obliga
  a cuantización explícita Q/DQ. TensorRT 10 no tiene versión para Python 3.14, por eso se usa 3.13.
- **ultralytics 8.4.164.** El calibrador INT8 depende de la versión del exportador de Ultralytics.

### Instalación (una sola vez, en WSL)

```bash
cd "/mnt/d/1-Personal/5-Studies/MaestriaIA/S2/Vision de computadora/proyecto_yolo/Codigo"
bash scripts/setup_wsl.sh            # instala uv, crea el venv con Python 3.13 e instala requirements.txt
source ~/.venvs/yolokd/bin/activate
python scripts/check_env.py          # verifica CUDA, TensorRT, OpenVINO y ultralytics → results/env_check.json
```

- `requirements.txt` tiene las dependencias directas con versión fija.
- `requirements.lock.txt` tiene el entorno completo congelado (todas las dependencias).

### Rutas en WSL

El código vive en `D:` (en WSL, `/mnt/d/...`), pero el venv y las imágenes convertidas viven en el disco de
Linux (`~/.venvs/yolokd` y `~/datasets/rdd2022_india_yolo`), porque leer miles de archivos desde `/mnt/d`
es lento. Los scripts calculan las rutas a partir de su propia ubicación, así que funcionan desde
cualquier carpeta.

### Cómo lanzar entrenamientos largos

Todos los entrenamientos se lanzan dentro de `tmux` para que sigan corriendo si se cierra la terminal:

```bash
tmux new -s nombre          # abrir sesión
bash scripts/....sh         # lanzar
# Ctrl+B y luego D          → dejarlo corriendo
tmux attach -t nombre       # volver a verlo
```

Conviene pausar las actualizaciones de Windows durante corridas largas: un reinicio corta el entrenamiento.

---

## 4. Estructura del repositorio

```
Codigo/
├── CLAUDE.md                 diseño experimental y reglas del proyecto
├── README.md                 esta guía
├── requirements.txt          dependencias directas (versiones fijas)
├── requirements.lock.txt     entorno completo congelado
├── configs/                  TODA la configuración (nada fijo dentro del código)
│   ├── data_prep.yaml        preparación de datos (clases, partición, calibración)
│   ├── data.yaml             dataset para Ultralytics (generado, no editar)
│   ├── calib.yaml            500 imágenes de train para calibrar INT8 (generado, no editar)
│   ├── train.yaml            hiperparámetros de maestros y alumnos
│   ├── kd.yaml               destilación CWD: maestro, τ, λ, piloto
│   ├── timing.yaml           pruebas de tiempo (Fase 1b)
│   └── exploracion_ptq.yaml  exploración de la degradación por PTQ
├── data/
│   ├── raw/                  RDD2022 original (no versionado)
│   └── splits/               listas train/val/test/calib (versionadas)
├── src/
│   ├── data/                 conversión, limpieza, partición y estadísticas
│   ├── train/                maestros, alumnos, CWD, piloto de λ, pruebas de tiempo
│   ├── quant/ eval/ bench/ report/   fases 5–9 (por implementar)
│   └── utils/                semillas, versiones y carga de configuración
├── scripts/                  lanzadores .sh, verificación del entorno y exploraciones
├── runs/                     corridas, pesos y exportaciones (no versionado)
├── results/                  LOG.md, tablas, figuras y métricas
└── paper/                    borrador del paper (no versionado)
```

Convención de nombres de corridas: `{rol}_{kd|base}_s{semilla}`, por ejemplo `student_kd_s1`.
El maestro YOLO11m es `teacher_s0` y el YOLO11l, `teacher_l_s0`.

---

## 5. Fases: qué se hace, cómo se ejecuta y qué se obtuvo

### Fase 0 — Entorno

Crea la estructura, fija versiones y verifica que la GPU, TensorRT y OpenVINO funcionen.

| Script | Para qué |
|---|---|
| `scripts/setup_wsl.sh` | crea el venv e instala dependencias |
| `scripts/check_env.py` | verifica el entorno y guarda `results/env_check.json` |
| `src/utils/repro.py` | fija semillas y registra versiones en cada corrida |

### Fase 1 — Datos

Convierte las anotaciones Pascal VOC (XML) de `data/raw/RDD2022/India/train` a formato YOLO, aplica las
reglas de limpieza, particiona y genera los yaml para Ultralytics. El test oficial de RDD2022 no tiene
etiquetas, así que train/val/test salen del train oficial.

```bash
python -m src.data.prepare_dataset     # conversión, limpieza, partición, calibración, auditoría
python -m src.data.dataset_stats       # imágenes e instancias por clase y split
```

- **Clases:** D00 (grieta longitudinal), D10 (grieta transversal), D20 (piel de cocodrilo), D40 (bache).
- **Limpieza:** se eliminan solo las cajas de otras etiquetas (D01, D11, D43, D44, D50 y una caja ambigua
  "D0w0"); ninguna imagen se elimina. Las imágenes sin cajas se conservan como fondo (58 %).
- **Partición:** 70/15/15 con estratificación iterativa multietiqueta (semilla 42): 5 396 / 1 144 / 1 166
  imágenes.
- **Calibración INT8:** 500 imágenes de train, estratificadas; nunca se calibra con val ni test.
- **Salidas:** `results/tables/data_audit.csv`, `results/tables/dataset_stats.csv`, `data/splits/`.

Limitación conocida: D10 tiene solo 68 instancias (9 en test). La mAP de 4 clases es la métrica principal
y la mAP sin D10 se reporta como análisis de sensibilidad.

### Fase 1b — Pruebas de tiempo

Mide el tiempo por época y la VRAM antes de comprometer horas de GPU.

```bash
bash scripts/fase1b_tiempos.sh         # src/train/timing_test.py + configs/timing.yaml
```

- Maestro YOLO11m: con AMP, 123 s por época y 4.2 GB; en FP32 satura los 8 GB y es unas 7 veces más lento.
- **Decisión:** precisión mixta (AMP) para todos los modelos. AMP no es cuantización: los pesos se guardan
  en FP32.
- Estimación total del proyecto: unas 41 h de GPU (`results/tables/timing.csv`).

### Fase 2 y 2b — Maestros

```bash
bash scripts/train_teacher.sh                  # YOLO11m → runs/teacher/teacher_s0
bash scripts/train_teacher.sh --rol maestro_l  # YOLO11l → runs/teacher/teacher_l_s0
bash scripts/train_teacher.sh --humo           # prueba rápida (1 época, 5 % de train)
```

Configuración común (`configs/train.yaml`): imgsz 640, batch 8, 300 épocas, SGD (lr0 0.01), aumentación
por defecto de Ultralytics, AMP, `deterministic=True`, caché en disco, parada temprana con paciencia 50.

| Maestro | best.pt (época) | val mAP@0.5 | val mAP@0.5:0.95 | D40 |
|---|---|---|---|---|
| **YOLO11m** | 148 | 0.345 | **0.147** | 0.437 |
| YOLO11l | 105 | 0.339 | 0.144 | 0.426 |

Se entrenó el YOLO11l porque el 11m apenas superaba a los alumnos. La regla, fijada antes de ver el
resultado, era usar el 11l solo si superaba al 11m en al menos 0.010 de mAP@0.5:0.95 en validación. No lo
hizo, así que **el maestro es YOLO11m**. Que un modelo un 26 % más grande no mejore indica que el techo lo
pone el conjunto de datos, no la capacidad.

### Fase 3 — Alumnos base

```bash
for s in 0 1 2; do bash scripts/train_student.sh --kd off --seed $s; done
```

YOLO11n, misma configuración que los maestros, 300 épocas sin parada temprana →
`runs/students/student_base_s{0,1,2}`. Unas 2.7 h por corrida, sin NaN.

**Entrenamiento:**

| Corrida | Época de best.pt | Horas | Pendiente de la mAP en las últimas 50 épocas (mAP@0.5 / mAP@0.5:0.95, por 50 ép.) |
|---|---|---|---|
| base_s0 | 198 | 2.71 | −0.0023 / −0.0027 |
| base_s1 | 238 | 2.73 | −0.0006 / −0.0029 |
| base_s2 | 114 | 2.74 | −0.0021 / −0.0000 |

Las tres corridas convergieron: al final del entrenamiento la mAP de validación está plana o baja levemente.

**Resultados en validación** (best.pt revalidado con imgsz 640, batch 8, conf 0.001, iou 0.7):

| Modelo | mAP@0.5 | mAP@0.5:0.95 | D00 | D10 | D20 | D40 | mAP@0.5 sin D10 |
|---|---|---|---|---|---|---|---|
| base_s0 | 0.366 | 0.151 | 0.344 | 0.149 | 0.568 | 0.401 | 0.438 |
| base_s1 | 0.334 | 0.142 | 0.326 | 0.034 | 0.576 | 0.399 | 0.434 |
| base_s2 | 0.325 | 0.141 | 0.249 | 0.097 | 0.562 | 0.391 | 0.401 |
| **Media ± d.e.** | **0.342 ± 0.022** | **0.145 ± 0.006** | | | | **0.397 ± 0.005** | **0.424 ± 0.020** |
| Maestro YOLO11m (referencia) | 0.345 | 0.147 | 0.311 | 0.034 | 0.597 | 0.437 | 0.449 |

- **Variabilidad entre semillas alta:** 4 puntos de mAP@0.5 entre s0 y s2, lo que justifica entrenar 3
  semillas. Parte viene de D10 (9 instancias en val), cuyo AP salta entre 0.03 y 0.15.
- **El maestro apenas supera a los alumnos.** En mAP@0.5 global, base_s0 lo supera por el AP de D10. Sin D10,
  el maestro gana por unos 2.5 puntos (0.449 frente a 0.424) y en D40 por unos 4 (0.437 frente a 0.397); en
  mAP@0.5:0.95 la diferencia es de 0.2 puntos. Esto motivó probar un maestro más grande (Fase 2b).

### Fase 4a — Destilación CWD

`src/train/cwd.py` implementa **Channel-Wise Distillation** (Shu et al., ICCV 2021) como subclase del
`DetectionTrainer` de Ultralytics. La fórmula está adaptada de MMRazor (commit `0919f69`).

- Se destilan las tres entradas de la cabeza de detección (P3, P4, P5). El maestro está congelado y ve el
  mismo batch aumentado que el alumno.
- Unas convoluciones 1×1 (adaptadores) llevan los canales del alumno (64/128/256) a los del maestro
  (256/512/512). Se entrenan con el alumno, pero **nunca se guardan en best.pt** ni se exportan.
- Pérdida: **L = L_det + λ·L_CWD**, con τ = 1. `results.csv` registra L_CWD por época (`train/cwd`).

```bash
bash scripts/train_student.sh --kd on --seed 0 --humo
bash scripts/train_student.sh --kd off --seed 0 --humo
python -m src.train.check_kd --kd runs/smoke/student_kd_s0_humo --base runs/smoke/student_base_s0_humo
```

`check_kd` verifica que L_CWD baja, que el best.pt del alumno KD tiene exactamente los mismos tensores que
el del base (sin adaptadores ni maestro) y que los ONNX exportados tienen el mismo número de parámetros.

### Fase 4b — Piloto de λ y alumnos KD

**Piloto:** 3 alumnos KD cortos (λ ∈ {0.5, 1.0, 2.0}, semilla 0, 50 épocas). Se elige el λ con mayor
mAP@0.5:0.95 en validación.

```bash
bash scripts/pilot_lambda.sh           # entrena los 3 λ y corre src/train/select_lambda.py
```

| λ | L_CWD (época 1 → 50) | Peso de λ·L_CWD sobre L_det | val mAP@0.5:0.95 |
|---|---|---|---|
| 0.5 | 1.46 → 0.47 | 4 % | 0.134 |
| **1.0** | 1.42 → 0.41 | 8 % | **0.144** |
| 2.0 | 1.37 → 0.36 | 14 % | 0.130 |

**λ = 1.0** queda en `configs/kd.yaml`. Gráfica: `results/figures/pilot_lambda.png`.

**Alumnos KD finales:**

```bash
for s in 0 1 2; do bash scripts/train_student.sh --kd on --seed $s; done
# si una corrida se corta (p. ej., reinicio del PC):
bash scripts/train_student.sh --kd on --seed N --reanudar
```

`--reanudar` retoma desde `weights/last.pt` y restaura los adaptadores, que se guardan aparte en
`weights/kd_adaptadores_last.pt`. Una corrida reanudada no queda idéntica bit a bit a una sin cortes; si
se usa, se anota en el LOG.

Resultado en validación (media ± d.e. de 3 semillas): mAP@0.5 = 0.347 ± 0.008 y mAP@0.5:0.95 =
0.147 ± 0.001, frente a 0.342 ± 0.022 y 0.145 ± 0.006 de los alumnos base. `check_kd` confirmó en las 3
semillas que los best.pt KD no contienen adaptadores.

### Fase 5 — Cuantización post-entrenamiento (PTQ)

```bash
bash scripts/export_all.sh                              # todos los modelos de configs/quant.yaml (reanudable)
bash scripts/export_all.sh --modelos student_kd_s0      # uno o varios
```

`src/quant/export_all.py` exporta cada modelo, cada precisión por separado desde su best.pt, con el
exportador de Ultralytics: TensorRT FP32/FP16/INT8 y OpenVINO FP32/INT8 (el maestro solo en TensorRT).
Antes de exportar verifica que las 500 imágenes de calibración sean de train y no toquen val ni test, y en
cada INT8 registra de dónde calibró. `src/quant/engine_layers.py` cuenta en qué precisión quedó cada capa
de los engines INT8.

- 33 exportaciones en `runs/exports/{modelo}_{formato}_{precision}/` (tabla `results/tables/exports.csv`).
- Tamaño de un alumno: TensorRT 12.9 MB (FP32) → 7.5 MB (FP16) → 4.5 MB (INT8); OpenVINO 10.2 → 3.2 MB.
- El INT8 de TensorRT es mixto: ~65–68 % de las capas de cómputo quedan en INT8 y el resto en FP16 o FP32
  (por ejemplo, el bloque de atención y la sigmoide final). El patrón es el mismo en alumnos base y KD
  (`results/tables/engine_layers.csv`).

### Fase 5b — QAT con NVIDIA ModelOpt

```bash
bash scripts/qat.sh                                # los 6 alumnos (reanudable)
bash scripts/qat.sh --alumnos student_base_s0 --humo   # prueba de humo
```

`src/quant/qat.py` (configuración congelada en `configs/qat.yaml`) usa la QAT integrada en Ultralytics
(`train(quantize=8)`, cuantización explícita Q/DQ INT8 con ModelOpt) con dos cambios para cumplir D5:

1. La calibración usa las mismas 500 imágenes de `calib.yaml` que la PTQ, sin aumentación (Ultralytics usa
   por defecto 8 lotes aumentados de entrenamiento). Calibrador de ModelOpt: algoritmo "max", activaciones
   por tensor y pesos por canal.
2. Antes del ajuste se guarda y exporta el modelo solo calibrado ("ModelOpt PTQ") como control, para medir
   la ganancia de la QAT contra su mismo esquema de cuantización.

El ajuste fino usa solo la pérdida de detección (también en los alumnos KD), 30 épocas, lr0 0.001, sin
warmup ni mosaico; el punto de control es best.pt por validación. Cada ajuste tardó ~1.1 h. Los engines
resultantes se registran en `exports.csv` como `mopt_int8` (control) y `qat_int8`.

### Fase 6 — Evaluación en test

```bash
bash scripts/evaluate.sh                 # evalúa lo que falte (reanudable) y recalcula las tablas
bash scripts/evaluate.sh --solo-tablas   # solo recalcula consistencia y ΔmAP
```

`src/eval/evaluate.py` (configuración `configs/eval.yaml`) evalúa en **test** el .pt FP32 y todas las
exportaciones: letterbox cuadrado 640×640, batch 1, conf 0.001, iou 0.7. Antes de evaluar cada engine verifica
su hash SHA-256 en `results/tables/engines.csv`. Guarda las predicciones por imagen (`runs/eval/*/predictions.json`),
que reutilizan el ECE y el bootstrap.

- **ECE** (`src/eval/ece.py`): detecciones con confianza ≥ 0.25, emparejadas de mayor a menor confianza con la
  caja real libre de mayor IoU (≥ 0.5); correcta si coincide la clase; 15 intervalos de igual ancho.
- **Control de consistencia:** el .pt FP32 debe coincidir con TensorRT FP32 (|Δ mAP@0.5| < 0.002); se cumple en
  los 7 modelos (máx. 0.0008).
- Salidas: `results/metrics.csv` (52 filas: una por modelo/formato/precisión),
  `results/tables/delta_map.csv` y `results/tables/consistency_pt_vs_trt.csv`.

### Fase 7 — Estadística (India)

```bash
python -m src.eval.evaluate --solo-stats   # una vez: guarda los aciertos exactos por imagen de Ultralytics
python -m src.eval.bootstrap               # bootstrap pareado D7 (configs/stats.yaml)
```

`src/eval/bootstrap.py` aplica D7 tal cual: 1 000 remuestreos de las 1 166 imágenes de test (semilla 42, mismos
índices para todos los modelos), diferencias por semilla promediadas, p bootstrap centrado, Holm sobre 5 contrastes.
La mAP de cada réplica usa los aciertos exactos de Ultralytics por imagen y su misma `ap_per_class`; sin remuestreo
reproduce la mAP de Ultralytics (máx. |Δ| = 0.000005).

| H | Pregunta | D observado | Intervalo | p (Holm) | Decisión |
|---|---|---|---|---|---|
| H1a | ¿KD pierde menos mAP@0.5 con PTQ INT8? | +0.004 | IC95 [−0.007, +0.015] | 1.00 | no significativo (2/3 semillas a favor de KD) |
| H1b | ¿KD pierde menos mAP@0.5 con QAT INT8? | −0.005 | IC95 [−0.020, +0.015] | 1.00 | no significativo (0/3) |
| H2a | ¿KD tiene mejor ECE con PTQ INT8? | −0.016 | IC95 [−0.031, +0.013] | 1.00 | no significativo (1/3) |
| H2b | ¿KD tiene mejor ECE con QAT INT8? | −0.008 | IC95 [−0.027, +0.014] | 1.00 | no significativo (1/3) |
| H3 | ¿KD + PTQ no es inferior a base + QAT (δ = 0.01)? | −0.019 | IC90 [−0.033, −0.009] | 1.00 | no se demuestra; base + QAT es mejor |

En India la KD no reduce de forma significativa la pérdida por cuantización ni mejora la calibración, y un alumno
KD con PTQ no alcanza a un alumno base con QAT. Detalle y exploratorios en `results/tables/hypothesis_tests.csv` y
`results/tables/exploratory.csv`.

### Fases 8 y 9 — Pendientes

| Fase | Qué hará |
|---|---|
| 8 — Eficiencia | Latencia, FPS y tamaño en GPU (TensorRT) y CPU (OpenVINO); sin otra carga, 3 repeticiones |
| 9 — Reporte | Tablas III–V y figuras del paper |

El plan estadístico (hipótesis, α, δ y número de remuestreos) está fijado de antemano en
[CLAUDE.md](CLAUDE.md), sección D7, y no se cambia después de ver resultados.

### Réplica en RDD2022-Japón (D9)

El estudio completo se repetirá con el subconjunto **Japón** de RDD2022, para tener dos datasets de entrada
y ver si los efectos se repiten. Se usa exactamente el mismo protocolo y el mismo código que en India; solo
cambian las rutas, que van todas a carpetas separadas: `configs/japan/`, `data/splits/japan/`, `runs/japan/`
y `results/japan/`.

**J1 — Datos (hecho):**

```bash
python -m src.data.prepare_dataset --config configs/japan/data_prep.yaml
python -m src.data.dataset_stats   --config configs/japan/data_prep.yaml
```

| | India | Japón |
|---|---|---|
| Imágenes etiquetadas | 7 706 | 10 506 |
| Partición train / val / test | 5 396 / 1 144 / 1 166 | 7 346 / 1 595 / 1 565 |
| Cajas conservadas (D00 / D10 / D20 / D40) | 1 555 / 68 / 2 021 / 3 187 | 4 049 / 3 979 / 6 198 / 2 243 |
| Instancias de D10 en test | 9 | 591 |
| Imágenes de fondo | 58.2 % | 24.8 % |
| Tamaño de imagen | 720×720 | 600×600, 540×540 y 1024×1024 |

En Japón se eliminaron 8 284 cajas de otras clases (D44, D50 y D43) y 1 caja de D20 de menos de 1 px; ninguna
imagen corrupta. Detalle en `results/japan/tables/data_audit.csv` y `results/japan/tables/dataset_stats.csv`.

Las fases J2 a J9 usan los **mismos scripts** con el argumento `--dataset japan`, que toma la configuración
de `configs/japan/` (mismos nombres de archivo que en India). Checklist:

| Fase | Comando | Tiempo estimado | Estado |
|---|---|---|---|
| J2 — Maestro YOLO11m | `bash scripts/train_teacher.sh --dataset japan` | 8–14 h (real: 5.6 h) | ✅ val mAP@0.5 0.571 |
| J3 — Alumnos base s0–s2 | `for s in 0 1 2; do bash scripts/train_student.sh --kd off --seed $s --dataset japan; done` | ~11 h | pendiente |
| J4a — Humo y verificación KD | `bash scripts/train_student.sh --kd on --seed 0 --humo --dataset japan` + `check_kd` | minutos | pendiente |
| J4b — Piloto de λ | `bash scripts/pilot_lambda.sh japan` (luego se fija λ en `configs/japan/kd.yaml`) | ~4 h | pendiente |
| J4c — Alumnos KD s0–s2 | `for s in 0 1 2; do bash scripts/train_student.sh --kd on --seed $s --dataset japan; done` | ~26 h | pendiente |
| J5 — PTQ | `bash scripts/export_all.sh --dataset japan` | ~2 h | pendiente |
| J5b — QAT | `bash scripts/qat.sh --dataset japan` | 7–9 h | pendiente |
| J6 — Evaluación en test | `bash scripts/evaluate.sh --dataset japan` | < 1 h | pendiente |
| J7 — Estadística | familia D7 y Holm propios de Japón | — | pendiente |
| J8 — Eficiencia | D8 | — | pendiente |
| J9 — Tablas y figuras | comparación India vs. Japón | — | pendiente |

Decisiones confirmadas antes de ver resultados de Japón (D9): solo maestro YOLO11m, piloto de λ propio y
estadística D7 completa por separado en cada dataset. Estimación: unas 65–75 h de GPU.

---

## 6. Exploraciones (fuera del diseño principal)

| Script | Pregunta | Resultado |
|---|---|---|
| `scripts/exploracion/explora_teacher_l.py` | ¿Cabe y cuánto cuesta un maestro YOLO11l? | 143 s/época y 5.3 GB; motivó la Fase 2b |
| `scripts/exploracion/teacher_int8_diag.py` | ¿Por qué colapsan las confianzas del maestro en TensorRT INT8? (en **validación**) | Valores extremos en las activaciones de su cabeza (máx/p99.99 hasta 37, frente a ≤ 2.4 en los alumnos) arruinan la escala MinMax; con OpenVINO/NNCF el maestro no pierde nada |
| `scripts/exploracion/ptq_check.py` | ¿La PTQ INT8 degrada a los alumnos base? (en **validación**, para no ver el test) | Sí, en 3/3 semillas: −3.1 puntos de mAP@0.5 (−9.5 %), −4.3 en D40 y −3.9 de recall; la precisión sube (modelo más conservador). FP16 no pierde nada |

Uso: `PYTHONPATH=. python scripts/exploracion/<script>.py`.

**Hallazgo metodológico de `ptq_check`:** evaluar el best.pt de PyTorch con la validación por defecto de
Ultralytics (letterbox rectangular) da 1–2 puntos más de mAP que un engine TensorRT con entrada fija de
640×640. Con letterbox cuadrado (`rect=False`), PyTorch y TensorRT FP32 coinciden (|Δ| ≤ 0.0001).

Por eso el diseño (D6) fija que **todos los modelos se evalúan con letterbox cuadrado 640×640 (`rect=False`)
y batch 1**, y que ΔmAP se calcula **dentro del mismo motor**: en GPU, TRT FP32 − TRT {FP16, INT8, QAT INT8};
en CPU, OV FP32 − OV INT8. El .pt FP32 evaluado con `rect=False` sirve de control: debe coincidir con TRT FP32
(|Δ mAP@0.5| < 0.002).

---

## 7. Reproducibilidad

- **Semillas:** Python, NumPy y PyTorch se fijan en cada corrida; `deterministic=True`.
- **Trazabilidad:** cada corrida guarda en su carpeta `args.yaml`, una copia de los yaml usados
  (`usada_*.yaml`) y `entorno.json` (versiones de librerías, GPU y commit de git).
- **Sin datos de test durante el desarrollo:** maestro, λ y todas las exploraciones se eligieron con
  validación. El split test solo se usará en la Fase 6.
- **Reglas fijadas antes de los resultados:** la selección del maestro (D2), la elección de λ (D3) y el
  plan estadístico (D7) quedaron escritos antes de tener los datos correspondientes.
- **Bitácora:** cada fase deja en [results/LOG.md](results/LOG.md) la fecha, el comando, el resultado
  principal y las decisiones tomadas.

---

## 8. Licencia

El código de este repositorio se distribuye bajo la **GNU Affero General Public License v3.0
(AGPL-3.0)**; el texto completo está en [LICENSE](LICENSE).

Se eligió AGPL-3.0 porque el proyecto se construye sobre
[Ultralytics](https://github.com/ultralytics/ultralytics), que usa esa misma licencia (por ejemplo,
`src/train/cwd.py` extiende su `DetectionTrainer`). En la práctica:

- Cualquiera puede usar, estudiar, modificar y ejecutar este código, también para reproducir los
  experimentos del paper.
- Quien distribuya una versión modificada, o la ofrezca como servicio en red, debe publicar su código
  fuente bajo AGPL-3.0.

El repositorio **no incluye** las imágenes ni las anotaciones de RDD2022 (solo las listas de nombres de
archivo en `data/splits/`). El conjunto de datos se obtiene de su fuente original y está sujeto a su
propia licencia.

## 9. Créditos y referencias

Si usas este código, cita el paper asociado (referencia por publicar) y las obras en las que se apoya:

- **Conjunto de datos RDD2022.** D. Arya, H. Maeda, S. K. Ghosh, D. Toshniwal y Y. Sekimoto,
  "RDD2022: A multi-national image dataset for automatic road damage detection", arXiv:2209.08538, 2022.
  <https://github.com/sekilab/RoadDamageDetector>
- **Ultralytics YOLO11** (AGPL-3.0). G. Jocher y J. Qiu, *Ultralytics YOLO11*, 2024.
  <https://github.com/ultralytics/ultralytics>
- **Channel-Wise Distillation (CWD).** C. Shu, Y. Liu, J. Gao, Z. Yan y C. Shen, "Channel-wise knowledge
  distillation for dense prediction", en *Proc. IEEE/CVF ICCV*, 2021.
- **MMRazor** (Apache-2.0), de OpenMMLab: implementación de referencia de CWD adaptada en
  `src/train/cwd.py` (commit `0919f692878fe0037ff94bc90e46eed721c64a39`).
  <https://github.com/open-mmlab/mmrazor>
- **Estratificación iterativa multietiqueta.** K. Sechidis, G. Tsoumakas e I. Vlahavas, "On the
  stratification of multi-label data", en *ECML PKDD*, 2011; implementación
  [iterative-stratification](https://github.com/trent-b/iterative-stratification).
- **Bootstrap.** B. Efron y R. J. Tibshirani, *An Introduction to the Bootstrap*, Chapman & Hall, 1993.
