# Bitácora de experimentos

## 2026-09-27 — Fase 0: entorno y estructura

- Entorno elegido: WSL2 (Ubuntu 26.04, 24 GB RAM), Python 3.13 en venv `~/.venvs/yolokd` (uv).
  Se descartó Python 3.14 porque TensorRT 10 no tiene versión para él y TensorRT 11 cambia el
  método INT8 (Q/DQ con ModelOpt, capas mixtas en float).
- Versiones fijadas en `requirements.txt`: ultralytics 8.4.164, torch 2.14.0+cu130,
  TensorRT 10.16.1.11 (rama 10, PTQ con IInt8EntropyCalibrator2), OpenVINO 2026.4.0, NNCF 3.4.0.
- Dataset movido de `dataset/` a `data/raw/` (RDD2022/India ya descomprimido:
  train = 7 706 imágenes + 7 706 XML; test = 1 959 imágenes sin etiquetas).
- Incidencia: numpy 2.5.3 chocaba con nncf 3.4.0 (exige numpy<2.5) → numpy fijado en 2.4.6.
- `scripts/setup_wsl.sh` ejecutado; lock completo en `requirements.lock.txt`.
- `scripts/check_env.py`: todo OK → torch 2.14.0+cu130 (CUDA 13.0, RTX 4060 8 GB),
  TensorRT 10.16.1.11 (FP16 e INT8 disponibles), OpenVINO 2026.4.0 (CPU i7-12700KF),
  NNCF 3.4.0, ultralytics 8.4.164, onnx 1.23.0. RAM visible en WSL: 23.5 GB.
- **Resultado: Fase 0 completada.**

## 2026-09-27 — Fase 1: preparación de datos

- `python -m src.data.prepare_dataset` y `python -m src.data.dataset_stats` (config `configs/data_prep.yaml`).
- 7 706 imágenes (720×720). Instancias conservadas: D00 1 555 · D10 68 · D20 2 021 · D40 3 187.
- Eliminadas 1 372 instancias (`results/tables/removed_labels.csv`): D44 1 062, D01 179, D43 57,
  D11 45, D50 28, D0w0 1 (probable errata de D00; se eliminó según la regla).
- Fondo: 3 921 imágenes sin anotaciones + 562 que quedaron sin cajas tras el filtrado (58 % del total).
- Partición estratificada por clase más rara presente (D10 > D00 > D20 > D40 > fondo), semilla 42:
  train 5 394 / val 1 155 / test 1 157. Listas en `data/splits/`.
- Test: D00 227 · D10 14 (9 imágenes) · D20 303 · D40 427 instancias.
- Calibración INT8: 500 imágenes de train (semilla 42), `configs/calib.yaml`; verificado que 0 están fuera de train.
- Datos YOLO en el disco de Linux: `~/datasets/rdd2022_india_yolo` (lectura ~4 GB/s).
- **Alerta:** D10 tiene solo 68 instancias; su AP en test (14 instancias) sería muy ruidoso.
- **Decisión (2026-09-27): se excluye D10 del estudio** → clases D00, D20, D40. Datos regenerados:
  - Conservadas: D00 1 555 · D20 2 021 · D40 3 187. Eliminadas 1 440 (incluye D10 68).
  - Fondo: 3 921 originales + 577 por filtrado (15 imágenes que solo tenían D10 pasan a fondo).
  - Estratos: D00 > D20 > D40 > fondo. train 5 394 / val 1 156 / test 1 156.
  - Test: D00 227 · D20 301 · D40 453 instancias. Calibración: 500 imágenes de train (492 instancias).
- **Resultado: Fase 1 completada.**

## 2026-09-27 — Fase 2: maestro (preparación)

- `configs/train.yaml`: imgsz 640, 150 épocas, patience 0 (sin parada temprana), SGD lr0 0.01,
  momentum 0.937, wd 5e-4, cos_lr, close_mosaic 10, aumentación por defecto, deterministic, seed 0.
- Prueba de velocidad YOLO11m (RTX 4060, 10 % de train, 1 época):
  batch 16 → 7.8 GB (satura), 0.047–0.10 s/img · batch 8 → 4.1 GB, 0.023 s/img.
  Decisión: maestro con batch 8 (nbs=64 → acumulación de gradiente equivalente). Alumnos se mantienen en 16.
- Prueba de humo (`--humo`, 1 época, 5 % de train) OK; pesos COCO en `runs/pretrained/`.
- Pendiente: entrenamiento completo (`bash scripts/train_teacher.sh`, ~6 h estimadas).

## 2026-09-28 — Cambio de diseño (paper v3) y Fase 1 rehecha

- CLAUDE.md actualizado al diseño v3 (P1 ΔmAP, P2 ECE, P3 KD+PTQ vs base+QAT; D1–D8).
- Se revierte la exclusión de D10 → 4 clases: D00=0, D10=1, D20=2, D40=3.
- Reglas de limpieza D1 aplicadas en orden (`python -m src.data.prepare_dataset`):

| Etiqueta original | Cajas | R1 otra clase | R2 recortadas | R3 <1 px | R4 duplicadas | R5 img corrupta | Conservadas |
|---|---|---|---|---|---|---|---|
| D40 | 3 187 | 0 | 0 | 0 | 0 | 0 | 3 187 |
| D20 | 2 021 | 0 | 0 | 0 | 0 | 0 | 2 021 |
| D00 | 1 555 | 0 | 0 | 0 | 0 | 0 | 1 555 |
| D44 | 1 062 | 1 062 | – | – | – | – | 0 |
| D01 | 179 | 179 | – | – | – | – | 0 |
| D10 | 68 | 0 | 0 | 0 | 0 | 0 | 68 |
| D43 | 57 | 57 | – | – | – | – | 0 |
| D11 | 45 | 45 | – | – | – | – | 0 |
| D50 | 28 | 28 | – | – | – | – | 0 |
| D0w0 | 1 | 1 | – | – | – | – | 0 |
| **Total** | **8 203** | **1 372** | **0** | **0** | **0** | **0** | **6 831** |

- a) Imágenes: raw 7 706 = data/yolo 7 706 (0 corruptas; 0 con tamaño XML ≠ tamaño real).
- c) Fondo: 4 483 imágenes (58.18 %) = 3 921 sin objetos en el XML + 562 cuyas únicas cajas eran de otras clases.
- d) "Filas" eliminadas en la versión anterior: eran **cajas** (líneas de etiqueta), no imágenes.
  v1 (4 clases, 2026-09-27): 1 372 cajas de clases fuera del estudio (D44, D01, D43, D11, D50, D0w0),
  las mismas que hoy elimina la regla 1. v2 (3 clases): 1 440 = esas 1 372 + las 68 de D10.
  Ninguna imagen se eliminó en ninguna versión; las imágenes que quedaron sin cajas pasaron a fondo
  (562 en v1, 577 en v2).
- Partición: estratificación iterativa multietiqueta (iterative-stratification 0.1.9; 4 clases + fondo),
  semilla 42 → train 5 396 / val 1 144 / test 1 166 imágenes.
- Test: D00 227 · D10 **9** (9 imágenes) · D20 300 · D40 464 instancias. **Aviso: D10 < 10 en test.**
- Calibración: 500 imágenes de train estratificadas (iterstrat devolvió 503; se quitaron 3 de fondo
  al azar con semilla 42). Imágenes con D00 72 · D10 4 · D20 114 · D40 99. Verificado: 0 fuera de train.

## 2026-09-28 — Decisiones tras la auditoría (Fase 1 aprobada)

- Datos aprobados. Clases eliminadas por la regla 1 y su justificación: D01/D11 (juntas de
  construcción), D43/D44 (desgaste de pasos peatonales y de líneas) y D50 (tapas de alcantarilla)
  no son clases de la tarea estándar RDD (Maeda et al., 2018). "D0w0" (1 caja) se considera una
  etiqueta ambigua y se elimina sin corregirla a mano. El 58.2 % de fondo se conserva sin submuestrear.
- D10: se mantiene la partición con semilla 42 (9 instancias en test). mAP de 4 clases = métrica
  principal; mAP sin D10 = análisis de sensibilidad.
- Calibrador INT8 (D4): exportador de Ultralytics 8.4.164 tal cual → MinMax, sigmoide de la cabeza
  en FP32, convoluciones de salida y DFL en FP16. Se reportará `results/tables/engine_layers.csv`.
- Caché de entrenamiento: `cache='disk'` para todas las corridas (D2).
- D7 reemplazada por el plan estadístico preespecificado (H1a, H1b, H2a, H2b, H3; Holm; δ = 0.010).

## 2026-09-28 — Fase 1b: pruebas de tiempo

- `bash scripts/fase1b_tiempos.sh` (config `configs/timing.yaml`), batch 8, imgsz 640, cache disk.
- Maestro YOLO11m, AMP vs FP32 (tiempo por época de entrenamiento; validación ~12 s aparte):

| Maestro YOLO11m | Época 1 | Épocas 2–3 | VRAM máx. | NaN |
|---|---|---|---|---|
| amp=True | 114 s | 111–112 s (+11–12 s val) | 4.2 GB | no |
| amp=False (FP32) | 947 s | ~810 s (1.2 s/lote, época 2 al 79 %) | 8.07 GB (satura 8 GB) | no |

  FP32 llena la VRAM de la RTX 4060 y desborda a memoria compartida del sistema: ~7× más lento.
  La prueba FP32 se detuvo en la época 2 (79 %) por decisión del usuario; los datos salen del log.
- **Decisión (usuario, 2026-09-28): amp=True para todos los modelos (maestro, alumnos y QAT).**
- Alumno YOLO11n (amp=True): 32.5 s/época (28.5 train + 4.0 val), VRAM 1.47 GB, sin NaN.
- Maestro YOLO11m (amp=True), promedio épocas 2–3: 123.4 s/época (111.8 train + 11.6 val), VRAM 4.21 GB, sin NaN.
- Pérdidas (results.csv) finitas y decrecientes en ambos casos; mAP50 de val tras 3 épocas: maestro 0.062, alumno 0.067.
- Forward del maestro (eval, autocast, batch 8): 52.7 ms/lote × 675 lotes = +36 s por época KD
  (sin la CWD ni los adaptadores) → época KD estimada ≈ 69 s.
- Estimación total (AMP; `results/tables/timing.csv`):

| Bloque | Épocas | Horas |
|---|---|---|
| Maestro (cota superior, patience=50) | 1 × 300 | 10.3 |
| Alumnos base | 3 × 300 | 8.1 |
| Alumnos KD | 3 × 300 | 17.0 |
| Piloto λ | 3 × 50 | 2.8 |
| QAT (factor 1.5 supuesto) | 6 × 30 | 2.4 |
| **Total** | | **40.7 h (~1.7 días de GPU)** |

- **Resultado: Fase 1b completada.**

## 2026-09-29 — Fase 2: maestro listo para lanzar

- Corrida previa `runs/teacher/teacher_s0` (diseño v2: 3 clases, 150 épocas, cos_lr; detenida en la
  época 47) movida a `runs/obsoletos/teacher_s0_v2_3clases_150ep`. No se usa en el diseño v3.
- Prueba de humo (`bash scripts/train_teacher.sh --humo`) con la config D2 OK: 4 clases, amp, batch 8,
  nbs 64, patience 50, cache disk, SGD, cos_lr=False, seed 0.
- Pendiente: entrenamiento completo (`bash scripts/train_teacher.sh`, ≤ ~10 h).

## 2026-09-30 — Fase 2: maestro entrenado

- `runs/teacher/teacher_s0` (log `runs/logs/teacher_20260929_231734.log`). 198 épocas en 6.2 h.
- EarlyStopping (patience=50): **best.pt = época 148** (fitness máx.), **last.pt = época 198**.
- Métricas de VALIDACIÓN (no test) de best.pt, validación final de Ultralytics:

| Clase | Inst. val | P | R | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|---|
| **Todas** | 1 026 | 0.362 | 0.389 | **0.344** | **0.148** |
| D00 | 231 | 0.356 | 0.346 | 0.309 | 0.135 |
| D10 | 9 | 0.104 | 0.111 | 0.035 | 0.013 |
| D20 | 304 | 0.538 | 0.622 | 0.597 | 0.270 |
| D40 | 482 | 0.451 | 0.477 | 0.435 | 0.174 |

- last.pt (época 198): mAP@0.5 0.346, mAP@0.5:0.95 0.143. Máx. mAP@0.5 de la corrida: 0.351 (época 159).
- Revalidación de best.pt en val (batch 8, conf 0.001, iou 0.7; el 0.344 de arriba usa batch 16 de
  la validación final del entrenamiento, diferencia por el relleno de imágenes):
  AP@0.5 D00 0.3114 · D10 0.0342 · D20 0.5971 · D40 0.4372 → mAP@0.5 (4 clases) 0.3450, mAP@0.5:0.95 0.1471.
  **Sin D10 (3 clases): mAP@0.5 = 0.4486, mAP@0.5:0.95 = 0.1921.**
- Convergencia (`results/figures/teacher_convergence.png`): fitness máx. en la época 148 (0.1676);
  máx. mAP@0.5:0.95 también en la 148 (0.1479); máx. mAP@0.5 en la 159 (0.3510).
  Regresión lineal de la mAP de val por época:

| Tramo | Métrica | Cambio por 50 épocas | IC 95 % | p |
|---|---|---|---|---|
| 149–198 (tras best) | mAP@0.5 | −0.0026 | [−0.0053, +0.0002] | 0.075 |
| 149–198 (tras best) | mAP@0.5:0.95 | −0.0019 | [−0.0025, −0.0013] | <0.001 |
| 99–148 (antes de best) | mAP@0.5 | +0.0279 | [+0.0238, +0.0319] | <0.001 |
| 99–148 (antes de best) | mAP@0.5:0.95 | +0.0127 | [+0.0113, +0.0141] | <0.001 |

  Conclusión: el maestro convergió; en las últimas 50 épocas la mAP@0.5 está plana y la
  mAP@0.5:0.95 baja levemente, así que la parada temprana no cortó un aprendizaje en curso.
- Observaciones: la mAP se estabiliza desde ~época 120; la pérdida de caja en val sube desde
  ~época 90 (1.93 → 2.09) mientras la de train sigue bajando → sobreajuste en localización, que la
  parada temprana corta. La precisión global oscila (0.3–0.6) por D10 (9 instancias en val).
- **Resultado: Fase 2 completada.** Maestro para KD: `runs/teacher/teacher_s0/weights/best.pt`.

## 2026-09-30 — Fase 4a: implementación CWD y prueba de humo

- Implementación propia (`src/train/cwd.py`) como subclase de `DetectionTrainer`. Fórmula CWD adaptada de
  MMRazor (`open-mmlab/mmrazor`, `mmrazor/models/losses/cwd.py`, commit 0919f692878fe0037ff94bc90e46eed721c64a39).
  Ultralytics 8.4.164 trae `DistillationModel` propio, pero usa L2 ponderada por confianza con proyector MLP
  (no CWD), así que no se usa.
- Detalles: características = entradas de Detect (P3, P4, P5) tomadas de `preds["feats"]` (sin hooks);
  maestro `runs/teacher/teacher_s0/weights/best.pt` congelado, eval(), no_grad, autocast, mismo batch aumentado;
  adaptadores conv 1×1 64/128/256 → 256/512/512 canales (214 272 parámetros) fuera del modelo (grupo propio del
  optimizador SGD con las mismas lr/momentum/weight decay; recorte de gradiente norma 10 como el modelo;
  inicializados con `fork_rng` para no alterar la secuencia aleatoria respecto del alumno base).
  L = L_det + λ·L_CWD·batch (L_det de Ultralytics ya viene × batch); τ = 1.0; L_CWD sumada en los 3 niveles.
  `results.csv` registra `train/cwd` (L_CWD sin λ) por época.
- Prueba de humo (`--kd on --seed 0 --humo`: λ = 1.0, 2 épocas, fraction 0.05) + verificación
  `python -m src.train.check_kd`:
  - L_CWD 1.4837 → 1.4432 (baja); L_det (box+cls+dfl) 16.61 → 14.84. Con λ = 1, L_CWD ≈ 9 % de L_det.
  - best.pt KD y base: 499 tensores con claves y formas idénticas; sin adaptadores ni maestro.
  - ONNX KD y base: 2 624 974 parámetros ambos.
- **Resultado: Fase 4a completada.** Listo para Fase 3 (`--kd off --seed N`) y piloto de λ (`--kd on --piloto λ`).

## 2026-10-01 — Fase 3: alumnos base (seeds 0, 1, 2)

- `for s in 0 1 2; do bash scripts/train_student.sh --kd off --seed $s; done` → `runs/students/student_base_s{0,1,2}`.
- 300 épocas cada uno (sin parada temprana), ~2.7 h por corrida, sin NaN.

| Corrida | best (época) | val mAP@0.5 / 0.5:0.95 en best | last (300) mAP@0.5 / 0.5:0.95 | Pendiente últ. 50 ép. (mAP@0.5 / 0.5:0.95, por 50 ép.) |
|---|---|---|---|---|
| base_s0 | 202 | 0.3670 / 0.1514 | 0.3415 / 0.1384 | −0.0023 / −0.0027 |
| base_s1 | 238 | 0.3332 / 0.1417 | 0.3321 / 0.1364 | −0.0006 / −0.0029 |
| base_s2 | 114 | 0.3244 / 0.1403 | 0.3294 / 0.1363 | −0.0021 / −0.0000 |

- Todos convergieron (mAP plana o bajando levemente en las últimas 50 épocas).
- Revalidación en val de best.pt (batch 8, conf 0.001, iou 0.7), comparada con el maestro:

| Modelo | mAP@0.5 | mAP@0.5:0.95 | D00 | D10 | D20 | D40 | mAP@0.5 sin D10 |
|---|---|---|---|---|---|---|---|
| Maestro YOLO11m | 0.3450 | 0.1471 | 0.3114 | 0.0342 | 0.5971 | 0.4372 | 0.4486 |
| base_s0 | 0.3657 | 0.1510 | 0.3439 | 0.1492 | 0.5684 | 0.4012 | 0.4378 |
| base_s1 | 0.3339 | 0.1417 | 0.3264 | 0.0342 | 0.5764 | 0.3985 | 0.4338 |
| base_s2 | 0.3248 | 0.1408 | 0.2491 | 0.0972 | 0.5619 | 0.3910 | 0.4007 |
| Alumnos base (media ± d.e.) | 0.3415 ± 0.0215 | 0.1445 ± 0.0057 | | | | 0.3969 ± 0.0053 | 0.4241 ± 0.0203 |

- **Alerta:** el maestro apenas supera a los alumnos. En mAP@0.5 global (4 clases) base_s0 lo supera, por el AP
  de D10 (9 instancias en val: 0.149 vs 0.034). Sin D10 el maestro gana por ~2.5 puntos (0.449 vs 0.424) y en
  D40 por ~4 puntos (0.437 vs 0.397); en mAP@0.5:0.95 la diferencia es de ~0.3 puntos. Pendiente de decisión
  del usuario antes del piloto de λ (Fase 4b).
- **Resultado: Fase 3 completada** (corridas válidas; la alerta afecta a la Fase 4b, no a estas corridas).
