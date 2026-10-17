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

## 2026-10-01 — Exploración: maestro YOLO11l (sin cambio de diseño)

- Motivo: el maestro YOLO11m apenas supera a los alumnos base (ver Fase 3).
- `scripts/exploracion/explora_teacher_l.py`: YOLO11l (COCO) con la config D2 (amp, batch 8, cache disk),
  3 épocas completas; salida en `runs/exploracion/teacher_l_3ep`, log `runs/logs/explora_teacher_l.log`.

| | YOLO11m (Fase 1b) | YOLO11l (exploración) |
|---|---|---|
| Parámetros / GFLOPs | 20.0 M / 67.8 | 25.3 M / 86.8 |
| s/época train + val (sin la 1.ª) | 111.8 + 11.6 = 123 s | 129.8 + 13.0 = 143 s |
| VRAM máx. (batch 8, AMP) | 4.2 GB | 5.3 GB |
| NaN en 3 épocas | no | no |
| Forward maestro (eval, autocast, batch 8) | 52.7 ms → +36 s/época KD | 66.3 ms → +45 s/época KD |
| Canales de entrada de Detect | 256/512/512 | 256/512/512 (mismos adaptadores) |

- Costo estimado de cambiar a YOLO11l: maestro ≤ 11.9 h (300 ép.; ~8 h si para como el 11m en ~200),
  KD +2.3 h (3 × 300) y piloto +0.4 h → ~11–15 h de GPU adicionales.
- Pendiente de decisión del usuario.

## 2026-10-01 — Fase 2b: segundo maestro YOLO11l (preparación)

- Decisión del usuario: entrenar también YOLO11l como maestro y conservar el YOLO11m (`runs/teacher/teacher_s0`).
- `configs/train.yaml`: nuevo bloque `maestro_l` (yolo11l.pt, seed 0, patience 50, `teacher_l_s0`), misma config D2.
  `src/train/train_teacher.py --rol maestro_l`.
- Regla de selección fijada ANTES del resultado (`configs/kd.yaml`, seleccion_maestro): revalidar best.pt de
  ambos en VALIDACIÓN (imgsz 640, batch 8, conf 0.001, iou 0.7); YOLO11l solo si su mAP@0.5:0.95 ≥ la del
  YOLO11m (0.1471) + 0.010; si no, YOLO11m.
- Prueba de humo `bash scripts/train_teacher.sh --rol maestro_l --humo` OK (args: yolo11l, batch 8, amp,
  patience 50, cache disk).
- Pendiente: entrenamiento completo (~8–12 h).

## 2026-10-02 — Fase 2b: maestro YOLO11l entrenado y selección de maestro

- `bash scripts/train_teacher.sh --rol maestro_l` → `runs/teacher/teacher_l_s0`. 155 épocas en 6.1 h, sin NaN.
  EarlyStopping (patience 50): best.pt = época 105, last.pt = época 155.

| | YOLO11m (teacher_s0) | YOLO11l (teacher_l_s0) |
|---|---|---|
| Épocas / horas | 198 / 6.2 h | 155 / 6.1 h |
| Época best / last | 148 / 198 | 105 / 155 |
| Máx. mAP@0.5 en la corrida | 0.3510 (ép. 159) | 0.3428 (ép. 133) |

- Revalidación de best.pt en VALIDACIÓN (imgsz 640, batch 8, conf 0.001, iou 0.7):

| Maestro | mAP@0.5 | **mAP@0.5:0.95** | D00 | D10 | D20 | D40 | mAP@0.5 sin D10 | mAP@0.5:0.95 sin D10 | P | R |
|---|---|---|---|---|---|---|---|---|---|---|
| YOLO11m | 0.3450 | **0.1471** | 0.3114 | 0.0342 | 0.5971 | 0.4372 | 0.4486 | 0.1921 | 0.363 | 0.389 |
| YOLO11l | 0.3389 | **0.1444** | 0.2839 | 0.0731 | 0.5727 | 0.4261 | 0.4276 | 0.1814 | 0.437 | 0.342 |

- **Regla de selección (fijada el 2026-10-01): YOLO11l necesitaba mAP@0.5:0.95 ≥ 0.1571; obtuvo 0.1444
  (−0.0027 frente al 11m). → Se usa YOLO11m** (`configs/kd.yaml` sin cambios de ruta). Ambos se conservan.
- Curvas (`results/figures/teachers_m_vs_l.png`): ambos maestros siguen trayectorias casi idénticas época a
  época hasta ~120 (misma semilla y orden de datos); el 11l tiene un pico en la época 105 y luego se mantiene
  en ~0.14 de mAP@0.5:0.95, por debajo del 11m. Regresión épocas 106–155 del 11l: +0.0024 por 50 ép.
  (IC 95 % ±0.0019), arrastrada por la caída inmediatamente posterior al pico; aun extrapolada a 300 épocas
  quedaría lejos del umbral (0.1571).
- Conclusión: más capacidad (+26 % de parámetros) no mejora al maestro en RDD2022-India; refuerza que el
  techo lo pone el conjunto de datos, no el tamaño del modelo. Se documenta en Limitaciones.
- **Resultado: Fase 2b completada.**

## 2026-10-03 — Exploración: ¿la PTQ degrada a los alumnos base? (en VALIDACIÓN, no test)

- `PYTHONPATH=. python scripts/exploracion/ptq_check.py` (config `configs/exploracion_ptq.yaml`); tabla
  `results/tables/exploracion_ptq_val.csv`; engines en `runs/exploracion/ptq_check/` (corridas originales intactas).
  Exportación según D4 (TensorRT 10.16, imgsz 640, batch 1, estático, simplify; INT8 con el exportador de
  Ultralytics y `configs/calib.yaml`). Evaluación en val, batch 1, conf 0.001, iou 0.7.
- Se usó validación a propósito: el split test queda reservado para la evaluación final (D6/D7).

| Alumno | Variante | mAP@0.5 | mAP@0.5:0.95 | AP@0.5 D40 | mAP@0.5 sin D10 | P | R | MB |
|---|---|---|---|---|---|---|---|---|
| base_s0 | PyTorch FP32 (rect) | 0.3659 | 0.1511 | 0.4011 | 0.4381 | 0.491 | 0.353 | 5.3 |
| base_s0 | PyTorch FP32 (cuadrado) | 0.3449 | 0.1414 | 0.3880 | 0.4218 | 0.381 | 0.389 | 5.3 |
| base_s0 | TensorRT FP32 | 0.3449 | 0.1415 | 0.3879 | 0.4218 | 0.382 | 0.388 | 12.8 |
| base_s0 | TensorRT FP16 | 0.3604 | 0.1439 | 0.3931 | 0.4232 | 0.383 | 0.391 | 7.4 |
| base_s0 | TensorRT INT8 | 0.3061 | 0.1221 | 0.3533 | 0.3755 | 0.419 | 0.349 | 4.5 |
| base_s1 | PyTorch FP32 (rect) | 0.3337 | 0.1416 | 0.3981 | 0.4336 | 0.410 | 0.356 | 5.3 |
| base_s1 | PyTorch FP32 (cuadrado) | 0.3231 | 0.1315 | 0.4009 | 0.4226 | 0.399 | 0.347 | 5.3 |
| base_s1 | TensorRT FP32 | 0.3232 | 0.1315 | 0.4008 | 0.4227 | 0.399 | 0.347 | 12.9 |
| base_s1 | TensorRT FP16 | 0.3230 | 0.1315 | 0.4023 | 0.4225 | 0.393 | 0.350 | 7.4 |
| base_s1 | TensorRT INT8 | 0.2928 | 0.1125 | 0.3532 | 0.3696 | 0.411 | 0.318 | 4.5 |
| base_s2 | PyTorch FP32 (rect) | 0.3248 | 0.1407 | 0.3911 | 0.4007 | 0.359 | 0.396 | 5.3 |
| base_s2 | PyTorch FP32 (cuadrado) | 0.3031 | 0.1263 | 0.3984 | 0.3793 | 0.296 | 0.357 | 5.3 |
| base_s2 | TensorRT FP32 | 0.3032 | 0.1264 | 0.3987 | 0.3794 | 0.281 | 0.350 | 12.9 |
| base_s2 | TensorRT FP16 | 0.3028 | 0.1261 | 0.3979 | 0.3790 | 0.297 | 0.358 | 7.6 |
| base_s2 | TensorRT INT8 | 0.2795 | 0.1163 | 0.3509 | 0.3448 | 0.534 | 0.301 | 4.5 |

- Pérdida respecto de TensorRT FP32 (Δ = FP32 − cuantizado; media ± d.e. de 3 semillas):

| Métrica | FP16 | INT8 | INT8 relativo |
|---|---|---|---|
| mAP@0.5 | −0.0050 ± 0.0091 | **+0.0310 ± 0.0076** | −9.5 % |
| mAP@0.5:0.95 | −0.0007 ± 0.0015 | **+0.0162 ± 0.0052** | −12.1 % |
| AP@0.5 D40 | −0.0020 ± 0.0030 | **+0.0433 ± 0.0076** | −10.9 % |
| mAP@0.5 sin D10 | −0.0003 ± 0.0010 | **+0.0447 ± 0.0094** | −10.9 % |
| Recall | −0.0040 ± 0.0029 | +0.0394 ± 0.0099 | −10.9 % |

- **Conclusión: la PTQ INT8 degrada a los 3 alumnos base de forma consistente (3/3 semillas, todas las
  métricas): ~3 puntos de mAP@0.5 (−9.5 %) y ~4.3 puntos en D40.** FP16 no pierde precisión (|Δ| ≤ 0.002 sin
  D10; el +1.5 de FP16 en base_s0 viene de D10, 9 instancias). El fundamento de P1 se confirma.
- INT8 sube la precisión y baja el recall (base_s2: P 0.28 → 0.53): la cuantización desplaza las confianzas,
  lo que apunta a efectos en la calibración (P2, ECE).
- **Hallazgo metodológico:** PyTorch FP32 con la validación por defecto de Ultralytics (rect=True, letterbox
  rectangular) da 1–2 puntos más de mAP@0.5 que con letterbox cuadrado; PyTorch con rect=False reproduce
  TensorRT FP32 (|Δ| ≤ 0.0001). Un engine estático de 640×640 siempre usa letterbox cuadrado. Para que ΔmAP
  mida solo la cuantización, la referencia FP32 debe evaluarse con el mismo preprocesado (rect=False o el
  engine TensorRT FP32). Pendiente de decisión del usuario para D6.

## 2026-10-15 — Fase 4b (parte 1): piloto de λ preparado

- `scripts/pilot_lambda.sh`: entrena `pilot_kd_l{λ}_s0` para λ ∈ {0.5, 1.0, 2.0} (seed 0, 50 épocas, config D2,
  maestro YOLO11m) en `runs/pilot/` y luego corre `python -m src.train.select_lambda`.
- Regla fijada antes de ver resultados (`configs/kd.yaml`, piloto): se elige el λ con mayor mAP@0.5:0.95 de
  best.pt revalidado en VALIDACIÓN (imgsz 640, batch 8, conf 0.001, iou 0.7). `select_lambda` también reporta
  por λ: L_CWD de la época 1 a la última, si baja, el peso de λ·L_CWD sobre L_det y NaN. No modifica kd.yaml.
- Prueba de humo del flujo completo (`--piloto λ --humo`, 2 épocas, 5 % de train, + `select_lambda --humo`) OK.
  L_CWD baja con los 3 λ (≈1.48 → 1.43–1.45); peso de λ·L_CWD sobre L_det: 5 % (λ=0.5), 10 % (λ=1), 20 % (λ=2).
- Pendiente: piloto completo (~3 h).

## 2026-10-16 — Fase 4b (parte 1): piloto de λ completado

- `bash scripts/pilot_lambda.sh` → `runs/pilot/pilot_kd_l{0.5,1.0,2.0}_s0` (50 épocas, ~63 min cada uno;
  terminó 02:57). Un reinicio del PC a las 03:58 no afectó las corridas. Selección:
  `runs/logs/pilot_lambda_seleccion_20261016_025713.log`, `results/tables/pilot_lambda.csv`,
  `results/figures/pilot_lambda.png`.

| λ | L_CWD época 1 → 50 | ¿Baja? | λ·L_CWD / L_det (final) | NaN | val mAP@0.5 | **val mAP@0.5:0.95** | AP@0.5 D40 | mAP@0.5 sin D10 |
|---|---|---|---|---|---|---|---|---|
| 0.5 | 1.456 → 0.467 | sí | 4.3 % | no | 0.3121 | 0.1341 | 0.3248 | 0.3726 |
| **1.0** | 1.421 → 0.409 | sí | 7.6 % | no | **0.3279** | **0.1437** | **0.3747** | 0.3969 |
| 2.0 | 1.366 → 0.363 | sí | 13.8 % | no | 0.3027 | 0.1302 | 0.3744 | 0.3991 |

- L_CWD baja con los 3 λ (−68 % a −73 %); el pequeño salto en la época 41 coincide con el cierre del mosaico
  (close_mosaic=10), esperado. Escala razonable: λ·L_CWD = 4–14 % de L_det.
- Según la regla fijada (mayor mAP@0.5:0.95 de validación): **λ = 1.0** (0.1437; +0.010 sobre λ=0.5,
  +0.014 sobre λ=2.0). λ=1.0 es el valor central de la rejilla (no está en un extremo).
- Advertencia: una sola semilla y 50 épocas; la mAP seguía subiendo en la época 50 para los 3 λ.
- Pendiente: confirmación del usuario para fijar `lambda: 1.0` en `configs/kd.yaml`.

## 2026-10-16 — Fase 4b (parte 2): λ fijado y alumnos KD listos

- **Decisión del usuario: λ = 1.0** (`configs/kd.yaml`), según la regla del piloto.
- Nuevo `--reanudar` en `src/train/train_student.py` (retoma desde `weights/last.pt`). Para el alumno KD, los
  adaptadores se guardan en `weights/kd_adaptadores_last.pt` en cada checkpoint (no entran en best.pt/last.pt)
  y se restauran al reanudar. Prueba: humo KD de 2 épocas, proceso matado tras la época 1 y reanudado → retoma en
  la época 2, adaptadores restaurados, termina OK. Una corrida reanudada no es idéntica bit a bit a una sin cortes
  (estado aleatorio del cargador de datos): si se usa en una corrida final, se anota aquí.
- Pendiente: `for s in 0 1 2; do bash scripts/train_student.sh --kd on --seed $s; done` (~17 h).

## 2026-10-17 — Repositorio preparado para publicación (GitHub público)

- `.gitignore` ampliado (pesos, exportaciones, cachés `.npy`, `.zip`, `.env`, editores) y `paper/` excluida
  (borrador no publicado).
- Historial reescrito con `git filter-branch` para eliminar `paper/` de todos los commits (copia de seguridad
  previa del `.git` fuera del repositorio). Verificado: 0 commits con `paper/`, 0 objetos `.docx`.
- Los identificadores de commit cambiaron. Los `entorno.json` de `runs/` guardan los antiguos; equivalencias:

| Commit antiguo | Commit nuevo | Mensaje |
|---|---|---|
| 6a788e7 | 8467def | Fase 0: estructura del proyecto, requirements y verificación de entorno |
| c218c81 | 4537746 | Fase 0 completada: entorno WSL2 + Python 3.13 verificado, lock de dependencias |
| b5c294c | b98dbf8 | Fase 1: preparación de datos RDD2022-India (YOLO, partición estratificada, calibración) |
| ed4f531 | b74a81e | Fase 1: excluir D10 (68 instancias) y regenerar datos con 3 clases |
| f7a0af0 | 2e4a9ef | Fase 2: script de entrenamiento del maestro YOLO11m y configs/train.yaml |
| 62de3a8 | 334f2cf | Diseño v3 en CLAUDE.md y Fase 1 rehecha: 4 clases, reglas D1, auditoría, estratificación iterativa |
| 126aec7 | c041af4 | Diseño: calibrador de Ultralytics, cache disk, D7 preespecificada; Fase 1 aprobada |
| 42b1813 | 4d8fc25 | Fase 1b: pruebas de tiempo (AMP vs FP32, alumno, forward del maestro) y estimación total |
| 8032b58 | 7ea415c | Fase 2: maestro listo (humo con config D2), corrida v2 archivada |
| 464b6ed | 2c9902b | Fase 2 (resultados) y Fase 4a: CWD propia, script de alumnos, verificación de humo |
| 4da7577 | 9d26c23 | LOG: mAP exacta sin D10 y análisis de convergencia del maestro |
| f4a1a78 | ab950e0 | Fase 3: alumnos base s0-s2 entrenados; alerta maestro vs alumnos |
| cd99f24 | c4a8f9d | Exploración: maestro YOLO11l (3 épocas, tiempos, VRAM, costo KD) |
| 9af54ed | e288c9d | Fase 2b: segundo maestro YOLO11l (config, --rol, regla de selección, humo OK) |
| f872c72 | 22dac85 | Fase 2b: YOLO11l entrenado; regla de selección → se usa YOLO11m |
| b8a0c4e | 809e374 | Exploración PTQ en val: INT8 degrada ~3 pts mAP50 (3/3 semillas); rect vs cuadrado |
| 1fd858d | 85380f7 | Fase 4b: piloto de λ (script, regla de selección, select_lambda) y humo del flujo |
| 52d1514 | 924e15a | Fase 4b: piloto de λ completado (propuesta λ=1.0) |
| 0162042 | d3bddc6 | Fase 4b: λ=1.0 en kd.yaml; --reanudar con restauración de adaptadores |
| eab2e7c | 9922569 | README: guía del proyecto, fases, scripts y resultados |
| 30d0e57 | f018aa4 | gitignore: pesos, exportaciones, cachés .npy, zip, .env y editores |
| 35ac428 | 747b716 | Excluir paper/ del repositorio (borrador no publicado) |

## 2026-10-17 — Fase 4b (parte 2): alumnos KD entrenados (λ = 1.0)

- `for s in 0 1 2; do bash scripts/train_student.sh --kd on --seed $s; done` → `runs/students/student_kd_s{0,1,2}`.
  300 épocas cada uno, 6.2–6.4 h por corrida (estimado: 5.7 h), sin NaN, **sin reanudaciones** (`resume: false`).
- best.pt: épocas 157 (s0), 274 (s1), 113 (s2). mAP de val plana o bajando levemente en las últimas 50 épocas.
- L_CWD: 1.42 → 0.41 (época 50) → 0.38 (época 300) en las 3 semillas; al final λ·L_CWD ≈ 11.4 % de L_det.
  En las últimas 100 épocas L_CWD sube levemente (+0.015 por 100 épocas), en parte por el cierre del mosaico
  en la época 291 (mismo salto visto en el piloto).
- `python -m src.train.check_kd` para cada semilla (KD vs. base de la misma semilla): **TODO OK** en las 3.
  best.pt KD con 499 tensores idénticos al base (sin adaptadores ni maestro); ONNX de 2 624 974 parámetros en ambos.
- Revalidación en VALIDACIÓN (imgsz 640, batch 8, conf 0.001, iou 0.7):

| Modelo | mAP@0.5 | mAP@0.5:0.95 | D00 | D10 | D20 | D40 | mAP@0.5 sin D10 | mAP@0.5:0.95 sin D10 |
|---|---|---|---|---|---|---|---|---|
| Maestro YOLO11m | 0.3450 | 0.1471 | 0.3114 | 0.0342 | 0.5971 | 0.4372 | 0.4486 | 0.1921 |
| kd_s0 | 0.3378 | 0.1478 | 0.3012 | 0.0671 | 0.5863 | 0.3966 | 0.4280 | 0.1859 |
| kd_s1 | 0.3493 | 0.1458 | 0.3378 | 0.0445 | 0.5868 | 0.4282 | 0.4509 | 0.1857 |
| kd_s2 | 0.3529 | 0.1465 | 0.3096 | 0.1142 | 0.5802 | 0.4077 | 0.4325 | 0.1851 |

| Métrica (val, media ± d.e.) | Alumnos base | Alumnos KD | KD − base | KD − base por semilla (s0, s1, s2) |
|---|---|---|---|---|
| mAP@0.5 | 0.3415 ± 0.0215 | 0.3467 ± 0.0079 | +0.0052 | −0.028, +0.015, +0.028 |
| mAP@0.5:0.95 | 0.1445 ± 0.0056 | 0.1467 ± 0.0010 | +0.0022 | −0.003, +0.004, +0.006 |
| AP@0.5 D40 | 0.3969 ± 0.0053 | 0.4108 ± 0.0160 | +0.0139 | −0.005, +0.030, +0.017 |
| mAP@0.5 sin D10 | 0.4241 ± 0.0204 | 0.4371 ± 0.0121 | +0.0130 | −0.010, +0.017, +0.032 |
| mAP@0.5:0.95 sin D10 | 0.1805 ± 0.0081 | 0.1856 ± 0.0004 | +0.0051 | −0.001, +0.002, +0.014 |

- Lectura (descriptiva, en val; no es un contraste del paper): la KD mejora levemente al alumno en FP32 (2/3
  semillas; s0 empeora en mAP@0.5 por D10) y, sobre todo, **reduce la variabilidad entre semillas** (d.e. de
  mAP@0.5:0.95 0.0010 vs 0.0056). Los alumnos KD alcanzan al maestro en mAP@0.5:0.95 (0.1467 vs 0.1471).
- **Resultado: Fase 4b completada.**

## 2026-10-17 — Decisión D6 (preprocesado) y complementos de la exploración PTQ

- **Confirmación: la exploración PTQ (2026-10-03) usó solo VALIDACIÓN, nunca test.** Evidencia:
  `configs/exploracion_ptq.yaml` (`split: val`); en `runs/logs/exploracion_ptq_check*.log` hay 11 lecturas de
  `labels/val` (evaluaciones), 6 de `labels/train` (cargadores de calibración INT8 sobre `calib.txt`, imágenes
  de train) y 0 menciones de `test`.
- Precisión y recall de esa exploración (punto de máximo F1 de Ultralytics; Δ = FP32 − cuantizado, TensorRT):

| Alumno | P FP32 | P FP16 | P INT8 | R FP32 | R FP16 | R INT8 |
|---|---|---|---|---|---|---|
| base_s0 | 0.382 | 0.383 | 0.419 | 0.388 | 0.391 | 0.349 |
| base_s1 | 0.399 | 0.393 | 0.411 | 0.347 | 0.350 | 0.318 |
| base_s2 | 0.281 | 0.297 | 0.534 | 0.350 | 0.358 | 0.301 |

| Δ (FP32 − cuantizado), media ± d.e. | FP16 | INT8 |
|---|---|---|
| Precisión | −0.004 ± 0.011 | −0.101 ± 0.133 (sube en 3/3 semillas) |
| Recall | −0.004 ± 0.003 | +0.039 ± 0.010 (baja en 3/3 semillas, −10.9 %) |

  INT8 vuelve al modelo más conservador (más precisión, menos recall). La precisión es inestable porque es un
  solo punto de operación (umbral de máximo F1 promediado sobre clases, sensible a D10).
- **Decisión del usuario (D6):** todos los modelos, incluido el .pt FP32, se evalúan con letterbox cuadrado
  640×640 (`rect=False`) y batch 1. ΔmAP dentro del mismo motor: GPU = TRT FP32 − TRT {FP16, INT8, QAT INT8};
  CPU = OV FP32 − OV INT8. El .pt FP32 (rect=False) es control de consistencia frente a TRT FP32
  (|Δ mAP@0.5| < 0.002; en la exploración fue ≤ 0.0001). ECE y bootstrap usan esas mismas predicciones.
