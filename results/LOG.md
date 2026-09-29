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
