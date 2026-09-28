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
