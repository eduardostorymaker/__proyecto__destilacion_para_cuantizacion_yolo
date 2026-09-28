# Bitácora de experimentos

## 2026-09-27 — Fase 0: entorno y estructura

- Entorno elegido: WSL2 (Ubuntu 26.04, 24 GB RAM), Python 3.13 en venv `~/.venvs/yolokd` (uv).
  Se descartó Python 3.14 porque TensorRT 10 no tiene versión para él y TensorRT 11 cambia el
  método INT8 (Q/DQ con ModelOpt, capas mixtas en float).
- Versiones fijadas en `requirements.txt`: ultralytics 8.4.164, torch 2.14.0+cu130,
  TensorRT 10.16.1.11 (rama 10, PTQ con IInt8EntropyCalibrator2), OpenVINO 2026.4.0, NNCF 3.4.0.
- Dataset movido de `dataset/` a `data/raw/` (RDD2022/India ya descomprimido:
  train = 7 706 imágenes + 7 706 XML; test = 1 959 imágenes sin etiquetas).
- Pendiente: ejecutar `scripts/setup_wsl.sh` y `scripts/check_env.py` en WSL.
