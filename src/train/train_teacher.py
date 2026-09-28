"""Fase 2: entrena el maestro YOLO11m (preentrenado en COCO) sobre RDD2022-India.

Uso (desde la raíz del proyecto, en WSL):
    python -m src.train.train_teacher            # entrenamiento completo
    python -m src.train.train_teacher --humo     # prueba rápida (1 época, 5 % de train)
"""

from __future__ import annotations

import argparse

from ultralytics import YOLO

from src.train.common import argumentos_entrenamiento, callback_metadatos, ruta_pesos_preentrenados
from src.utils.config import cargar_yaml
from src.utils.repro import fijar_semillas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--humo", action="store_true", help="prueba de humo (ver sección 'humo' del yaml)")
    args = ap.parse_args()

    cfg = cargar_yaml(args.config)
    m = cfg["maestro"]
    fijar_semillas(m["seed"])

    model = YOLO(ruta_pesos_preentrenados(m["modelo"]))
    model.add_callback(
        "on_pretrain_routine_start",
        callback_metadatos([args.config, cfg["comun"]["data"]], {"rol": "teacher", "humo": args.humo}),
    )
    kwargs = argumentos_entrenamiento(cfg, "maestro", m["seed"], m["name"], args.humo)
    model.train(**kwargs)
    print(f"\nPesos: {model.trainer.best}")


if __name__ == "__main__":
    main()
