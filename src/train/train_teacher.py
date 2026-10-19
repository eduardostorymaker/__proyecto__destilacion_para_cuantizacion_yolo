"""Fase 2: entrena un maestro (preentrenado en COCO) sobre RDD2022-India.

Hay dos maestros candidatos en configs/train.yaml: `maestro` (YOLO11m → teacher_s0) y
`maestro_l` (YOLO11l → teacher_l_s0). Se elige uno con la regla de configs/kd.yaml.

Uso (desde la raíz del proyecto, en WSL):
    python -m src.train.train_teacher                      # YOLO11m, entrenamiento completo
    python -m src.train.train_teacher --rol maestro_l      # YOLO11l, entrenamiento completo
    python -m src.train.train_teacher --rol maestro_l --humo   # prueba rápida (1 época, 5 % de train)
    python -m src.train.train_teacher --dataset japan          # réplica en Japón (configs/japan/train.yaml)
"""

from __future__ import annotations

import argparse

from ultralytics import YOLO

from src.train.common import argumentos_entrenamiento, callback_metadatos, ruta_pesos_preentrenados
from src.utils.config import cargar_yaml
from src.utils.dataset import agregar_argumento, ruta_config
from src.utils.repro import fijar_semillas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="por defecto, train.yaml del dataset")
    agregar_argumento(ap)
    ap.add_argument("--rol", default="maestro", choices=["maestro", "maestro_l"], help="bloque de train.yaml")
    ap.add_argument("--humo", action="store_true", help="prueba de humo (ver sección 'humo' del yaml)")
    args = ap.parse_args()
    args.config = args.config or ruta_config("train.yaml", args.dataset)

    cfg = cargar_yaml(args.config)
    m = cfg[args.rol]
    fijar_semillas(m["seed"])

    model = YOLO(ruta_pesos_preentrenados(m["modelo"]))
    model.add_callback(
        "on_pretrain_routine_start",
        callback_metadatos([args.config, cfg["comun"]["data"]],
                           {"rol": "teacher", "bloque": args.rol, "modelo": m["modelo"], "humo": args.humo}),
    )
    kwargs = argumentos_entrenamiento(cfg, args.rol, m["seed"], m["name"], args.humo)
    model.train(**kwargs)
    print(f"\nPesos: {model.trainer.best}")


if __name__ == "__main__":
    main()
