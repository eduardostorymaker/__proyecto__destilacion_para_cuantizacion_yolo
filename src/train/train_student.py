"""Fases 3 y 4: entrena un alumno YOLO11n, base (sin KD) o con destilación CWD.

Ambos usan exactamente configs/train.yaml; la única diferencia es la pérdida CWD (configs/kd.yaml).

Uso (desde la raíz del proyecto, en WSL):
    python -m src.train.train_student --kd off --seed 0                 # alumno base  → student_base_s0
    python -m src.train.train_student --kd on  --seed 0                 # alumno KD    → student_kd_s0 (λ de kd.yaml)
    python -m src.train.train_student --kd on  --piloto 1.0             # piloto de λ  → pilot_kd_l1.0_s0
    python -m src.train.train_student --kd on  --seed 0 --humo          # prueba de humo
    python -m src.train.train_student --kd on  --seed 1 --reanudar      # retoma student_kd_s1 desde last.pt
"""

from __future__ import annotations

import argparse

from ultralytics import YOLO

from src.train.common import argumentos_entrenamiento, callback_metadatos, ruta_pesos_preentrenados
from src.train.cwd import crear_trainer_cwd
from src.utils.config import cargar_yaml, resolver_ruta
from src.utils.repro import fijar_semillas

CFG_TRAIN, CFG_KD = "configs/train.yaml", "configs/kd.yaml"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kd", choices=["on", "off"], required=True)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--piloto", type=float, metavar="LAMBDA", help="corrida del piloto de λ (solo --kd on)")
    ap.add_argument("--humo", action="store_true")
    ap.add_argument("--reanudar", action="store_true", help="retoma la corrida desde su weights/last.pt")
    args = ap.parse_args()

    cfg, kd = cargar_yaml(CFG_TRAIN), cargar_yaml(CFG_KD)
    usa_kd = args.kd == "on"
    if args.piloto is not None and not usa_kd:
        ap.error("--piloto requiere --kd on")

    if args.piloto is not None:
        seed, lam = kd["piloto"]["seed"], args.piloto
        nombre = f"pilot_kd_l{lam}_s{seed}"
    else:
        if args.seed is None:
            ap.error("--seed es obligatorio (salvo en --piloto)")
        seed = args.seed
        if args.seed not in cfg["alumno"]["semillas"] and not args.humo:
            ap.error(f"semilla {seed} fuera de configs/train.yaml {cfg['alumno']['semillas']}")
        lam = (kd["humo"]["lambda"] if args.humo else kd["lambda"]) if usa_kd else None
        nombre = f"student_{'kd' if usa_kd else 'base'}_s{seed}"
    if usa_kd and lam is None:
        raise ValueError("configs/kd.yaml: 'lambda' es null; fíjalo tras el piloto (Fase 4b)")

    fijar_semillas(seed)
    kwargs = argumentos_entrenamiento(cfg, "alumno", seed, nombre, args.humo)
    if args.piloto is not None:
        kwargs.update(epochs=kd["piloto"]["epocas"], project=str(resolver_ruta(kd["piloto"]["project"])))
    if args.humo and usa_kd:
        h = kd["humo"]
        kwargs.update(epochs=h["epochs"], fraction=h["fraction"], project=str(resolver_ruta(h["project"])))

    if args.reanudar:
        # Ultralytics recupera los argumentos y el estado del optimizador desde last.pt
        last = resolver_ruta(kwargs["project"]) / kwargs["name"] / "weights" / "last.pt"
        if not last.exists():
            raise FileNotFoundError(f"No existe {last}: nada que reanudar")
        model = YOLO(str(last))
        trainer = crear_trainer_cwd(str(resolver_ruta(kd["maestro"])), lam, kd["tau"]) if usa_kd else None
        model.train(resume=True, trainer=trainer)
        print(f"\nPesos: {model.trainer.best}")
        return

    model = YOLO(ruta_pesos_preentrenados(cfg["alumno"]["modelo"]))
    extra = {"rol": "student", "kd": usa_kd, "lambda": lam, "tau": kd["tau"] if usa_kd else None,
             "maestro": kd["maestro"] if usa_kd else None, "humo": args.humo, "piloto": args.piloto is not None}
    model.add_callback("on_pretrain_routine_start",
                       callback_metadatos([CFG_TRAIN, cfg["comun"]["data"]] + ([CFG_KD] if usa_kd else []), extra))
    trainer = crear_trainer_cwd(str(resolver_ruta(kd["maestro"])), lam, kd["tau"]) if usa_kd else None
    model.train(trainer=trainer, **kwargs)
    print(f"\nPesos: {model.trainer.best}")


if __name__ == "__main__":
    main()
