"""Destilación CWD (Channel-Wise Distillation; Shu et al., ICCV 2021) para YOLO de Ultralytics.

Diseño (D3 del CLAUDE.md):
  - Se destilan las 3 entradas de la cabeza Detect (P3, P4, P5). En modo entrenamiento, la cabeza
    devuelve esas entradas en preds["feats"], así que no hacen falta hooks.
  - El maestro (best.pt de la Fase 2) está congelado, en eval(), sin gradientes y con autocast;
    recibe el MISMO batch aumentado que el alumno (batch["img"]).
  - Adaptadores conv 1×1 llevan los canales del alumno a los del maestro. Viven en el trainer,
    NO dentro del modelo: se entrenan con el alumno (grupo propio del optimizador) pero nunca
    entran en la EMA ni en best.pt/last.pt.
  - L = L_det + λ·L_CWD. La pérdida de detección de Ultralytics viene multiplicada por el tamaño
    del batch, así que L_CWD (promedio por imagen y canal, como en MMRazor) se multiplica igual.
  - "cwd" se agrega al diccionario de pérdidas: Ultralytics la promedia por época y la escribe en
    results.csv como train/cwd (valor SIN multiplicar por λ).

Fórmula adaptada de MMRazor (open-mmlab/mmrazor, mmrazor/models/losses/cwd.py,
commit 0919f692878fe0037ff94bc90e46eed721c64a39).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import unwrap_model


def perdida_cwd(f_alumno: torch.Tensor, f_maestro: torch.Tensor, tau: float) -> torch.Tensor:
    """CWD de un nivel: KL(softmax espacial del maestro ‖ del alumno) por canal, promedio sobre N·C."""
    assert f_alumno.shape == f_maestro.shape, f"{f_alumno.shape} != {f_maestro.shape}"
    n, c, h, w = f_alumno.shape
    t = f_maestro.float().reshape(-1, h * w) / tau
    s = f_alumno.float().reshape(-1, h * w) / tau
    p_t = F.softmax(t, dim=1)
    kl = torch.sum(p_t * (F.log_softmax(t, dim=1) - F.log_softmax(s, dim=1))) * tau**2
    return kl / (n * c)


def canales_detect(modelo: nn.Module) -> list[int]:
    """Canales de entrada de la cabeza Detect (uno por nivel P3, P4, P5)."""
    cabeza = next(m for m in modelo.modules() if isinstance(m, Detect))
    return [cabeza.cv2[i][0].conv.in_channels for i in range(cabeza.nl)]


class CriterioCWD:
    """Envuelve el criterio de detección del alumno y le suma λ·L_CWD durante el entrenamiento."""

    def __init__(self, base, alumno: nn.Module, maestro: nn.Module, adaptadores: nn.ModuleList,
                 lam: float, tau: float, amp: bool):
        self.base, self.alumno, self.maestro = base, alumno, maestro
        self.adaptadores, self.lam, self.tau, self.amp = adaptadores, lam, tau, amp

    def __getattr__(self, nombre):  # el resto de atributos (hyp, etc.) son los del criterio original
        return getattr(self.base, nombre)

    def __call__(self, preds, batch):
        perdida, items = self.base(preds, batch)
        if not self.alumno.training:  # validación: solo pérdida de detección
            return perdida, items
        f_alumno = self.base.parse_output(preds)["feats"]
        with torch.no_grad(), torch.autocast("cuda", enabled=self.amp):
            f_maestro = self.maestro(batch["img"])[1]["feats"]  # eval() → (y, preds)
        cwd = sum(perdida_cwd(a(fa), fm, self.tau)
                  for a, fa, fm in zip(self.adaptadores, f_alumno, f_maestro))
        bs = batch["img"].shape[0]
        perdida = torch.cat([perdida, (self.lam * cwd * bs).reshape(1)])
        return perdida, {**items, "cwd": cwd.detach()}


def crear_trainer_cwd(ruta_maestro: str, lam: float, tau: float):
    """Devuelve una subclase de DetectionTrainer con CWD (Ultralytics instancia el trainer por clase)."""

    class TrainerCWD(DetectionTrainer):
        def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
            optimizador = super().build_optimizer(model, name, lr, momentum, decay, iterations)
            alumno = unwrap_model(model)
            maestro = load_checkpoint(ruta_maestro)[0].float().to(self.device).eval()
            for p in maestro.parameters():
                p.requires_grad_(False)
            ch_a, ch_m = canales_detect(alumno), canales_detect(maestro)
            # fork_rng: inicializar los adaptadores no altera el estado aleatorio global, así el alumno KD
            # ve exactamente la misma secuencia aleatoria (aumentación, etc.) que el alumno base.
            with torch.random.fork_rng(devices=[self.device] if self.device.type == "cuda" else []):
                self.adaptadores = nn.ModuleList(nn.Conv2d(a, m, 1) for a, m in zip(ch_a, ch_m)).to(self.device)
            self.maestro = maestro
            grupo = next(g for g in optimizador.param_groups if g.get("param_group") == "weight")
            optimizador.add_param_group({k: v for k, v in grupo.items() if k != "params"}
                                        | {"params": list(self.adaptadores.parameters()), "param_group": "kd_adapter"})
            n = sum(p.numel() for p in self.adaptadores.parameters())
            LOGGER.info(f"CWD: maestro={ruta_maestro} λ={lam} τ={tau} canales alumno {ch_a} → maestro {ch_m}; "
                        f"adaptadores 1×1 con {n} parámetros (no se guardan en el checkpoint)")
            return optimizador

        def _setup_train(self):
            super()._setup_train()  # crea optimizador (y adaptadores) y la EMA, todavía sin el criterio CWD
            alumno = unwrap_model(self.model)
            base = alumno.criterion if getattr(alumno, "criterion", None) is not None else alumno.init_criterion()
            alumno.criterion = CriterioCWD(base, alumno, self.maestro, self.adaptadores, lam, tau, bool(self.amp))

        def optimizer_step(self):
            # Igual que BaseTrainer.optimizer_step (8.4.164), pero el recorte de gradientes (norma 10) se
            # aplica también a los adaptadores, que no están en self.model. unscale_ solo puede llamarse una vez.
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            torch.nn.utils.clip_grad_norm_(self.adaptadores.parameters(), max_norm=10.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            if self.ema:
                self.ema.update(self.model)

    return TrainerCWD
