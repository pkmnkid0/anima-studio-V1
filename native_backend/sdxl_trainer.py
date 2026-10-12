"""
SDXL LoRA training for Anima Studio's native backend.

SDXL runs through the vendored sd-scripts trainer (sdxl_train_network) - the same engine as Anima -
so it gets real optimizers, schedulers, noise offset, min-SNR, bucketing, caption handling and
checkpointing. See anima_trainer.run_kind().
"""
from __future__ import annotations


class TrainerError(RuntimeError):
    pass


def run(args: dict, dataset: dict, on_progress, should_stop) -> None:
    from . import anima_trainer
    anima_trainer.run_kind("sdxl", args, dataset, on_progress, should_stop)
