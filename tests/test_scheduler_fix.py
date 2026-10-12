"""The Rex scheduler's cycle length must be set, or the LR collapses within a few steps."""
from pathlib import Path

import pytest

TRAIN_UTIL = Path(__file__).resolve().parent.parent / "training_backend" / "sd_scripts" / "library" / "train_util.py"


def test_first_cycle_max_steps_is_injected_in_source():
    src = TRAIN_UTIL.read_text(encoding="utf-8")
    assert "first_cycle_max_steps" in src
    assert 'lr_scheduler_kwargs["first_cycle_max_steps"]' in src


def test_rex_lr_stays_high_early_with_real_cycle_length():
    torch = pytest.importorskip("torch")
    import importlib.util
    path = (Path(__file__).resolve().parent.parent / "training_backend" / "custom_scheduler"
            / "LoraEasyCustomOptimizer" / "RexAnnealingWarmRestarts.py")
    spec = importlib.util.spec_from_file_location("rex", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([param], lr=3e-5)
    sched = mod.RexAnnealingWarmRestarts(opt, gamma=0.9, first_cycle_max_steps=840, min_lr=1e-6, d=0.9)
    for _ in range(100):
        opt.step()
        sched.step()
    assert opt.param_groups[0]["lr"] > 2.5e-5


def test_rex_lr_collapses_without_cycle_length():
    """Documents the bug the injection prevents."""
    torch = pytest.importorskip("torch")
    import importlib.util
    path = (Path(__file__).resolve().parent.parent / "training_backend" / "custom_scheduler"
            / "LoraEasyCustomOptimizer" / "RexAnnealingWarmRestarts.py")
    spec = importlib.util.spec_from_file_location("rex2", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([param], lr=3e-5)
    sched = mod.RexAnnealingWarmRestarts(opt, gamma=0.1, min_lr=0.0)
    for _ in range(20):
        opt.step()
        sched.step()
    assert opt.param_groups[0]["lr"] < 1e-9
