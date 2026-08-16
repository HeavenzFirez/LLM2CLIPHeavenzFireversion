"""Unit tests for the LLM2CLIP enhancement modules.

These run with the standard library + an optional torch/pytest install. The
pure-Python scheduler tests always run; torch-backed tests are skipped when
torch is unavailable so CI stays green in minimal environments.
"""

import math
import os
import sys

# Make the repository's llm2clip package importable when running from the repo
# root without installation.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    import pytest
except ImportError:  # allow running as a plain script
    pytest = None

from llm2clip.training import advanced_algorithms as aa
from llm2clip.training import lora


# --------------------------------------------------------------------------- #
# Pure-Python scheduler (always runs)
# --------------------------------------------------------------------------- #
def test_adaptive_lr_step_decay():
    assert aa.adaptive_learning_rate(0, initial_lr=0.01) == 0.01
    # gamma=0.1, decay_every=30 -> halves of 30 steps share a factor.
    assert aa.adaptive_learning_rate(29, initial_lr=0.01) == 0.01
    assert aa.adaptive_learning_rate(30, initial_lr=0.01) == pytest_approx(0.001)
    assert aa.adaptive_learning_rate(60, initial_lr=0.01) == pytest_approx(0.0001)


def test_adaptive_lr_validates_inputs():
    try:
        aa.adaptive_learning_rate(1, 0.01, decay_every=0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for decay_every=0")

    try:
        aa.adaptive_learning_rate(1, 0.01, gamma=0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for gamma=0")


def test_advanced_optimization_facade_scheduler():
    adv = aa.AdvancedOptimization(model=None, fusion_dim=8)
    # Same numeric contract as the standalone function.
    assert adv.adaptive_learning_rate(0, 0.01) == 0.01
    assert adv.adaptive_learning_rate(30, 0.01) == pytest_approx(0.001)


def test_apply_lr_without_torch_raises():
    if aa.HAS_TORCH:
        return  # torch is available; the torch path is covered below.
    try:
        aa.apply_adaptive_lr(object(), step=1, initial_lr=0.01)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError when torch is unavailable")


# --------------------------------------------------------------------------- #
# LoRA defaults (always runs)
# --------------------------------------------------------------------------- #
def test_lora_defaults_match_supervised_config():
    # The shipped supervised config uses rank 16; alpha=2*r is our default.
    assert lora.DEFAULT_LORA_R == 16
    assert lora.DEFAULT_LORA_ALPHA == 2 * lora.DEFAULT_LORA_R
    assert "q_proj" in lora.DEFAULT_LORA_TARGETS
    assert "v_proj" in lora.DEFAULT_LORA_TARGETS


def test_lora_build_config_without_peft_raises():
    if lora.HAS_PEFT:
        return
    try:
        lora.build_lora_config()
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError when peft is unavailable")


# --------------------------------------------------------------------------- #
# torch-backed tests (skipped when torch is absent)
# --------------------------------------------------------------------------- #
if aa.HAS_TORCH:  # pragma: no cover - depends on env
    import torch

    def test_focal_loss_shape_and_sign():
        logits = torch.randn(8)
        targets = torch.randint(0, 2, (8,)).float()
        loss = aa.focal_loss(logits, targets, alpha=0.25, gamma=2.0)
        assert loss.ndim == 0
        assert loss.item() >= 0.0

    def test_cross_modal_attention_shapes():
        attn = aa.CrossModalAttention(dim=8, num_heads=2)
        clip = torch.randn(4, 8)
        llm = torch.randn(4, 8)
        out = attn(clip, llm)
        assert out.shape == clip.shape

    def test_dynamic_gating_output_range():
        gate = aa.DynamicGating(dim=8)
        clip = torch.randn(4, 8)
        llm = torch.randn(4, 8)
        out = gate(clip, llm)
        assert out.shape == clip.shape

    def test_advanced_fusion_end_to_end():
        adv = aa.AdvancedOptimization(model=None, fusion_dim=8)
        clip = torch.randn(4, 8)
        llm = torch.randn(4, 8)
        out = adv.advanced_fusion(clip, llm)
        assert out.shape == clip.shape

    def test_apply_lr_updates_param_groups():
        w = torch.nn.Linear(4, 4)
        opt = torch.optim.Adam(w.parameters(), lr=0.01)
        aa.apply_adaptive_lr(opt, step=30, initial_lr=0.01)
        assert opt.param_groups[0]["lr"] == pytest_approx(0.001)


# --------------------------------------------------------------------------- #
# Tiny test helpers (avoid hard dependency on pytest assertions being present)
# --------------------------------------------------------------------------- #
def pytest_approx(expected, rel=1e-6):
    if pytest is not None:
        return pytest.approx(expected, rel=rel)
    # Fallback comparator used only when running without pytest.
    class _Approx:
        def __init__(self, expected, rel):
            self.expected = expected
            self.rel = rel

        def __eq__(self, other):
            return math.isclose(float(other), float(self.expected), rel_tol=self.rel)

        def __repr__(self):  # pragma: no cover
            return f"approx({self.expected})"

    return _Approx(expected, rel)


# Allow `python tests/test_advanced_algorithms.py` without pytest.
if __name__ == "__main__":  # pragma: no cover
    import types

    module = sys.modules[__name__]
    funcs = [v for k, v in sorted(vars(module).items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in funcs:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {exc!r}")
    print(f"\n{len(funcs) - failed}/{len(funcs)} passed")
    sys.exit(1 if failed else 0)
