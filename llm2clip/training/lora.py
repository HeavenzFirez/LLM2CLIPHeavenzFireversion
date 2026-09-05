"""LoRA (Low-Rank Adaptation) helpers for efficient LLM2CLIP fine-tuning.

The supervised caption-contrastive stage already applies LoRA via ``peft`` (see
``llm_caption_contrastive/run_supervised.py`` and the configs under
``train_configs/supervised``). This module centralises a small, reusable toolkit
around that workflow:

* :func:`build_lora_config` — construct a ``peft.LoraConfig`` from the same
  defaults used by the shipped configs (rank 16, alpha 32, bfloat16), with sane
  overrides.
* :func:`apply_lora` — wrap a frozen model with LoRA adapters.
* :func:`count_trainable_parameters` — report the trainable/total parameter
  split, which is the headline efficiency benefit of LoRA.
* :func:`merge_and_unload` — collapse adapters back into the base weights for
  inference.

All ``peft``/``torch`` imports are guarded so the module can be imported (and
the pure-Python helpers tested) without those packages installed.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

try:
    import torch

    HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    HAS_TORCH = False

try:
    from peft import LoraConfig, get_peft_model, PeftModel

    HAS_PEFT = True
except ImportError:  # pragma: no cover
    LoraConfig = None  # type: ignore[assignment]
    get_peft_model = None  # type: ignore[assignment]
    PeftModel = None  # type: ignore[assignment]
    HAS_PEFT = False


# Defaults mirrored from llm_caption_contrastive/train_configs/supervised/...
# MetaLlama3_cc3m.json so this module matches the shipped training recipe.
DEFAULT_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 2 * DEFAULT_LORA_R  # alpha = 2*r is the common LoRA heuristic
DEFAULT_LORA_DROPOUT = 0.05


def _require(module_name: str, available: bool):
    if not available:
        raise RuntimeError(
            f"{module_name} is required for this operation; install it via "
            "`pip install peft torch` or the project requirements."
        )


def build_lora_config(
    r: int = DEFAULT_LORA_R,
    alpha: int = DEFAULT_LORA_ALPHA,
    dropout: float = DEFAULT_LORA_DROPOUT,
    target_modules: Optional[list] = None,
    bias: str = "none",
    task_type: Any = None,
    **kwargs,
):
    """Build a ``peft.LoraConfig`` with LLM2CLIP-friendly defaults.

    Keeping ``alpha = 2 * r`` matches the scaling behaviour of the supervised
    config and is a robust default across decoder LLMs used as text encoders.
    """
    _require("peft", HAS_PEFT)
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules or list(DEFAULT_LORA_TARGETS),
        bias=bias,
        task_type=task_type,
        **kwargs,
    )


def apply_lora(model, config=None, **config_kwargs):
    """Wrap ``model`` with LoRA adapters defined by ``config`` (or kwargs)."""
    _require("peft", HAS_PEFT)
    if config is None:
        config = build_lora_config(**config_kwargs)
    return get_peft_model(model, config)


def count_trainable_parameters(model) -> Dict[str, int]:
    """Return ``{trainable, total, trainable_pct}`` for a (peft) model.

    Implemented defensively so it works on both torch ``nn.Module`` and
    ``peft`` wrappers without assuming a particular structure.
    """
    _require("torch", HAS_TORCH)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = (trainable / total * 100.0) if total > 0 else 0.0
    return {"trainable": trainable, "total": total, "trainable_pct": round(pct, 4)}


def merge_and_unload(model):
    """Merge LoRA adapters into the base model and return the unwrapped model.

    Mirrors the ``merge_and_unload`` calls in
    ``llm_caption_contrastive/llm2vec_wrapper.py`` so the inference path stays
    consistent with the training path.
    """
    _require("peft", HAS_PEFT)
    if not isinstance(model, PeftModel):
        # Nothing to merge; return as-is.
        return model
    return model.merge_and_unload()


__all__ = [
    "DEFAULT_LORA_TARGETS",
    "DEFAULT_LORA_R",
    "DEFAULT_LORA_ALPHA",
    "DEFAULT_LORA_DROPOUT",
    "build_lora_config",
    "apply_lora",
    "count_trainable_parameters",
    "merge_and_unload",
]
