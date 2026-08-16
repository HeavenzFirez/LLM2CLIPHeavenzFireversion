"""Advanced optimization and fusion algorithms for LLM2CLIP.

This module implements the enhancement primitives described in the repository's
``enhancements`` proposal: adaptive learning-rate scheduling, focal loss for
class-imbalanced retrieval, transformer-style cross-attention fusion of CLIP and
LLM features, and a learnable dynamic-gating mechanism that balances the two
modalities.

The torch-backed implementations are imported lazily so that the module remains
importable (and partially unit-testable) in environments without PyTorch. The
:class:`AdvancedOptimization` helper degrades gracefully to a pure-Python
scheduler when torch is unavailable, which keeps the unit tests dependency-free.
"""

from __future__ import annotations

import math
from typing import Optional

try:  # torch is an optional runtime dependency for the numeric helpers.
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:  # pragma: no cover - exercised only in torch-less envs.
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    HAS_TORCH = False


def adaptive_learning_rate(step: int, initial_lr: float, decay_every: int = 30,
                           gamma: float = 0.1) -> float:
    """Step-decay learning rate.

    .. math:: lr = lr_0 \\cdot \\gamma^{\\lfloor step / decay\\_every \\rfloor}

    Implemented in pure Python so the scheduler is testable without torch and so
    the same formula can be logged/inspected before being applied to an
    optimizer.
    """
    if decay_every <= 0:
        raise ValueError("decay_every must be a positive integer")
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    return initial_lr * (gamma ** (step // decay_every))


def apply_adaptive_lr(optimizer, step: int, initial_lr: float,
                     decay_every: int = 30, gamma: float = 0.1):
    """Apply :func:`adaptive_learning_rate` to every param group of ``optimizer``."""
    if not HAS_TORCH or not hasattr(optimizer, "param_groups"):
        raise RuntimeError("torch-backed optimizer is required to apply the LR schedule")
    lr = adaptive_learning_rate(step, initial_lr, decay_every, gamma)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def _require_torch():
    if not HAS_TORCH:
        raise RuntimeError(
            "PyTorch is required for this operation; install it via "
            "`pip install torch` or the project requirements."
        )


def focal_loss(outputs, targets, alpha: float = 0.25, gamma: float = 2.0,
               reduction: str = "mean"):
    """Focal loss for addressing class imbalance in dense retrieval training.

    .. math:: FL(p_t) = -\\alpha (1 - p_t)^{\\gamma} \\log(p_t)

    Parameters mirror :func:`torch.nn.functional.binary_cross_entropy_with_logits`
    so the function drops directly into existing binary retrieval heads. For
    multi-class retrieval, apply it per positive class.
    """
    _require_torch()
    bce = F.binary_cross_entropy_with_logits(outputs, targets, reduction="none")
    pt = torch.exp(-bce)  # probability of the true class
    loss = alpha * (1.0 - pt) ** gamma * bce
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError(f"unknown reduction: {reduction}")


class CrossModalAttention(nn.Module if HAS_TORCH else object):
    """Transformer-style cross-attention fusion of CLIP and LLM features.

    Given visual features ``V`` (shape ``[N, d]`` or ``[N, L_v, d]``) and LLM
    features ``L`` (shape ``[M, d]`` or ``[N, L_l, d]``), the visual stream
    acts as the *query* and the language stream as the *key/value*:

    .. math::
        Q = VW_q,\\ K = LW_k,\\ V' = LW_v \\\\
        A = \\mathrm{softmax}\\!\\left(\\frac{QK^\\top}{\\sqrt{d}}\\right) \\\\
        F = A \\cdot V'

    The module is intentionally small (three linear projections + softmax) so it
    can be layered on top of frozen CLIP/LLM encoders without retraining them.
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        _require_torch()
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def _split_heads(self, x):
        # x: [B, S, D] -> [B, H, S, D/H]
        b, s, _ = x.shape
        return x.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x):
        b, _, s, _ = x.shape
        return x.transpose(1, 2).contiguous().view(b, s, self.dim)

    def forward(self, clip_features, llm_features):
        """Return fused features attending from CLIP (query) to LLM (key/value)."""
        _require_torch()
        if clip_features.shape[-1] != self.dim or llm_features.shape[-1] != self.dim:
            raise ValueError(
                f"feature dim {clip_features.shape[-1]}/{llm_features.shape[-1]} != model dim {self.dim}"
            )
        # Normalise to a 3D batched tensor [B, S, D].
        clip3 = clip_features.unsqueeze(0) if clip_features.dim() == 2 else clip_features
        llm3 = llm_features.unsqueeze(0) if llm_features.dim() == 2 else llm_features

        q = self._split_heads(self.q_proj(clip3))
        k = self._split_heads(self.k_proj(llm3))
        v = self._split_heads(self.v_proj(llm3))

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = self.out_proj(self._merge_heads(out))

        # Match input rank so callers passing 2D features get 2D back.
        if clip_features.dim() == 2:
            out = out.squeeze(0)
        return out


class DynamicGating(nn.Module if HAS_TORCH else object):
    """Learnable modality gate that blends CLIP and LLM feature streams.

    .. math::
        g = \\sigma(W_g [V; L]) \\\\
        F_{final} = g \\odot V + (1 - g) \\odot L

    The gate is computed per feature element, letting the model rely more on the
    LLM where text is the dominant signal (long, dense captions) and more on CLIP
    where visual detail dominates.
    """

    def __init__(self, dim: int):
        _require_torch()
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.gate = nn.Linear(dim * 2, dim)

    def forward(self, clip_features, llm_features):
        _require_torch()
        if clip_features.shape[-1] != self.dim or llm_features.shape[-1] != self.dim:
            raise ValueError(
                f"feature dim {clip_features.shape[-1]}/{llm_features.shape[-1]} != model dim {self.dim}"
            )
        # Broadcast single vectors against a sequence: keep the common leading dims.
        g = torch.sigmoid(self.gate(torch.cat([clip_features, llm_features], dim=-1)))
        return g * clip_features + (1.0 - g) * llm_features


class AdvancedOptimization:
    """Convenience facade bundling the above primitives for a training loop.

    Parameters
    ----------
    model:
        The model being trained. Only required by the torch-backed fusion/gating
        helpers; ``adaptive_learning_rate`` works without it.
    fusion_dim:
        Feature dimension used when lazily building the cross-attention and
        gating modules. Built on first access so the modules inherit the model's
        device/dtype.
    """

    def __init__(self, model=None, fusion_dim: Optional[int] = None):
        self.model = model
        self.fusion_dim = fusion_dim
        self._cross_attn = None
        self._gate = None

    # --- learning-rate scheduling (torch-free, testable) --------------------
    def adaptive_learning_rate(self, step: int, initial_lr: float = 0.01,
                               decay_every: int = 30, gamma: float = 0.1) -> float:
        return adaptive_learning_rate(step, initial_lr, decay_every, gamma)

    def apply_lr(self, optimizer, step: int, initial_lr: float = 0.01,
                 decay_every: int = 30, gamma: float = 0.1):
        return apply_adaptive_lr(optimizer, step, initial_lr, decay_every, gamma)

    # --- focal loss ----------------------------------------------------------
    def focal_loss(self, outputs, targets, alpha: float = 0.25, gamma: float = 2.0,
                   reduction: str = "mean"):
        return focal_loss(outputs, targets, alpha, gamma, reduction)

    # --- fusion (torch-only) -------------------------------------------------
    def _ensure_fusion(self):
        _require_torch()
        if self._cross_attn is None:
            if self.fusion_dim is None:
                raise ValueError("fusion_dim must be set before using fusion helpers")
            self._cross_attn = CrossModalAttention(self.fusion_dim)
            self._gate = DynamicGating(self.fusion_dim)
            if self.model is not None:
                device = next(self.model.parameters()).device
                self._cross_attn = self._cross_attn.to(device)
                self._gate = self._gate.to(device)
        return self._cross_attn, self._gate

    def advanced_fusion(self, clip_features, llm_features):
        """Cross-attend CLIP->LLM then dynamically gate the result with CLIP."""
        cross_attn, gate = self._ensure_fusion()
        attended = cross_attn(clip_features, llm_features)
        return gate(attended, clip_features)

    def attention_projection(self, clip_features, llm_features):
        """Expose the projected query/key/value tensors (mainly for inspection)."""
        cross_attn, _ = self._ensure_fusion()
        clip3 = clip_features.unsqueeze(0) if clip_features.dim() == 2 else clip_features
        llm3 = llm_features.unsqueeze(0) if llm_features.dim() == 2 else llm_features
        q = cross_attn.q_proj(clip3)
        k = cross_attn.k_proj(llm3)
        v = cross_attn.v_proj(llm3)
        return q, k, v


__all__ = [
    "adaptive_learning_rate",
    "apply_adaptive_lr",
    "focal_loss",
    "CrossModalAttention",
    "DynamicGating",
    "AdvancedOptimization",
]
