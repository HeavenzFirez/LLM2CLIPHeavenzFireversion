"""EnhancedLLM2CLIP inference/encoding wrapper.

This wraps a frozen CLIP visual encoder and an LLM text encoder behind a single
``encode_image`` / ``encode_text`` / ``encode_multimodal`` API, optionally
fusing the two streams with the cross-attention + dynamic-gating modules from
:mod:`llm2clip.training.advanced_algorithms`.

It is intentionally minimal and side-effect-free: it does not load checkpoints
or hit the network by itself. Callers pass already-constructed CLIP and LLM
objects (e.g. from ``llm2clip/eva_clip/factory.py`` and a HuggingFace/llm2vec
text encoder), which keeps the wrapper usable both for research prototyping and
for the docs/examples referenced in the ``enhancements`` proposal.
"""

from __future__ import annotations

from typing import Optional

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]
    HAS_TORCH = False

from .advanced_algorithms import AdvancedOptimization, _require_torch


class EnhancedLLM2CLIP(nn):
    """Unified encode_image / encode_text / encode_multimodal facade.

    Parameters
    ----------
    clip_model:
        Object exposing ``encode_image(images, normalize=True)`` (the EVA-CLIP
        contract used in ``llm2clip/eva_clip/factory.py``).
    llm_model:
        Object exposing ``encode(text_inputs)`` returning ``[B, d]`` features
        (the llm2vec contract). May be ``None`` for image-only use.
    fusion_dim:
        Shared feature dimension. Required to enable cross-modal fusion.
    use_fusion:
        When ``True`` and ``fusion_dim`` is set, ``encode_multimodal`` blends
        the CLIP and LLM streams via cross-attention + dynamic gating.
    """

    def __init__(self, clip_model, llm_model=None, fusion_dim: Optional[int] = None,
                 use_fusion: bool = True):
        _require_torch()
        super().__init__()
        self.clip_model = clip_model
        self.llm_model = llm_model
        self.use_fusion = use_fusion and fusion_dim is not None and llm_model is not None
        self.adv = AdvancedOptimization(model=self, fusion_dim=fusion_dim) if self.use_fusion else None

    @classmethod
    def from_pretrained(cls, clip_model, llm_model=None, fusion_dim: Optional[int] = None,
                        use_fusion: bool = True, **_kwargs):
        """Construct from already-loaded encoders.

        Accepts arbitrary ``**_kwargs`` so it matches the ``from_pretrained``
        shape sketched in the ``enhancements`` proposal; the kwargs are ignored
        because the actual checkpoint loading is delegated to the caller.
        """
        return cls(clip_model, llm_model, fusion_dim, use_fusion)

    def encode_image(self, images, normalize: bool = True):
        """Encode images through the CLIP visual encoder."""
        _require_torch()
        features = self.clip_model.encode_image(images, normalize=normalize)
        return features

    def encode_text(self, text_inputs, normalize: bool = True):
        """Encode text through the LLM text encoder."""
        _require_torch()
        if self.llm_model is None:
            raise RuntimeError("No LLM text encoder was provided to this wrapper")
        features = self.llm_model.encode(text_inputs)
        if normalize:
            features = torch.nn.functional.normalize(features, dim=-1)
        return features

    def encode_multimodal(self, images, text_inputs, normalize: bool = True):
        """Encode both modalities and, if enabled, fuse them into one vector.

        Without fusion the two streams are simply concatenated. With fusion the
        LLM features attend to the CLIP features and the result is gated, which
        is the ``advanced_fusion`` path from the proposal.
        """
        _require_torch()
        image_features = self.encode_image(images, normalize=False)
        text_features = self.encode_text(text_inputs, normalize=False)

        if image_features.shape[-1] != text_features.shape[-1]:
            raise ValueError(
                "CLIP and LLM feature dims differ "
                f"({image_features.shape[-1]} vs {text_features.shape[-1]}); "
                "set fusion_dim to a shared projection dim before fusing"
            )

        if self.use_fusion:
            fused = self.adv.advanced_fusion(image_features, text_features)
        else:
            # Concatenate along the feature dim as a fallback fusion.
            fused = torch.cat([image_features, text_features], dim=-1)

        if normalize:
            fused = torch.nn.functional.normalize(fused, dim=-1)
        return fused


__all__ = ["EnhancedLLM2CLIP"]
