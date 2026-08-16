# Cross-Modal Fusion & Advanced Algorithms

This document describes the enhancement modules added to LLM2CLIP under
`llm2clip/training/` (`advanced_algorithms.py`, `lora.py`, `enhanced_model.py`)
and the mathematical formulations behind them. These extend the original
caption-to-caption contrastive recipe with efficiency and fusion techniques.

> All modules import defensively: `torch`/`peft` are optional, so the
> pure-Python helpers (e.g. the LR scheduler) stay usable in minimal
> environments and unit tests run without heavy dependencies.

---

## 1. Transformer-Based Cross-Attention Fusion

Given CLIP visual features **V** and LLM text features **L** (both in
ℝ^{B×d}, or sequence-shaped ℝ^{B×S×d}), the visual stream acts as the **query**
and the language stream as the **key/value**. This lets dense LLM captions
refine the visual representation where text carries stronger signal.

### Projection

$$
Q = V W_q,\qquad K = L W_k,\qquad V' = L W_v
$$

### Scaled dot-product attention (multi-head)

$$
A = \mathrm{softmax}\!\left(\frac{Q K^{\top}}{\sqrt{d_h}}\right),\qquad
F_{\text{attn}} = A \cdot V'
$$

where $d_h = d / h$ is the per-head dimension. See `CrossModalAttention` in
`advanced_algorithms.py`.

---

## 2. Dynamic Gating

A per-element learnable gate balances the two modalities, letting the model lean
on the LLM for long/dense captions and on CLIP for fine visual detail:

$$
g = \sigma\!\left(W_g [\,V;\,L\,]\right)
$$

$$
F_{\text{final}} = g \odot V + (1 - g) \odot L
$$

`AdvancedOptimization.advanced_fusion` chains cross-attention (CLIP→LLM)
followed by dynamic gating of the attended output with the raw CLIP features.

---

## 3. Focal Loss for Imbalanced Retrieval

For binary or per-class retrieval heads where positive pairs are rare, focal
loss down-weights easy negatives:

$$
\text{FL}(p_t) = -\alpha\,(1 - p_t)^{\gamma}\,\log(p_t)
$$

with $p_t = e^{-\text{BCE}}$. Implemented in `focal_loss(...)` with the same
signature as `F.binary_cross_entropy_with_logits`, so it drops into existing
heads.

---

## 4. Adaptive Learning-Rate Schedule

A deterministic step-decay schedule used by `AdvancedOptimization`:

$$
\text{lr}(\text{step}) = \text{lr}_0 \cdot \gamma^{\lfloor \text{step} / T \rfloor}
$$

Defaults follow the shipped supervised config: $T = 30$, $\gamma = 0.1$. The
function is pure-Python so the schedule can be logged/inspected before being
applied to an optimizer via `apply_adaptive_lr`.

---

## 5. LoRA Efficiency

Low-Rank Adaptation freezes the base LLM and trains rank-$r$ update matrices
$\Delta W = B A$ where $A \in \mathbb{R}^{r\times d}$, $B \in \mathbb{R}^{d\times r}$:

$$
W' = W + \frac{\alpha}{r}\, B A
$$

`lora.py` centralises this with defaults matching `train_configs/supervised/`
(rank 16, $\alpha = 2r$, targets `q_proj`/`k_proj`/`v_proj`/`o_proj`), plus
parameter counting and `merge_and_unload` for inference.

---

## 6. EnhancedLLM2CLIP Wrapper

`enhanced_model.py` exposes a single facade over a frozen CLIP visual encoder
and an LLM text encoder:

```python
from llm2clip.training.enhanced_model import EnhancedLLM2CLIP

model = EnhancedLLM2CLIP(clip_model, llm_model, fusion_dim=768, use_fusion=True)
img = model.encode_image(images)            # CLIP visual stream
txt = model.encode_text(text_inputs)         # LLM text stream
mm  = model.encode_multimodal(images, text)  # cross-attention + gating
```

When `use_fusion=False` (or no LLM is supplied), the wrapper simply concatenates
the two normalized streams, preserving the original concatenation-fusion
behaviour.

---

## Benchmarks (from the proposal)

| Method          | COCO mAP@1 | Inference Speed (ms) |
|-----------------|------------|----------------------|
| Concatenation   | 68.2       | 12.4                 |
| Cross-Attention | **76.5**   | 14.7                 |
| Dynamic Gating  | 74.8       | **11.9**             |
| Advanced Fusion | **80.0**   | 13.2                 |

These are target figures from the enhancement proposal; reproduce them by
training with the corresponding fusion option enabled.
