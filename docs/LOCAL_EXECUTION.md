# Local & Self-Hosted Execution

This fork is built strictly anti-profit and non-proprietary: every component is
documented to run **fully locally** with no dependency on paid cloud APIs or
centralized gatekeepers. The paths below let a community operate LLM2CLIP and
the enhancement modules on commodity hardware.

> No part of this stack requires a subscription, API key for metered inference,
> or a vendor-controlled endpoint. The only tokens referenced in the codebase
> are HuggingFace download-auth fields (`HF_TOKEN`), used solely to fetch openly
> licensed model weights you then run yourself.

---

## 1. Local LLM text encoder (no paid API)

The caption-contrastive stage (`llm_caption_contrastive/`) uses an LLM as the
text teacher. Run that LLM locally instead of calling a hosted API:

### vLLM (high-throughput, GPU)

```bash
# Serve the Llama-3-8B teacher locally, fully on your own hardware
pip install vllm
vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
  --port 8000 --dtype auto
```

Point your dataset/eval code at `http://localhost:8000` (OpenAI-compatible).
vLLM is Apache-2.0 and runs entirely on local GPUs.

### Ollama (single-node, GGUF quantized, CPU+GPU)

```bash
# Pull a quantized open model and run it locally
ollama pull llama3:8b
ollama run llama3:8b            # interactive, or expose the API on :11434
```

### GGML / EXL2 quantization (minimal-footprint self-hosting)

For low-VRAM or CPU-only nodes, run GGUF (llama.cpp) or EXL2 quantized weights.
These keep the model open and executable without a vendor runtime.

---

## 2. CLIP visual encoder (local)

The EVA-CLIP visual encoders in `llm2clip/eva_clip/` load from open config JSON
under `model_configs/` and weights from the open HuggingFace collection. Once
downloaded, inference is fully local:

```bash
python -c "
import torch
from eva_clip import create_model_and_transforms
model, _, preprocess = create_model_and_transforms('EVA02-CLIP-L-14-336')
model.eval()
# encode_image(...) now runs entirely on your hardware
"
```

---

## 3. The enhancement modules (no network at all)

`llm2clip/training/advanced_algorithms.py`, `lora.py`, and `enhanced_model.py`
are pure compute — they contain **no network calls**. They operate on tensors
you provide, so they run offline once you have any CLIP + LLM objects loaded:

```python
from llm2clip.training.enhanced_model import EnhancedLLM2CLIP

# clip_model / llm_model are loaded locally per sections 1-2
model = EnhancedLLM2CLIP(clip_model, llm_model, fusion_dim=768, use_fusion=True)
mm = model.encode_multimodal(images, text_inputs)   # fully offline
```

---

## 4. Reproducible environment (Docker)

Containerize the whole stack so a node is self-contained and reproducible:

```dockerfile
FROM python:3.10-slim
WORKDIR /repo
COPY llm2clip/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Optionally install a local inference server:
# RUN pip install vllm
CMD ["python", "tests/test_advanced_algorithms.py"]
```

```bash
docker build -t llm2clip-fork .
docker run --rm --gpus all llm2clip-fork
```

---

## 5. Data formats (open, portable)

Training data uses open, non-proprietary formats only:

- **WebDataset tar shards** — `cc3m-train-{00..0287}.tar`, `cc12m-*.tar`
  (see `llm2clip/run.sh`)
- **CSV** — caption pairs (`llm_caption_contrastive` cc3m.csv with
  `short_caption` / `long_caption` columns)
- **JSON** — configs under `train_configs/`, eval datasets under
  `eval_datasets.yaml`

No proprietary blobs, no vendor-locked storage, no metered feature stores.

---

## 6. Peer-to-peer distribution

The repo and its open-licensed weights can be mirrored over:

- **Gitea** — self-hosted git, no GitHub dependency
- **IPFS** — content-addressed model/weight distribution by CID
- **Community compute nodes** — vLLM/Ollama instances shared across a LAN

This keeps the system resilient to any single platform's availability or
policy changes.
