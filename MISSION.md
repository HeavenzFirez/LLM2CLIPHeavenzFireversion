# Mission: Anti-Profit & Open-Access Engineering

This repository is built strictly **anti-profit, non-proprietary, and for
public-good distribution**. When software and AI systems are built this way, the
priority shifts entirely toward maximum accessibility, peer-to-peer resilience,
and self-hosted autonomy.

This is a binding design constraint for every contribution to this fork, not a
marketing statement. Contributions that introduce paid tiers, metered access,
proprietary lock-in, or gated features will not be accepted.

---

## Operational Pillars

### 1. Copyleft & Permissive Open-Source Licensing

Architectures, models, and codebases are released under **AGPL-3.0**
(see [`LICENSE`](LICENSE)) to guarantee that all downstream modifications remain
free, transparent, and unencumbered by corporate paywalls or proprietary vendor
lock-in. AGPL-3.0 §13 specifically ensures that network-served derivatives must
also publish their source — closing the "SaaS loophole" that permissive licenses
leave open.

The upstream Microsoft LLM2CLIP code was MIT-licensed; that notice is preserved
verbatim in [`UPSTREAM_LICENSE_MIT.txt`](UPSTREAM_LICENSE_MIT.txt) as its terms
require.

### 2. Decentralized & Sovereign Execution

Tools are structured to run locally — via **vLLM, Ollama, GGML/EXL2
quantization, and Docker** — so communities can execute high-performance models
and full-stack applications without depending on paid cloud APIs or centralized
gateways.

See [`docs/LOCAL_EXECUTION.md`](docs/LOCAL_EXECUTION.md) for concrete, copy-paste
local-run paths. No component in this repo requires a metered endpoint to
function.

### 3. Public Domain Data & Infrastructure

Data formats use open schemas only — **SQLite, PostgreSQL, open CSV/JSON
schemas, WebDataset tar shards** — and the project favors non-profit hosting,
peer-to-peer storage (IPFS/Gitea), and community-driven compute nodes. No
proprietary storage formats or vendor-locked data pipelines.

### 4. Universal Access & Transparency

Subscription tiers, gated feature flags, and token metering are **removed from
the code architecture** to ensure equal access to auditing, reporting, and
engineering tools regardless of financial resources.

Audits of this principle:
- `grep -rniE "subscription|paywall|metering|gated|premium|tier"` returns no
  gating logic in the codebase.
- The only token references are HuggingFace download-auth fields (`HF_TOKEN`),
  used solely to fetch openly licensed weights you then run yourself — not
  inference metering.

---

## Compliance check for contributors

Before merging, a change must satisfy:

1. **License** — new code is AGPL-3.0-compatible; no proprietary-license
   dependencies added.
2. **Local run** — the feature is documented to run on local hardware with no
   paid API.
3. **Open formats** — any new data format is an open schema (CSV/JSON/tar/etc.).
4. **No gating** — no subscription/tier/metering logic introduced.

The CI workflow (`.github/workflows/python-test.yml`) keeps the enhancement
modules verified under both minimal (stdlib-only) and torch-backed environments,
ensuring the stack stays runnable on low-resource community hardware.
