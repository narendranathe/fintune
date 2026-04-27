# FinTune — Engineering Specs & Decision Log

This is the engineering source of truth for FinTune. It documents *why* the system looks the way it does, not just *what* it does. Decisions are recorded with their trade-offs so future contributors (or future-me) can revisit them with full context.

---

## Why This Project Exists

Generic large language models are not safe to drop into a financial workflow. They hallucinate fund names, mis-classify "guidance cut" as positive sentiment, leak account numbers and SSNs back in completions, and have no calibration on financial-specific phrasing like "write-down," "impairment," or "earnings beat." When a model serves an analyst desk, a compliance pipeline, or a customer-facing summarizer, those failures translate directly into regulatory exposure (MNPI handling, PII under GLBA / GDPR) and bad trades.

FinTune is a small, opinionated reference implementation of how to take a base language model, **fine-tune it on a public financial corpus** with QLoRA so the cost stays on a single consumer GPU, and then **wrap inference in the production primitives** — pre-inference PII redaction, confidence thresholding, a circuit breaker, drift detection, and self-recovery — that make it deployable instead of demo-able. The point is not to set a benchmark; the point is to show the shape of a financial NLP service that a risk team would actually sign off on.

---

## Architecture Overview

```
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ financial_       │───▶│  QLoRA           │───▶│  Merged Model    │───▶│  Quantized       │
│ phrasebank       │    │  Fine-Tuning     │    │  (LoRA folded    │    │  Inference       │
│ (HF Hub)         │    │  (PEFT)          │    │   into base)     │    │  (4-bit NF4)     │
└──────────────────┘    └──────────────────┘    └──────────────────┘    └──────────────────┘
        │                       │                                                 │
        ▼                       ▼                                                 ▼
  Quality validator       Cosine LR scheduler                            ┌──────────────────┐
  (empty / dup /          paged AdamW 8-bit                              │  FastAPI         │
  imbalance check)        Gradient checkpointing                         │  /predict        │
        │                 Early stopping                                 │  /predict/batch  │
        └───── data.py    Eval F1-macro                                  │  /health         │
              data_       train.py                                       │  /metrics        │
              pipeline.py                                                └──────────────────┘
                                                                                  │
                                                                                  ▼
                                                                   ┌──────────────────────────────┐
                                                                   │ Pre-inference guardrails     │
                                                                   │  ├─ PII redact (regex)       │
                                                                   │  ├─ Confidence threshold     │
                                                                   │  └─ Output label validation  │
                                                                   ├──────────────────────────────┤
                                                                   │ SystemMonitor (singleton)    │
                                                                   │  ├─ p50/p95/p99 latency      │
                                                                   │  ├─ Throughput / error rate  │
                                                                   │  ├─ KL-divergence drift      │
                                                                   │  └─ Composite health 0-100   │
                                                                   ├──────────────────────────────┤
                                                                   │ RecoveryManager              │
                                                                   │  ├─ CircuitBreaker (3-state) │
                                                                   │  ├─ Exp. backoff reload      │
                                                                   │  ├─ OOM → batch reduction    │
                                                                   │  └─ Quality drop → reload    │
                                                                   └──────────────────────────────┘
```

Inference flow per request:
`POST /predict` → CircuitBreaker.allow_request() → classifier → apply_guardrails() → SystemMonitor.record_request() → JSON response

---

## Key Architectural Decisions

### Decision 1: QLoRA over full fine-tuning

- **What:** 4-bit NF4 quantization of the base model + LoRA adapters on attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`), `r=16`, `alpha=32`, `dropout=0.05`. Configured in `src/model.py` and `configs/qlora_config.yaml`.
- **Why:** Full fine-tuning a 7B parameter model in bf16 needs roughly 56 GB of VRAM (4 bytes/param × 7B × ~2 for optimizer states). QLoRA with NF4 + double quantization brings the footprint to ~6 GB, which fits a T4 / 3060 / a free Colab GPU. The base weights are frozen and only the LoRA adapters (a few MB) are trained.
- **Trade-off:** Roughly 1–2% F1 degradation versus full fine-tuning in published QLoRA results (Dettmers et al., 2023). The compute saving is ~10×; we accept the quality hit.
- **Alternatives considered:**
  - **LoRA without quantization** — still needs ~14 GB for 7B. Doable on a single A10/L4 but not on consumer hardware.
  - **Adapter layers / IA³** — fewer params but less expressive; worse downstream quality on classification.
  - **Prompt tuning** — zero training cost but unstable on small classification datasets like financial_phrasebank.

### Decision 2: financial_phrasebank as the training corpus

- **What:** `takala/financial_phrasebank`, `sentences_allagree` subset (Malo et al., 2014). 4,845 financial news sentences, 3 labels (`negative` / `neutral` / `positive`). Loaded via `datasets` in `src/data.py`. Stratified 80/20 split with fixed seed 42.
- **Why:** Most high-quality financial NLP benchmarks (FiQA, FPB-multi, ECC corpora) are paywalled or behind license walls. financial_phrasebank is **public, peer-reviewed, and annotated by domain experts** (16 annotators with finance backgrounds). The `allagree` filter — only sentences where every annotator agreed on the label — gives the cleanest signal at the cost of dataset size.
- **Trade-off:** The domain is **news headlines and short prose**, not earnings call transcripts, 10-K filings, or analyst notes. A model trained here will transfer poorly to long-form filings and likely needs a second fine-tune for those distributions. We accept this because it's the strongest free signal available, and "swap the dataset" is a one-file change in `src/data.py`.

### Decision 3: Guardrails before inference, not after

- **What:** `src/guardrails.py::apply_guardrails()` runs **PII redaction on the input text** and confidence + label validation **after the model returns**. The redacted text is what gets logged; the original is only held in memory for the duration of the request.
- **Why:** In a financial service, the failure mode that matters is "the model echoed back a customer's account number in the response and that response got logged to disk / Splunk / a metrics dashboard." Post-inference guardrails are too late if logging happens in the middleware layer. Pre-inference redaction guarantees the audit log never sees raw PII even if the inference itself fails.
- **Trade-off:** Regex-based PII detection (`PII_PATTERNS` in `guardrails.py`) has known false positives — `account_number` matches any 8–17 digit string, which will catch order IDs and ticker counts. This is intentional for v0.x: prefer false positives (over-redact) over false negatives (leak). A future iteration should swap to Presidio (already in `requirements.txt`) for ML-based NER detection.

### Decision 4: Circuit breaker pattern for self-recovery

- **What:** `src/self_recovery.py::CircuitBreaker` is a 3-state machine — `CLOSED` (normal) → `OPEN` (rejecting) → `HALF_OPEN` (probing) → `CLOSED`. After 5 consecutive failures the circuit opens and `/predict` returns 503. After a 30-second timeout the circuit goes half-open and probes; 2 consecutive successes close it again.
- **Why:** ML inference services have a fault mode that web services don't: a single bad input or a corrupted model file can put the service into a state where every subsequent request fails the same way. Without a breaker, the failure mode is "100% error rate, retry storm from clients, GPU pegged at 100% doing useless work." The breaker fails fast, gives the system room to recover (or be recovered by the `RecoveryManager`), and surfaces the degraded state in `/health` so an orchestrator can route around it.
- **Trade-off:** During the `OPEN` window, even legitimate requests get rejected. We accept this because the alternative — letting the request queue grow under sustained failure — is strictly worse for latency tail.

### Decision 5: distilbert-base-uncased as the CPU-testable base model

- **What:** Two configs ship side-by-side. `configs/qlora_config.yaml` targets `mistralai/Mistral-7B-v0.3` for the real GPU run; `configs/qlora_distilbert_cpu.yaml` targets `distilbert-base-uncased` with `r=8`, `alpha=16`, no 4-bit quantization, `target_modules: [q_lin, v_lin]` (DistilBERT's attention naming).
- **Why:** A contributor without a GPU should be able to run `python -m src.train --config configs/qlora_distilbert_cpu.yaml` and see the full pipeline execute end-to-end in minutes. DistilBERT is 66M parameters, trains on CPU, and produces a real (if smaller) classifier so all the downstream serving / guardrails / monitoring code is exercised against a real artifact.
- **Trade-off:** DistilBERT was pre-trained on general English (BookCorpus + Wikipedia), not financial text. Its baseline financial sentiment performance is well below the Mistral-7B run, but it's good enough to validate the *plumbing*. The sklearn baseline in `src/benchmark.py` exists to put a number on this gap.

---

## Phase History

| Phase | What was built | Date |
|-------|----------------|------|
| v0.1.0 | Project scaffold: directory layout, requirements.txt, base Dockerfile | (initial commit) |
| v0.2.0 | QLoRA training loop, FastAPI serving, guardrails, monitor, self-recovery, sklearn baselines, full test suite | 2026-04-26 |
| v0.2.1 | `trust_remote_code=True` fix for `takala/financial_phrasebank` (HF dataset loader change) | 2026-04-26 |
| v0.3.0 (planned) | Real GPU training run with published metrics, Presidio swap-in for PII, GGUF export | TBD |

---

## Environment Variables

| Variable | Purpose | Required | Default |
|----------|---------|----------|---------|
| `FINTUNE_MODEL_PATH` | Path to merged model directory loaded by `src/serve.py` on startup | No | `outputs/fintune-financial` |
| `FINTUNE_FALLBACK_MODEL_PATH` | Path to quantized fallback model used if primary load fails | No | `outputs/fintune-financial-quantized` |
| `HF_TOKEN` | HuggingFace Hub access token | No | unset (financial_phrasebank and Mistral-7B-v0.3 are public; only needed for gated models) |
| `CUDA_VISIBLE_DEVICES` | GPU selection for multi-GPU machines | No | all visible |
| `TRANSFORMERS_CACHE` / `HF_HOME` | Override HuggingFace download cache location | No | `~/.cache/huggingface` |

No project-specific config file is needed — all training params live in `configs/*.yaml`, all serving params in env vars or `src/serve.py` defaults.

---

## Testing & Coverage

The `tests/` directory contains test modules covering each layer:

| Module | What it covers |
|--------|----------------|
| `test_data.py` | Dataset loading, tokenization, label mapping |
| `test_data_pipeline.py` | Quality validator, deduplication, stratified sampling, streaming loader |
| `test_model.py` | QLoRA / LoRA config builders, model load path |
| `test_guardrails.py` | PII regex coverage, confidence threshold, label validation |
| `test_serve.py` | FastAPI endpoints, request/response schemas |
| `test_monitor.py` | Latency percentiles, error rate, throughput, KL-divergence drift |
| `test_self_recovery.py` | Circuit breaker state transitions, batch-size reduction, retry backoff |

Run with `pytest tests/ -v` from the project root.

---

## Open Issues / Next Steps

- [ ] **Real earnings-call dataset** — pull SEC EDGAR 10-K / 10-Q risk-factor sections and add as a second fine-tune stage; financial_phrasebank alone won't generalize to long-form filings.
- [ ] **GPU inference benchmarks** — current latency numbers are CPU baseline only; need real p50/p95/p99 on T4 / L4 / A10 with the merged Mistral model.
- [ ] **Replace regex PII with Presidio** — `presidio-analyzer` is already pinned in `requirements.txt`; wire it into `guardrails.py` behind a `PII_BACKEND=regex|presidio` flag.
- [ ] **RLHF / DPO alignment pass** — current model is supervised-only; an alignment pass on "abstain when uncertain" would reduce the false-confident-prediction failure mode that the confidence threshold currently catches reactively.
- [ ] **Multi-language support** — financial_phrasebank is English-only; for European desks we'd need at minimum German + French.
- [ ] **GGUF export** — for CPU-only deployment via `llama.cpp`. Requires a merge step from the PEFT adapter to a single `.gguf` file.
- [ ] **Drift detection baseline persistence** — `SystemMonitor.set_baseline_distribution()` is currently set in-memory at startup; should load from disk so it survives restarts.
- [ ] **Per-tenant rate limiting** — `/predict` is currently un-throttled; add an upstream limiter before exposing to multiple clients.
- [ ] **Replace 8–17 digit `account_number` regex** — too aggressive; matches ticker counts, order IDs, ZIP+4 strings. Either narrow the regex or hand off entirely to Presidio.
