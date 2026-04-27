# FinTune

Domain-tuned language model for financial sentiment, with the production primitives that make it deployable.

---

## Why this exists

Generic LLMs are a poor fit for financial NLP, in three specific ways that matter for production:

1. **They hallucinate financial facts.** A general-purpose model has no calibration on phrasing like "guidance cut," "write-down," or "earnings beat." It will read "guidance" as the noun and miss that "cut guidance" is unambiguously negative.
2. **They leak PII.** Account numbers, SSNs, and customer emails routinely pass through inference logs, metrics, and prompt traces. Under GLBA / GDPR that's a breach.
3. **They have no fault model.** A standard `transformers.pipeline()` deployed behind FastAPI has no circuit breaker, no drift detection, no batch-size adaptation under OOM, no health score for orchestrators to route on. The first bad input puts the service into a permanent failure loop.

FinTune is a small reference implementation that addresses all three: a domain-fine-tuned classifier with **pre-inference PII redaction**, **a 3-state circuit breaker**, **KL-divergence drift monitoring**, and **autonomous recovery actions** (model reload, batch-size reduction, fallback-to-quantized).

---

## What it does

Fine-tunes a base language model on the public **financial_phrasebank** corpus (Malo et al., 2014) for 3-class sentiment classification (`positive` / `neutral` / `negative`), using **QLoRA** (4-bit NF4 quantization + LoRA adapters) so the whole training run fits on a single consumer GPU. Then serves the merged model behind a FastAPI endpoint that wraps every prediction in pre-inference guardrails, real-time monitoring, and a self-recovery layer.

---

## Results

> Numbers below are placeholders to be filled by the next pipeline run on this branch. See `specs/README.md` for the result-collection protocol.

**QLoRA fine-tune (Mistral-7B-v0.3, financial_phrasebank `sentences_allagree`, stratified 80/20 split, seed 42):**

| Metric | Value |
|--------|-------|
| F1 (macro) | `{RESULT_F1_MACRO}` |
| Accuracy | `{RESULT_ACCURACY}` |
| Precision (macro) | `{RESULT_PRECISION_MACRO}` |
| Recall (macro) | `{RESULT_RECALL_MACRO}` |
| Eval loss | `{RESULT_EVAL_LOSS}` |
| Training time | `{RESULT_TRAIN_TIME}` |
| Hardware | `{RESULT_HARDWARE}` |

**Per-class F1:**

| Class | F1 |
|-------|-----|
| negative | `{RESULT_F1_NEGATIVE}` |
| neutral  | `{RESULT_F1_NEUTRAL}` |
| positive | `{RESULT_F1_POSITIVE}` |

**Inference latency (merged model, 4-bit NF4, single sample, no batching):**

| Percentile | Latency |
|------------|---------|
| p50 | `{RESULT_P50_MS}` ms |
| p95 | `{RESULT_P95_MS}` ms |
| p99 | `{RESULT_P99_MS}` ms |

**Sklearn baselines (TF-IDF + classifier, same split, CPU only):**

| Baseline | F1 (macro) | Accuracy |
|----------|-----------|----------|
| TF-IDF + LogisticRegression | `{RESULT_BASELINE_LR_F1}` | `{RESULT_BASELINE_LR_ACC}` |
| TF-IDF + RandomForest | `{RESULT_BASELINE_RF_F1}` | `{RESULT_BASELINE_RF_ACC}` |
| TF-IDF + LinearSVC | `{RESULT_BASELINE_SVC_F1}` | `{RESULT_BASELINE_SVC_ACC}` |

References: Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs" (NeurIPS 2023). Malo, Sinha, Korhonen, Wallenius, Takala, "Good debt or bad debt: Detecting semantic orientations in economic texts," *JASIST* 65(4), 2014.

---

## How to run it

### One command (Docker, CPU or GPU)

```bash
docker-compose up --build
# API at http://localhost:8000  ·  health: GET /health  ·  metrics: GET /metrics
```

### Train + evaluate locally (GPU, ~15 min on a T4)

```bash
pip install -r requirements.txt
python -m src.train --config configs/qlora_config.yaml
python -m src.evaluate --model-path outputs/fintune-financial --dataset test
```

### CPU prototype run (DistilBERT, no GPU required)

```bash
pip install -r requirements.txt
python -m src.train --config configs/qlora_distilbert_cpu.yaml
```

### Sklearn baselines (no model training, ~30 sec)

```bash
python -m src.benchmark
# → outputs/benchmark_results.json
```

### Single prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Q3 revenue beat consensus by 8%; guidance raised for FY.", "confidence_threshold": 0.7}'
```

```json
{
  "label": "positive",
  "confidence": 0.9542,
  "guardrails_passed": true,
  "flags": [],
  "pii_detected": [],
  "latency_ms": 12.45
}
```

---

## Architecture

```
financial_         QLoRA            Merged model         Quantized          FastAPI
phrasebank   ───▶  fine-tune  ───▶  (LoRA folded   ───▶ inference   ───▶   /predict
(HF Hub)           (PEFT)           into base)          (4-bit NF4)        /health
                                                                           /metrics
                                                                              │
                                                                              ▼
                                                          Pre-inference guardrails
                                                          (PII redact, confidence,
                                                           label validation)
                                                                              │
                                                                              ▼
                                                          SystemMonitor + RecoveryManager
                                                          (latency p50/p95/p99,
                                                           KL-divergence drift,
                                                           3-state circuit breaker,
                                                           OOM → batch reduction)
```

Full per-decision rationale and trade-offs in [`specs/README.md`](specs/README.md). Domain glossary in [`UBIQUITOUS_LANGUAGE.md`](UBIQUITOUS_LANGUAGE.md).

---

## Tech stack

| Layer | Choice | Why |
|-------|--------|-----|
| Base model (GPU config) | `mistralai/Mistral-7B-v0.3` | Strong open-weight 7B with permissive license |
| Base model (CPU config) | `distilbert-base-uncased` | Lets contributors run the full pipeline without a GPU |
| Fine-tuning | QLoRA via `peft` + `bitsandbytes` | ~10× VRAM savings vs. full fine-tune; <2% F1 loss in practice |
| Training corpus | `takala/financial_phrasebank` (`sentences_allagree`) | Public, expert-annotated, 100% annotator agreement subset |
| Training framework | HuggingFace `Trainer` | F1-macro best-checkpoint selection, early stopping (patience 3) |
| Optimizer | `paged_adamw_8bit` | Required for QLoRA; pages optimizer state to CPU |
| Serving | FastAPI + Uvicorn | Async, Pydantic validation, lifespan-managed model load |
| Guardrails | Regex-based PII redaction | Pre-inference, fails to over-redact rather than leak; `presidio-analyzer` pinned for v0.3 swap |
| Observability | Custom `SystemMonitor` (singleton) | Latency percentiles, throughput, error rate, KL-divergence drift, composite health score |
| Self-recovery | `RecoveryManager` + 3-state `CircuitBreaker` | Exponential-backoff reload, OOM batch reduction, latency-spike → quantized fallback |
| Quantization | bitsandbytes NF4 + double quantization | Used both during training (QLoRA) and post-training (`src/quantize.py`) |
| Container | `nvidia/cuda:12.1.1-runtime-ubuntu22.04` | GPU-capable; `docker-compose.yml` reserves 1 NVIDIA device |
| Tests | `pytest` (35+ cases across 7 modules) | Covers data, model, guardrails, serve, monitor, self-recovery, pipeline |

---

## License

MIT
