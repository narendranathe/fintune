# FinTune

QLoRA fine-tuned Mistral-7B for financial sentiment, served behind FastAPI with pre-inference PII redaction, a 3-state circuit breaker, and KL-divergence drift monitoring. The model is the easy part; this repo is about the machinery that makes a checkpoint deployable in a regulated environment.

---

## The engineering problem

A stock `transformers.pipeline()` behind FastAPI fails in three ways that matter in finance:

1. **It leaks PII.** Account numbers, SSNs, and customer emails pass through inference logs, metrics, and prompt traces. Under GLBA / GDPR that is a reportable breach, not a bug.
2. **It has no fault model.** No circuit breaker, no drift detection, no batch-size adaptation under OOM. One bad input can wedge the service into a permanent failure loop with no signal to the orchestrator.
3. **It is not calibrated for the domain.** A general-purpose model reads "guidance" as a neutral noun and misses that "cut guidance" is unambiguously negative.

FinTune addresses each with a specific mechanism, not a wrapper.

---

## Mechanisms

| Problem | Mechanism | Consequence |
|---------|-----------|-------------|
| PII leakage | Pre-inference regex redaction, biased to over-redact | Sensitive tokens never reach the model or the logs; `presidio-analyzer` pinned for a v0.3 swap |
| Cascading failure | 3-state circuit breaker + `RecoveryManager` | OOM triggers batch-size reduction; latency spikes trigger fallback to the quantized model; load failures trigger exponential-backoff reload |
| Silent quality decay | KL-divergence drift monitoring on output distributions | Distribution shift shows up in `/metrics` before accuracy visibly drops |
| Training cost | QLoRA (4-bit NF4 + LoRA adapters, `paged_adamw_8bit`) | Fine-tuning a 7B model fits on a single consumer GPU; under 2% F1 loss vs full fine-tune in practice |

---

## Pipeline

```
financial_phrasebank          QLoRA fine-tune         Merged model        4-bit NF4         FastAPI
(HF Hub, sentences_allagree,  (peft, bitsandbytes,    (LoRA folded  ───▶  inference   ───▶  /predict
 stratified 80/20, seed 42)    F1-macro checkpoint,    into base)                            /health
                               early stopping 3)                                             /metrics
                                                                                               │
                                                                                               ▼
                                                                              Pre-inference guardrails
                                                                              (PII redaction, confidence
                                                                               threshold, label validation)
                                                                                               │
                                                                                               ▼
                                                                              SystemMonitor + RecoveryManager
                                                                              (p50/p95/p99 latency, error rate,
                                                                               KL-divergence drift, health score,
                                                                               3-state circuit breaker)
```

Per-decision rationale and trade-offs in [`specs/README.md`](specs/README.md). Domain glossary in [`UBIQUITOUS_LANGUAGE.md`](UBIQUITOUS_LANGUAGE.md).

---

## Evaluation

Metrics are produced by the evaluation pipeline, not hand-written into this README. Run:

```bash
python -m src.evaluate --model-path outputs/fintune-financial --dataset test
python -m src.benchmark   # TF-IDF + LogisticRegression / RandomForest / LinearSVC baselines, same split, CPU
```

`src.evaluate` writes F1 (macro and per-class), accuracy, and eval loss to `outputs/`; `src.benchmark` writes sklearn baseline scores for the same stratified split so the fine-tune has an honest floor to beat. The result-collection protocol, including the latency measurement method (p50 / p95 / p99, single sample, no batching, 4-bit NF4), is in [`specs/README.md`](specs/README.md).

Training corpus: `takala/financial_phrasebank`, `sentences_allagree` subset (100% annotator agreement, Malo et al., 2014). References: Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs" (NeurIPS 2023).

---

## Known weaknesses

From `specs/README.md`, which is the decision log for this repo:

- The `account_number` PII regex matches any 8 to 17 digit string, so it redacts order IDs, ticker counts, and ZIP+4 codes along with account numbers. That is deliberate (over-redact beats leak in v0.x), but `presidio-analyzer` is already pinned in `requirements.txt` and not wired in. The swap is the first thing a v0.3 should do.
- The drift-detection baseline lives in memory. `SystemMonitor.set_baseline_distribution()` runs at startup, so every restart silently resets the reference distribution and the KL-divergence monitor is blind until it rebuilds. It should load from disk.
- The latency numbers in `outputs/` are CPU-only, from the DistilBERT config. There are no published p50/p95/p99 figures for the merged Mistral model on a T4/L4/A10 yet.
- The training corpus is 4,845 sentences of news prose. It will not transfer to 10-K risk sections or earnings-call transcripts without a second fine-tune stage, which is planned and not done.
- `/predict` has no rate limiting. Fine on a laptop, not fine with more than one client.
- v0.2.1 exists because a HuggingFace `datasets` loader change broke training data loading mid-project; the fix was `trust_remote_code=True` in `src/data.py`. Pin your data loader versions.

---

## Run it

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
python -m src.train --config configs/qlora_distilbert_cpu.yaml
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

## Tech stack

| Layer | Choice | Why |
|-------|--------|-----|
| Base model (GPU) | `mistralai/Mistral-7B-v0.3` | Strong open-weight 7B, permissive license |
| Base model (CPU) | `distilbert-base-uncased` | Full pipeline runnable without a GPU |
| Fine-tuning | QLoRA via `peft` + `bitsandbytes` | ~10x VRAM savings vs full fine-tune |
| Serving | FastAPI + Uvicorn | Async, Pydantic validation, lifespan-managed model load |
| Observability | Custom `SystemMonitor` (singleton) | Latency percentiles, throughput, error rate, drift, composite health score |
| Self-recovery | `RecoveryManager` + 3-state `CircuitBreaker` | Backoff reload, OOM batch reduction, quantized fallback |
| Container | `nvidia/cuda:12.1.1-runtime-ubuntu22.04` | `docker-compose.yml` reserves 1 NVIDIA device |
| Tests | `pytest`, 35+ cases across 7 modules | Data, model, guardrails, serve, monitor, self-recovery, pipeline |

---

## License

MIT
