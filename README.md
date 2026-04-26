# FinTune — Production-Grade Financial NLP with QLoRA Fine-Tuning

End-to-end ML system for financial text sentiment classification: QLoRA fine-tuning on Mistral-7B, 4-bit quantized inference, AI guardrails, real-time monitoring, self-recovery, and FastAPI serving — built for production at scale.

## Architecture

```
┌─────────────┐    ┌──────────────┐    ┌───────────────┐    ┌──────────────┐
│  Financial   │───▶│  QLoRA       │───▶│  Quantized    │───▶│  FastAPI      │
│  Dataset     │    │  Fine-Tune   │    │  Inference    │    │  + Guardrails │
│  (HF Hub)    │    │  (PEFT)      │    │  (4-bit NF4)  │    │  + Monitoring │
└─────────────┘    └──────────────┘    └───────────────┘    └──────────────┘
       │                  │                    │                    │
   Data Pipeline     PyTorch + HF        bitsandbytes      Circuit Breaker
   + Quality QA      Transformers        double quant       Self-Recovery
   + Dedup           LoRA r=16/α=32      ~4x compression    Real-time Metrics
```

## Key Capabilities

| Domain | Implementation |
|---|---|
| **Fine-Tuning** | QLoRA (4-bit NF4) with PEFT LoRA adapters on Mistral-7B attention layers |
| **Training** | HF Trainer, cosine scheduler, paged AdamW 8-bit, early stopping, gradient checkpointing |
| **Data Pipeline** | Streaming loader, quality validation, deduplication, stratified sampling, caching |
| **Evaluation** | Per-class F1/precision/recall, confusion matrix, latency benchmarks (p50/p95/p99) |
| **Quantization** | Post-training 4-bit/8-bit via bitsandbytes for ~4x VRAM reduction |
| **AI Guardrails** | PII redaction (SSN, CC, email, phone), confidence thresholding, output validation |
| **Monitoring** | Real-time health scoring, latency histograms, drift detection (KL-divergence), throughput tracking |
| **Self-Recovery** | Circuit breaker pattern, auto model reload, OOM batch reduction, fallback model switching |
| **Serving** | FastAPI with `/predict`, `/predict/batch`, `/health`, `/metrics` endpoints |
| **Baselines** | Sklearn benchmarks (TF-IDF + LogReg/RF/SVM) for comparison metrics |
| **DevOps** | Docker + docker-compose (GPU), GitHub Actions CI (lint + test + build) |
| **Testing** | 35+ test cases covering data, model, guardrails, API, monitoring, recovery, pipeline |

## Quick Start

### Option 1: Full GPU Training (Recommended)

```bash
git clone https://github.com/narendranathe/fintune.git
cd fintune
pip install -r requirements.txt

# Run full pipeline: tests → benchmark → train → evaluate → quantize
bash scripts/run_local_gpu.sh

# Or on Windows with PowerShell:
.\scripts\run_local_gpu.ps1
```

### Option 2: CPU Baseline Benchmark

```bash
# Produces real metrics without GPU using sklearn
python -m src.benchmark
# Results saved to outputs/benchmark_results.json
```

### Option 3: QLoRA Training Only

```bash
python -m src.train --config configs/qlora_config.yaml
python -m src.evaluate --model-path outputs/fintune-financial --dataset test
```

### Option 4: Docker Deployment

```bash
docker-compose up --build
# API available at http://localhost:8000
# Health check: GET /health
# Metrics: GET /metrics
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | Single text classification with guardrails |
| `/predict/batch` | POST | Batch prediction for throughput workloads |
| `/health` | GET | Health check with monitoring metrics, circuit breaker state |
| `/metrics` | GET | Full system metrics export for dashboards |

**Example:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Revenue increased 20% year-over-year", "confidence_threshold": 0.7}'
```

**Response:**
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

## Project Structure

```
fintune/
├── src/
│   ├── __init__.py          # Package init, version 0.2.0
│   ├── data.py              # HF dataset loading, tokenization, stratified split
│   ├── data_pipeline.py     # Streaming loader, quality validation, dedup, caching
│   ├── model.py             # QLoRA config, BitsAndBytes 4-bit, PEFT LoRA adapters
│   ├── train.py             # Full training pipeline with HF Trainer
│   ├── evaluate.py          # Per-class metrics, confusion matrix, latency benchmarks
│   ├── quantize.py          # Post-training 4-bit/8-bit quantization
│   ├── guardrails.py        # PII redaction, confidence thresholding, audit logging
│   ├── serve.py             # FastAPI with monitoring + circuit breaker integration
│   ├── monitor.py           # Real-time health scoring, drift detection, metrics export
│   ├── self_recovery.py     # Circuit breaker, auto-remediation, graceful degradation
│   └── benchmark.py         # Sklearn baselines (TF-IDF + LogReg/RF/SVC)
├── configs/
│   ├── qlora_config.yaml          # Mistral-7B QLoRA config (GPU)
│   └── qlora_distilbert_cpu.yaml  # DistilBERT config (CPU testing)
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   ├── test_guardrails.py
│   ├── test_serve.py
│   ├── test_monitor.py
│   ├── test_self_recovery.py
│   └── test_data_pipeline.py
├── scripts/
│   ├── run_train.sh         # Original training script
│   ├── run_local_gpu.sh     # Full pipeline runner (Linux/WSL)
│   └── run_local_gpu.ps1    # Full pipeline runner (Windows)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .github/workflows/ci.yml
└── README.md
```

## System Design: Self-Recovery & Monitoring

### Circuit Breaker Pattern
The inference endpoint uses a three-state circuit breaker (CLOSED → OPEN → HALF_OPEN → CLOSED) that prevents cascading failures. When error rate exceeds threshold, the circuit opens and returns 503, allowing the system to recover autonomously.

### Real-Time Monitoring
`SystemMonitor` tracks latency percentiles, throughput, error rates, and prediction distribution in sliding windows. A composite health score (0-100) drives alerting. Model drift detection uses KL-divergence between current prediction distribution and baseline.

### Autonomous Recovery
`RecoveryManager` handles:
- **Model loading failures** → Retry with exponential backoff, fallback to quantized model
- **Latency spikes** → Switch to lighter quantized model
- **OOM errors** → Dynamically reduce batch size
- **Quality degradation** → Trigger model reload from checkpoint

## Dataset

Uses **financial_phrasebank** (Malo et al., 2014) from Hugging Face Hub — 4,845 financial news sentences with sentiment labels (positive/neutral/negative). All-agree subset used for highest annotation quality.

## Hardware Requirements

| Mode | Requirements | Time |
|---|---|---|
| **Sklearn Baseline** | CPU only, 4GB RAM | ~30 seconds |
| **QLoRA Fine-Tuning** | 1x GPU, ≥4GB VRAM | ~15 min on T4 |
| **Quantized Inference** | CPU or GPU | <50ms per request |
| **Docker Deployment** | NVIDIA Container Toolkit | Instant after build |

## Technologies

PyTorch, Hugging Face (Transformers, PEFT, Datasets, Evaluate, TRL), bitsandbytes, QLoRA, LoRA, Scikit-learn, FastAPI, Docker, GitHub Actions, Pydantic

## License

MIT
