# Ubiquitous Language — FinTune

A DDD-style glossary of the canonical terms used across this project's code, tests, configs, docs, and conversation. If a concept appears here, use **this** name for it; aliases are listed under "Aliases to avoid."

The goal: when a code reviewer says "the LoRA rank is too high," everyone reading that comment maps it to the same `r` parameter in `src/model.py` and the same trainable-parameter math. No translation layer.

---

## Table of contents

- [ML / Training](#ml--training)
- [Financial NLP](#financial-nlp)
- [Production / Serving](#production--serving)
- [Flagged ambiguities](#flagged-ambiguities)
- [Example dialogue](#example-dialogue)

---

## ML / Training

### QLoRA
**Definition:** A parameter-efficient fine-tuning method that combines 4-bit quantization of the frozen base model with trainable LoRA adapters.
**In this project:** Configured in `src/model.py::load_model_with_qlora()`; the canonical recipe lives in `configs/qlora_config.yaml`. This is *the* training method; we do not do full fine-tuning anywhere.
**Related:** LoRA, NF4 quantization, PEFT, double quantization.

### LoRA
**Definition:** Low-Rank Adaptation — adds two small matrices (`A`, `B` of rank `r`) to selected linear layers and trains only those, freezing everything else.
**In this project:** Built by `build_lora_config()` in `src/model.py`. We attach to attention projections only, not MLP layers.
**Related:** QLoRA, PEFT, adapter, rank.

### PEFT
**Definition:** Parameter-Efficient Fine-Tuning — the umbrella library (`peft` from HuggingFace) and the family of techniques (LoRA, IA³, prefix-tuning, etc.) that train a small fraction of model parameters.
**In this project:** The HuggingFace `peft` package; `get_peft_model()` wraps the base model with LoRA adapters in `src/model.py`.
**Related:** LoRA, QLoRA, adapter.

### Adapter
**Definition:** A small set of trainable parameters inserted into a frozen base model to specialize it for a downstream task without modifying the original weights.
**In this project:** Specifically the LoRA adapter matrices attached to attention projections; the base model stays frozen.
**Related:** LoRA, PEFT, merge and save.

### Rank (`r`)
**Definition:** The inner dimension of the LoRA matrices `A` (d × r) and `B` (r × d); higher rank means more adapter capacity and more trainable parameters.
**In this project:** `r=16` for the Mistral-7B GPU config, `r=8` for the DistilBERT CPU config. Trade-off between expressivity and overfitting.
**Related:** LoRA alpha, target modules.

### LoRA alpha (`lora_alpha`)
**Definition:** Scaling factor applied to the LoRA output before adding to the frozen layer's output; effective scale is `alpha / r`.
**In this project:** `lora_alpha=32` paired with `r=16` (effective scale 2.0). Convention is `alpha = 2 * r`.
**Related:** Rank.

### Target modules
**Definition:** The named submodules of the base model where LoRA adapters are attached.
**In this project:** `["q_proj", "k_proj", "v_proj", "o_proj"]` for Mistral-style models; `["q_lin", "v_lin"]` for DistilBERT (different naming convention). MLP layers are deliberately excluded.
**Related:** LoRA, rank.

### NF4 quantization
**Definition:** "NormalFloat 4-bit" — a 4-bit data type from the QLoRA paper designed for the actual distribution of pre-trained model weights (roughly normal), giving better quality than naive INT4.
**In this project:** Used in `build_quantization_config()` via `bnb_4bit_quant_type="nf4"`. Activated by `load_in_4bit=True`.
**Related:** Double quantization, compute dtype, bitsandbytes.

### Double quantization
**Definition:** Quantizing the quantization constants themselves to save additional memory (~0.4 bits/parameter).
**In this project:** Enabled via `bnb_4bit_use_double_quant=True` in `build_quantization_config()`. Always on for the GPU config.
**Related:** NF4 quantization.

### Compute dtype
**Definition:** The precision used during forward/backward passes for quantized layers — the weights stay in 4-bit storage but are dequantized to this dtype during compute.
**In this project:** `bfloat16` (`bnb_4bit_compute_dtype="bfloat16"`). bf16 over fp16 because of better numeric range on modern GPUs.
**Related:** NF4 quantization.

### Quant type
**Definition:** The quantization data type — either `nf4` (NormalFloat 4-bit) or `fp4` (vanilla FP4) for 4-bit, or INT8 for 8-bit.
**In this project:** Always `nf4` for training (`build_quantization_config()`) and for post-training quantization in `src/quantize.py`. `fp4` is exposed as an arg but never used.
**Related:** NF4 quantization.

### Fine-tuning
**Definition:** Continuing training of a pre-trained model on task-specific data, updating some or all parameters.
**In this project:** Specifically QLoRA fine-tuning — only the LoRA adapters update. Driven by `src/train.py`. Distinct from pre-training (we never do that) and RLHF (planned, not built).
**Related:** QLoRA, pre-training, RLHF.

### Pre-training
**Definition:** The initial self-supervised training of a base model on a large general corpus (Wikipedia, BookCorpus, web text).
**In this project:** Out of scope — we always start from a HuggingFace pre-trained checkpoint (`mistralai/Mistral-7B-v0.3` or `distilbert-base-uncased`).
**Related:** Fine-tuning, base model.

### RLHF
**Definition:** Reinforcement Learning from Human Feedback — a post-training alignment pass that uses human preference rankings to steer model behavior.
**In this project:** Not implemented; listed in `specs/README.md` open issues as a future alignment step. The `trl` package is in `requirements.txt` for when this lands.
**Related:** Fine-tuning, alignment.

### Training loss
**Definition:** The loss value computed on training batches during the forward pass; what the optimizer is actively minimizing.
**In this project:** Standard cross-entropy from the HF `Trainer` for sequence classification. Logged every 10 steps (`logging_steps=10`).
**Related:** Eval loss.

### Eval loss
**Definition:** The same loss function computed on the held-out evaluation set; the signal for early stopping and best-checkpoint selection.
**In this project:** Computed every `eval_steps=50` steps. We *don't* select the best model on eval loss — we use F1-macro instead (`metric_for_best_model="f1_macro"`).
**Related:** Training loss, F1 score, early stopping.

### F1 score (macro)
**Definition:** The unweighted mean of per-class F1 scores; treats every class equally regardless of frequency.
**In this project:** Our **primary metric**. Computed in `src/train.py::compute_metrics()` with `average="macro"`. Used for best-checkpoint selection and reported in `specs/README.md`.
**Related:** F1 score (weighted), precision, recall, confusion matrix.

### F1 score (weighted)
**Definition:** Per-class F1 weighted by class support (number of true instances), giving frequent classes more influence.
**In this project:** Reported by the sklearn baselines in `src/benchmark.py` as a secondary signal but **not** the metric we optimize against. Macro is fairer on the class-imbalanced financial_phrasebank.
**Related:** F1 score (macro).

### Precision
**Definition:** Of the predictions for class X, the fraction that were actually class X. `TP / (TP + FP)`.
**In this project:** Reported macro-averaged in `compute_metrics()`. In the guardrails / PII context, "precision" specifically means low false-positive PII detection (see [Recall (PII)](#recall-pii)).
**Related:** Recall, F1 score.

### Recall
**Definition:** Of the actual instances of class X, the fraction we caught. `TP / (TP + FN)`.
**In this project:** Reported macro-averaged in `compute_metrics()`. See [Recall (PII)](#recall-pii) for the guardrails-specific use.
**Related:** Precision, F1 score, false positive rate.

### Confusion matrix
**Definition:** A square matrix where row `i`, column `j` is the count of samples whose true label is `i` and predicted label is `j`.
**In this project:** Computed in `src/evaluate.py` via sklearn and saved per-baseline as PNG heatmaps in `outputs/`. Diagonal = correct predictions.
**Related:** F1 score, precision, recall.

### Cosine scheduler
**Definition:** A learning rate schedule that follows half a cosine wave from the peak LR down to (near) zero over training.
**In this project:** `lr_scheduler_type="cosine"` in `build_training_args()`. Paired with a warmup ratio of 0.06.
**Related:** Learning rate warmup.

### Learning rate warmup
**Definition:** A short initial phase where the learning rate ramps from 0 up to its peak, before the main schedule takes over.
**In this project:** `warmup_ratio=0.06` (6% of total steps). Helps QLoRA training stability.
**Related:** Cosine scheduler.

### Early stopping
**Definition:** Halting training when a validation metric stops improving for `patience` consecutive evals, to avoid overfitting.
**In this project:** `EarlyStoppingCallback(early_stopping_patience=3)` in `src/train.py`. Watches `f1_macro` (since `metric_for_best_model="f1_macro"`).
**Related:** Eval loss, F1 score.

### Gradient checkpointing
**Definition:** Trading compute for memory by re-computing activations during the backward pass instead of storing them.
**In this project:** Always on (`gradient_checkpointing=True` in training args; `use_gradient_checkpointing=True` in `prepare_model_for_kbit_training()`). Required to fit Mistral-7B QLoRA in <8 GB VRAM.
**Related:** Batch size, QLoRA.

### Batch size
**Definition:** Number of training examples processed per forward/backward pass. Effective batch size = `per_device_batch_size * gradient_accumulation_steps * num_devices`.
**In this project:** `batch_size=8` per device × `gradient_accumulation=4` = effective 32 for the GPU config. Reduced dynamically by `RecoveryManager.handle_oom_error()` if OOM is detected at runtime.
**Related:** Gradient checkpointing, OOM, gradient accumulation.

### Tokenizer
**Definition:** The component that converts raw text into token IDs (and back). Pre-trained alongside the base model.
**In this project:** Built by `src/data.py::build_tokenizer()`. Pad token is set to EOS if the base model lacks one (Mistral does); `model_max_length=512`.
**Related:** Input IDs, attention mask, labels.

### Input IDs
**Definition:** Integer token indices produced by the tokenizer, one per token in the input sequence.
**In this project:** Output of `tokenize_dataset()` in `src/data.py`. Fed to the model alongside the attention mask and labels.
**Related:** Tokenizer, attention mask, labels.

### Attention mask
**Definition:** A 0/1 mask per token telling the model which positions are real tokens vs. padding (so attention ignores padding).
**In this project:** Auto-produced by the tokenizer when `padding="max_length"` is set.
**Related:** Tokenizer, input IDs.

### Labels
**Definition:** Target values for supervised training — for sequence classification, integer class IDs.
**In this project:** `0=negative`, `1=neutral`, `2=positive` from `LABEL_MAP` in `src/data.py`. Attached to the tokenized example dict by `tokenize_dataset()`.
**Related:** Input IDs, label map.

### Checkpoint
**Definition:** A saved snapshot of model weights (and optionally optimizer state) at a particular training step.
**In this project:** HF `Trainer` writes checkpoints to `output_dir` every `save_steps=50`. `save_total_limit=2` keeps only the two most recent.
**Related:** Merge and save, best model.

### Merge and save
**Definition:** Folding the trained LoRA adapter matrices `A @ B` back into the original frozen base weights, then saving the result as a single dense model that no longer needs PEFT to load.
**In this project:** `src/model.py::merge_and_save()`, called at the end of `src/train.py`. The merged model is what gets quantized in `src/quantize.py` and served by `src/serve.py`.
**Related:** LoRA, GGUF export, quantize for inference.

### GGUF export
**Definition:** Converting a HuggingFace-format model to the `.gguf` format used by `llama.cpp` for CPU inference.
**In this project:** Not implemented; listed in `specs/README.md` open issues. Would happen after `merge_and_save()`.
**Related:** Merge and save, quantized inference.

---

## Financial NLP

### Financial sentiment
**Definition:** A 3-class label — `positive`, `negative`, `neutral` — applied to a financial text fragment, where the polarity is assessed *from an investor's perspective* (so "rates rose" is positive for a bank but the phrase alone is neutral without context).
**In this project:** The single classification task. Labels in `src/data.py::LABEL_MAP`. Distinct from generic sentiment because of the perspective-conditioning.
**Related:** financial_phrasebank, label map.

### financial_phrasebank
**Definition:** A public dataset of 4,845 English financial news sentences with sentiment labels annotated by 16 finance-background annotators (Malo et al., 2014, "Good debt or bad debt: Detecting semantic orientations in economic texts," *Journal of the Association for Information Science and Technology*).
**In this project:** Loaded as `takala/financial_phrasebank`, subset `sentences_allagree`, in `src/data.py::load_financial_phrasebank()`. The "all-agree" subset is used for highest signal quality.
**Related:** Financial sentiment, all-agree subset, stratified split.

### All-agree subset
**Definition:** The subset of financial_phrasebank where 100% of annotators agreed on the label, vs. weaker agreement levels (≥75%, ≥66%, ≥50%).
**In this project:** Hardcoded as `agreement_level="sentences_allagree"` in `load_financial_phrasebank()`. Trades dataset size for label noise.
**Related:** financial_phrasebank.

### Earnings beat / miss
**Definition:** Reported earnings exceeding (beat) or falling short of (miss) analyst consensus expectations.
**In this project:** Target phrasing the model should classify as `positive` (beat) or `negative` (miss). Not a label name itself.
**Related:** Financial sentiment, guidance.

### Guidance
**Definition:** A company's forward-looking statement about expected future financial performance, usually issued during earnings releases.
**In this project:** "Raised guidance" → positive; "cut guidance" / "withdrew guidance" → negative. Hard for generic LLMs because the polarity depends on the verb, not the noun.
**Related:** Earnings beat / miss.

### Write-down / impairment
**Definition:** A reduction in the carrying value of an asset on the balance sheet, recognizing that its market value has fallen below book value. Always negative sentiment.
**In this project:** Examples of finance-specific vocabulary that justifies a domain fine-tune; a generic LLM may miss the polarity if it doesn't know "impairment" implies loss.
**Related:** Financial sentiment.

### EBITDA
**Definition:** Earnings Before Interest, Taxes, Depreciation, and Amortization — a profitability proxy that strips out capital-structure and non-cash effects.
**In this project:** Vocabulary the model needs to understand at a domain level; "EBITDA margin expanded" should classify as positive.
**Related:** Earnings beat / miss.

### MNPI (Material Non-Public Information)
**Definition:** Information about a company that is not publicly disclosed and that a reasonable investor would consider material to a trading decision. Trading on MNPI is illegal (insider trading).
**In this project:** A motivating compliance concern for the guardrails layer — a model deployed inside a financial firm must not generate or echo content that could constitute MNPI disclosure. Currently addressed indirectly via PII redaction; explicit MNPI controls are out of scope for v0.x.
**Related:** PII (financial), guardrails.

### PII (financial context)
**Definition:** Personally Identifiable Information — for finance specifically: SSN, credit card numbers, bank account numbers, brokerage account numbers, customer email/phone tied to financial accounts.
**In this project:** Detected and redacted by `src/guardrails.py::redact_pii()` using the `PII_PATTERNS` regex dict (keys: `ssn`, `credit_card`, `email`, `phone`, `account_number`).
**Related:** Guardrails, PII redaction, MNPI.

### SEC filings
**Definition:** Documents companies are required to file with the U.S. Securities and Exchange Commission, available via EDGAR.
**In this project:** Out-of-domain for v0.x training data (financial_phrasebank is news, not filings) but flagged in `specs/README.md` as a future training corpus.
**Related:** 10-K, 10-Q.

### 10-K
**Definition:** An annual SEC filing required of public U.S. companies, containing audited financial statements, risk factors, MD&A, etc.
**In this project:** Not currently used; future fine-tuning target for long-form financial text.
**Related:** SEC filings, 10-Q.

### 10-Q
**Definition:** A quarterly SEC filing required of public U.S. companies, containing unaudited financial statements and management commentary.
**In this project:** Not currently used; future fine-tuning target.
**Related:** SEC filings, 10-K.

### Earnings call transcript
**Definition:** The text record of a quarterly earnings conference call between a company's executives and analysts.
**In this project:** Not currently a training corpus; flagged as future work because it's where a lot of nuanced sentiment lives ("we're pleased with the trajectory" = guarded positive).
**Related:** Earnings beat / miss, guidance.

---

## Production / Serving

### Guardrails
**Definition:** Pre- and post-inference checks that constrain model behavior to comply with safety, privacy, and quality requirements before the response leaves the service.
**In this project:** Implemented in `src/guardrails.py`. Specifically: PII redaction on input, confidence thresholding on output, label validation. Aggregated by `apply_guardrails()` returning a `GuardrailResult`.
**Related:** PII redaction, confidence threshold, GuardrailResult.

### PII redaction
**Definition:** Detecting PII in text and replacing it with a placeholder marker before further processing or logging.
**In this project:** `src/guardrails.py::redact_pii()` runs the input text through `PII_PATTERNS` and substitutes with `[REDACTED_<TYPE>]` markers. Returns `(redacted_text, list_of_pii_types_found)`.
**Related:** PII, guardrails, false positive rate.

### Confidence threshold
**Definition:** A minimum prediction confidence below which the model's output is flagged as low-confidence and should be escalated rather than auto-served.
**In this project:** Default `0.7`, configurable per-request in `PredictRequest`. Enforced by `check_confidence()`. Not the same as a refusal — the prediction is still returned, but `passed=False` and `flags` contains `LOW_CONFIDENCE`.
**Related:** Guardrails, GuardrailResult.

### Circuit breaker
**Definition:** A 3-state machine (`CLOSED` → `OPEN` → `HALF_OPEN` → `CLOSED`) that fails fast after a failure threshold and probes for recovery after a timeout.
**In this project:** `src/self_recovery.py::CircuitBreaker`. Defaults: `failure_threshold=5`, `recovery_timeout_sec=30`, `half_open_success_threshold=2`. `OPEN` causes `/predict` to return 503.
**Related:** Self-recovery, RecoveryManager, RecoveryPolicy.

### Self-recovery
**Definition:** Automatic remediation actions taken by the service in response to detected faults, without human intervention.
**In this project:** `src/self_recovery.py::RecoveryManager` orchestrates: model-load retry with exponential backoff, batch-size reduction on OOM, model reload on quality degradation, switch to quantized model on persistent latency spikes.
**Related:** Circuit breaker, RecoveryManager, RecoveryAction.

### RecoveryManager
**Definition:** The class coordinating self-recovery actions; owns a `CircuitBreaker`, a `RecoveryPolicy`, and an audit trail of `RecoveryAction` records.
**In this project:** Defined in `src/self_recovery.py`. Instantiated as `_recovery_manager` global in `src/serve.py`.
**Related:** Circuit breaker, self-recovery, RecoveryPolicy.

### Drift detection
**Definition:** Monitoring for statistical change in model inputs (data drift) or outputs (concept drift / prediction drift) over time relative to a baseline.
**In this project:** `src/monitor.py::SystemMonitor._detect_drift()` computes KL-divergence between the recent prediction distribution and a baseline distribution; alerts when divergence exceeds `kl_divergence_threshold=0.15`.
**Related:** KL divergence, concept drift, data drift.

### KL divergence
**Definition:** Kullback–Leibler divergence — a non-symmetric measure of how one probability distribution diverges from a reference distribution. `D(P||Q) = sum P(x) log(P(x)/Q(x))`.
**In this project:** Computed over the categorical prediction distribution (negative / neutral / positive) in `SystemMonitor._detect_drift()`. Threshold of 0.15 was chosen empirically; not tuned.
**Related:** Drift detection.

### Concept drift
**Definition:** A change in the relationship between inputs and the correct outputs over time (the underlying task changes — e.g., new financial slang).
**In this project:** Detected indirectly via prediction-distribution drift; we don't have ground truth in production so true concept drift is not directly observable.
**Related:** Data drift, drift detection.

### Data drift
**Definition:** A change in the distribution of inputs over time, while the input-output relationship may stay the same.
**In this project:** Same observability story as concept drift — proxied via output distribution shifts.
**Related:** Concept drift, drift detection.

### Quantized inference
**Definition:** Running inference with model weights in a low-precision format (4-bit / 8-bit) instead of fp16/bf16/fp32, to reduce memory and (sometimes) latency.
**In this project:** `src/quantize.py::quantize_for_inference()` post-processes the merged model into a 4-bit NF4 or 8-bit INT8 variant saved to `outputs/fintune-financial-quantized`. Used as the fallback model in `src/serve.py`.
**Related:** NF4 quantization, merge and save.

### Model serving
**Definition:** Exposing a trained model behind a network API that handles requests, batching, scaling, monitoring, and faults.
**In this project:** FastAPI-based, `src/serve.py`. Endpoints: `/predict`, `/predict/batch`, `/health`, `/metrics`. Lifespan loads the model on startup with fallback.
**Related:** Health endpoint, circuit breaker.

### Health endpoint
**Definition:** An HTTP endpoint that reports whether the service is functioning, including diagnostic metadata for orchestrators.
**In this project:** `GET /health` in `src/serve.py`. Returns `status` (`healthy` / `degraded`), `model_loaded`, `health_score` (0–100 from `SystemMonitor`), `circuit_breaker_state`, `total_requests`, `error_rate`.
**Related:** Health score, circuit breaker, SystemMonitor.

### Health score
**Definition:** A composite scalar (0–100) summarizing service health derived from error rate, p99 latency, and throughput.
**In this project:** Computed by `SystemMonitor._compute_health_score()`. Penalties: error rate (up to −50), latency p99 above 100ms (up to −25), low throughput below 1 req/sec (up to −25).
**Related:** Health endpoint, SystemMonitor.

### SystemMonitor
**Definition:** The thread-safe singleton that aggregates per-request observability data and exposes snapshot metrics.
**In this project:** `src/monitor.py::SystemMonitor`. Tracks latency percentiles in a sliding window of 1000 samples, throughput over a 1-hour window, prediction distribution for drift over 500 samples.
**Related:** Health score, drift detection, latency percentiles.

### Latency percentiles
**Definition:** p50 (median), p95, p99 — the latency thresholds below which 50%, 95%, 99% of requests complete.
**In this project:** Computed in `SystemMonitor._calculate_latency_percentiles()` from the sliding window. Surfaced via `/metrics` and used by `RecoveryManager.handle_latency_spike()` to trigger model swaps.
**Related:** SystemMonitor, throughput.

### Throughput
**Definition:** Requests served per unit time (here: requests per second).
**In this project:** Computed in `SystemMonitor._calculate_throughput()` over a 1-hour sliding window. Feeds into the health score.
**Related:** Latency percentiles, health score.

### False positive rate (guardrails context)
**Definition:** Fraction of clean (no PII) inputs that the PII detector incorrectly flags as containing PII.
**In this project:** A known concern for the regex-based detector, especially `account_number` which matches any 8–17 digit string. We deliberately accept high FPR to keep the false-negative rate low.
**Related:** PII redaction, recall (PII).

### Recall (PII)
**Definition:** Fraction of actual PII instances that the detector successfully catches.
**In this project:** The metric that matters most for guardrails — missing real PII is a compliance incident, over-redacting is just annoying. We optimize for recall over precision.
**Related:** PII redaction, false positive rate.

### CircuitState
**Definition:** The enum of states a `CircuitBreaker` can occupy: `CLOSED`, `OPEN`, `HALF_OPEN`.
**In this project:** Defined in `src/self_recovery.py::CircuitState`. Surfaced in `/health` response.
**Related:** Circuit breaker.

### RecoveryPolicy
**Definition:** A dataclass of all configurable thresholds and timeouts for the self-recovery system (failure thresholds, backoff intervals, batch-size reduction factors, etc.).
**In this project:** `src/self_recovery.py::RecoveryPolicy`. Single source of truth for recovery behavior; pass a custom instance to `RecoveryManager` to override defaults.
**Related:** RecoveryManager, circuit breaker.

### RecoveryAction
**Definition:** An audit-log record of one recovery action taken (timestamp, type, reason, success, details).
**In this project:** `src/self_recovery.py::RecoveryAction`. Accumulated in `RecoveryManager._recovery_actions` and exposed via `get_recovery_history()`.
**Related:** Self-recovery, RecoveryManager.

### GuardrailResult
**Definition:** The dataclass returned by `apply_guardrails()` carrying the redacted text, prediction, confidence, pass/fail flag, list of flags, list of PII types found, and a UTC timestamp.
**In this project:** `src/guardrails.py::GuardrailResult`. The single object that the serving layer reads to construct the API response.
**Related:** Guardrails, PII redaction, confidence threshold.

---

## Flagged ambiguities

| Ambiguous term | Resolution in this project |
|----------------|----------------------------|
| **"Sentiment"** | Always means **financial sentiment** (3-class, perspective-conditioned). Never used for generic / movie-review sentiment. |
| **"Recall"** | Means the classification metric (`TP / (TP + FN)` over class labels) **unless** the context is guardrails / PII, where it means [Recall (PII)](#recall-pii). The financial-sentiment usage is the default; PII usage must be explicitly qualified. |
| **"Precision"** | Same — classification metric by default, PII detection precision must be qualified. |
| **"Quantization"** | Refers to **NF4 4-bit quantization** unless explicitly suffixed (e.g. "8-bit quantization"). The training pipeline is 4-bit; 8-bit is only an option in `src/quantize.py`. |
| **"Drift"** | Means **prediction-distribution drift detected via KL-divergence** in `SystemMonitor`. We don't currently distinguish data drift from concept drift in code (we can't — no ground truth at serving time). |
| **"Health"** | Means the composite `SystemMonitor.health_score` (0–100), not a Kubernetes liveness/readiness check. The `/health` endpoint surfaces both. |
| **"Adapter"** | Always means a **LoRA adapter** in this project. Not the adapter-layer technique (Houlsby et al., 2019). |
| **"Threshold"** | Ambiguous on its own — always qualify: confidence threshold (guardrails), KL divergence threshold (drift), failure threshold (circuit breaker), imbalance threshold (data quality). |

### Aliases to avoid

- ❌ "fine-tune" with no qualifier when QLoRA is meant → ✅ "QLoRA fine-tune" (we never do full fine-tuning)
- ❌ "lora rank", "LoRA dimension" → ✅ "rank" or `r`
- ❌ "PII filter" / "PII scrubber" → ✅ "PII redaction"
- ❌ "model breaker" / "trip switch" → ✅ "circuit breaker"
- ❌ "adapter weights" with no context → ✅ "LoRA adapter"
- ❌ "4-bit" alone → ✅ "NF4 quantization" (specifies the data type)
- ❌ "score" alone in serving context → ✅ "health score" or "confidence"

---

## Example dialogue

> **Reviewer:** The training run OOM'd at step 200 even with `r=16`. Should we drop the rank?
>
> **Implementer:** Rank's not the bottleneck — at `r=16` on `q/k/v/o_proj` the LoRA adapters add maybe 10 MB. The OOM is from the activations. Did gradient checkpointing actually take effect?
>
> **Reviewer:** Hmm. It's set in `build_training_args`, but `prepare_model_for_kbit_training` is what actually wires it into the base model. Let me check.
>
> **Implementer:** Right — and check the compute dtype too. If it fell back to fp32 instead of bfloat16, that doubles activation memory even with NF4 storage.
>
> **Reviewer:** Confirmed: `bnb_4bit_compute_dtype` was bfloat16. Gradient checkpointing was on. Real culprit was `gradient_accumulation=4` × `batch_size=8` — effective batch 32 was too aggressive.
>
> **Implementer:** OK. Rather than dropping the rank — which would hurt F1-macro — let's let the `RecoveryManager` handle it: it'll halve the batch size on OOM via `handle_oom_error()` and retry. Confirm `oom_min_batch_size=1` so we don't stall.
>
> **Reviewer:** Done. Restarted, batch reduced to 4, training resumed. F1-macro on eval is 0.78 at step 400 — about a point below the previous best checkpoint. Worth noting the circuit-breaker state stayed CLOSED throughout, so no spurious 503s during the recovery.
