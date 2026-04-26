"""Training loop using Hugging Face Trainer with QLoRA."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml
import numpy as np
from evaluate import load as load_metric
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback

from .data import build_tokenizer, load_financial_phrasebank, tokenize_dataset
from .model import (
    build_lora_config,
    build_quantization_config,
    load_model_with_qlora,
    merge_and_save,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

f1_metric = load_metric("f1")
accuracy_metric = load_metric("accuracy")
precision_metric = load_metric("precision")
recall_metric = load_metric("recall")


def compute_metrics(eval_pred) -> dict[str, float]:
    """Compute classification metrics for Trainer evaluation.

    Returns F1 (macro), accuracy, precision, recall — standard for
    imbalanced financial text datasets.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")
    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    prec = precision_metric.compute(predictions=predictions, references=labels, average="macro")
    rec = recall_metric.compute(predictions=predictions, references=labels, average="macro")

    return {
        "f1_macro": f1["f1"],
        "accuracy": acc["accuracy"],
        "precision_macro": prec["precision"],
        "recall_macro": rec["recall"],
    }


def build_training_args(config: dict) -> TrainingArguments:
    """Build TrainingArguments from config dict."""
    train_cfg = config.get("training", {})

    return TrainingArguments(
        output_dir=train_cfg.get("output_dir", "outputs/fintune-financial"),
        num_train_epochs=train_cfg.get("epochs", 3),
        per_device_train_batch_size=train_cfg.get("batch_size", 8),
        per_device_eval_batch_size=train_cfg.get("eval_batch_size", 16),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation", 4),
        learning_rate=float(train_cfg.get("learning_rate", 2e-4)),
        lr_scheduler_type=train_cfg.get("scheduler", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.06),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        fp16=False,
        bf16=train_cfg.get("bf16", True),
        gradient_checkpointing=True,
        evaluation_strategy="steps",
        eval_steps=train_cfg.get("eval_steps", 50),
        save_strategy="steps",
        save_steps=train_cfg.get("save_steps", 50),
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=10,
        report_to="none",
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        seed=42,
    )


def train(config_path: str) -> None:
    """End-to-end fine-tuning pipeline.

    1. Load config
    2. Load and tokenize dataset
    3. Build QLoRA model
    4. Train with HF Trainer
    5. Evaluate on held-out test set
    6. Merge adapters and save
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_name = config["model"]["name"]
    max_length = config["model"].get("max_length", 512)

    # Data
    logger.info("Step 1/5: Loading dataset...")
    dataset = load_financial_phrasebank()
    tokenizer = build_tokenizer(model_name, max_length=max_length)
    tokenized = tokenize_dataset(dataset, tokenizer, max_length=max_length)

    # Model
    logger.info("Step 2/5: Building QLoRA model...")
    lora_cfg = config.get("lora", {})
    quant_cfg = config.get("quantization", {})

    model = load_model_with_qlora(
        model_name=model_name,
        quantization_config=build_quantization_config(**quant_cfg),
        lora_config=build_lora_config(**lora_cfg),
    )

    # Trainer
    logger.info("Step 3/5: Starting training...")
    training_args = build_training_args(config)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()

    # Evaluate
    logger.info("Step 4/5: Evaluating on test set...")
    metrics = trainer.evaluate()
    logger.info("Test metrics: %s", metrics)

    # Save
    logger.info("Step 5/5: Merging adapters and saving...")
    output_dir = config["training"].get("output_dir", "outputs/fintune-financial")
    merge_and_save(model, output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info("Training complete. Model saved to %s", output_dir)
    logger.info("Final F1 (macro): %.4f | Accuracy: %.4f", metrics["eval_f1_macro"], metrics["eval_accuracy"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune LLM with QLoRA")
    parser.add_argument("--config", type=str, default="configs/qlora_config.yaml")
    args = parser.parse_args()
    train(args.config)
