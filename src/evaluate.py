"""Model evaluation: per-class metrics, confusion matrix, inference benchmarks."""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from .data import ID_TO_LABEL

logger = logging.getLogger(__name__)


def evaluate_model(
    model_path: str,
    dataset_split: str = "test",
    batch_size: int = 32,
) -> dict:
    """Run full evaluation on held-out set with per-class breakdown.

    Args:
        model_path: Path to merged model directory.
        dataset_split: Which split to evaluate.
        batch_size: Inference batch size.

    Returns:
        Dict with metrics, confusion matrix, and latency stats.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto",
    )

    dataset = load_dataset(
        "takala/financial_phrasebank", "sentences_allagree", trust_remote_code=True,
    )
    splits = dataset["train"].train_test_split(test_size=0.2, seed=42, stratify_by_column="label")
    test_data = splits[dataset_split]

    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        device_map="auto",
    )

    texts = test_data["sentence"]
    true_labels = test_data["label"]

    # Inference with latency tracking
    latencies = []
    predictions = []
    for text in texts:
        start = time.perf_counter()
        pred = classifier(text)[0]
        latencies.append(time.perf_counter() - start)
        predictions.append(pred["label"])

    # Map string labels back to ints
    label_to_id = {v: k for k, v in ID_TO_LABEL.items()}
    pred_ids = [label_to_id.get(p, -1) for p in predictions]

    # Metrics
    report = classification_report(
        true_labels, pred_ids,
        target_names=list(ID_TO_LABEL.values()),
        output_dict=True,
    )
    cm = confusion_matrix(true_labels, pred_ids)

    latency_arr = np.array(latencies) * 1000  # convert to ms

    results = {
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "latency_ms": {
            "mean": float(np.mean(latency_arr)),
            "p50": float(np.median(latency_arr)),
            "p95": float(np.percentile(latency_arr, 95)),
            "p99": float(np.percentile(latency_arr, 99)),
        },
        "total_samples": len(texts),
    }

    logger.info("Evaluation complete on %d samples", len(texts))
    logger.info("Macro F1: %.4f | Accuracy: %.4f", report["macro avg"]["f1-score"], report["accuracy"])
    logger.info("Latency - mean: %.1fms, p95: %.1fms", results["latency_ms"]["mean"], results["latency_ms"]["p95"])

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset", default="test")
    args = parser.parse_args()
    evaluate_model(args.model_path, args.dataset)
