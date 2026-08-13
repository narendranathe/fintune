"""Baseline benchmark using scikit-learn for comparison metrics.

Trains TF-IDF + LogisticRegression, TF-IDF + RandomForest, and TF-IDF + LinearSVC
on financial_phrasebank to establish baseline metrics that QLoRA fine-tuned models
should beat.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

logger = logging.getLogger(__name__)


class SklearnBaselineBenchmark:
    """Manages sklearn baseline benchmarking against financial_phrasebank."""

    def __init__(self, output_dir: str = "outputs"):
        """Initialize benchmark.

        Args:
            output_dir: Directory to save results and plots.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: dict[str, Any] = {}

    def load_financial_phrasebank(
        self, train_size: float = 0.8, seed: int = 42
    ) -> tuple[list[str], list[str], list[str], list[str]]:
        """Load financial_phrasebank from HuggingFace datasets.

        Args:
            train_size: Fraction of data for training (remaining for test).
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (X_train, X_test, y_train, y_test) with stratified split.
        """
        try:
            from datasets import load_dataset
        except ImportError:
            logger.error("datasets library not installed. Install with: pip install datasets")
            raise

        logger.info("Loading financial_phrasebank dataset...")
        dataset = load_dataset("takala/financial_phrasebank", "sentences_allagree", trust_remote_code=True)

        texts = dataset["train"]["sentence"]
        labels = dataset["train"]["label"]

        # Map labels to strings
        label_map = {0: "negative", 1: "neutral", 2: "positive"}
        labels = [label_map[label] for label in labels]

        # Stratified train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=1 - train_size, random_state=seed, stratify=labels
        )

        logger.info(
            f"Loaded {len(texts)} samples. Train: {len(X_train)}, Test: {len(X_test)}"
        )
        logger.info(f"Label distribution (train): {dict(zip(*np.unique(y_train, return_counts=True)))}")

        return X_train, X_test, y_train, y_test

    def build_tfidf_logreg_pipeline(self) -> Pipeline:
        """Build TF-IDF + LogisticRegression pipeline.

        Returns:
            Fitted sklearn Pipeline.
        """
        return Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        max_features=10000,
                        min_df=2,
                        max_df=0.95,
                    ),
                ),
                ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
            ]
        )

    def build_tfidf_rf_pipeline(self) -> Pipeline:
        """Build TF-IDF + RandomForest pipeline.

        Returns:
            Fitted sklearn Pipeline.
        """
        return Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        max_features=10000,
                        min_df=2,
                        max_df=0.95,
                    ),
                ),
                ("clf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
            ]
        )

    def build_tfidf_svc_pipeline(self) -> Pipeline:
        """Build TF-IDF + LinearSVC pipeline.

        Returns:
            Fitted sklearn Pipeline.
        """
        return Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        max_features=10000,
                        min_df=2,
                        max_df=0.95,
                    ),
                ),
                ("clf", LinearSVC(C=1.0, max_iter=2000, random_state=42)),
            ]
        )

    def evaluate_model(
        self,
        model: Pipeline,
        X_test: list[str],
        y_test: list[str],
        model_name: str,
    ) -> dict[str, Any]:
        """Evaluate model on test set.

        Args:
            model: Trained sklearn Pipeline.
            X_test: Test texts.
            y_test: Test labels.
            model_name: Name of the model for logging.

        Returns:
            Dict with metrics, confusion matrix, and classification report.
        """
        logger.info(f"Evaluating {model_name}...")

        # Predictions
        y_pred = model.predict(X_test)

        # Overall metrics
        accuracy = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        weighted_f1 = f1_score(y_test, y_pred, average="weighted")

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average=None, zero_division=0
        )

        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Classification report
        class_report = classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        )

        results = {
            "model_name": model_name,
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "per_class_metrics": {
                "precision": [float(p) for p in precision],
                "recall": [float(r) for r in recall],
                "f1": [float(f) for f in f1],
                "support": [int(s) for s in support],
            },
            "confusion_matrix": conf_matrix.tolist(),
            "classification_report": class_report,
        }

        logger.info(f"{model_name} - Accuracy: {accuracy:.4f}, Macro F1: {macro_f1:.4f}")

        return results

    def benchmark_inference_latency(
        self,
        model: Pipeline,
        X_test: list[str],
        num_samples: int = 100,
    ) -> dict[str, float]:
        """Benchmark inference latency per sample.

        Args:
            model: Trained sklearn Pipeline.
            X_test: Test texts.
            num_samples: Number of samples to use for latency benchmark.

        Returns:
            Dict with mean, median, min, max latency in milliseconds.
        """
        logger.info("Benchmarking inference latency...")

        sample_texts = X_test[:num_samples]
        latencies = []

        for text in sample_texts:
            start = time.time()
            model.predict([text])
            elapsed = (time.time() - start) * 1000  # ms

            latencies.append(elapsed)

        return {
            "mean_ms": float(np.mean(latencies)),
            "median_ms": float(np.median(latencies)),
            "min_ms": float(np.min(latencies)),
            "max_ms": float(np.max(latencies)),
            "std_ms": float(np.std(latencies)),
        }

    def save_confusion_matrix_plot(
        self, conf_matrix: list[list[int]], labels: list[str], model_name: str
    ) -> str:
        """Save confusion matrix as a plot image.

        Args:
            conf_matrix: Confusion matrix from sklearn.
            labels: Class labels.
            model_name: Name of the model.

        Returns:
            Path to saved plot.
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            plt.figure(figsize=(8, 6))
            sns.heatmap(
                conf_matrix,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=labels,
                yticklabels=labels,
            )
            plt.title(f"Confusion Matrix - {model_name}")
            plt.ylabel("True Label")
            plt.xlabel("Predicted Label")
            plt.tight_layout()

            filepath = self.output_dir / f"confusion_matrix_{model_name.replace(' ', '_').lower()}.png"
            plt.savefig(filepath, dpi=100)
            plt.close()

            logger.info(f"Saved confusion matrix plot to {filepath}")
            return str(filepath)
        except ImportError:
            logger.warning("matplotlib/seaborn not installed, skipping plot generation")
            return ""

    def save_classification_report(
        self, class_report: dict[str, Any], model_name: str
    ) -> str:
        """Save classification report as text file.

        Args:
            class_report: Classification report dict from sklearn.
            model_name: Name of the model.

        Returns:
            Path to saved report.
        """
        filepath = self.output_dir / f"classification_report_{model_name.replace(' ', '_').lower()}.txt"

        with open(filepath, "w") as f:
            f.write(f"Classification Report - {model_name}\n")
            f.write("=" * 60 + "\n\n")

            # Per-class metrics
            for label, metrics in class_report.items():
                if label not in ["accuracy", "macro avg", "weighted avg"]:
                    f.write(f"\nClass: {label}\n")
                    f.write(f"  Precision: {metrics.get('precision', 0):.4f}\n")
                    f.write(f"  Recall:    {metrics.get('recall', 0):.4f}\n")
                    f.write(f"  F1-Score:  {metrics.get('f1-score', 0):.4f}\n")
                    f.write(f"  Support:   {int(metrics.get('support', 0))}\n")

            # Macro/weighted averages
            if "macro avg" in class_report:
                f.write("\nMacro Average:\n")
                f.write(f"  Precision: {class_report['macro avg'].get('precision', 0):.4f}\n")
                f.write(f"  Recall:    {class_report['macro avg'].get('recall', 0):.4f}\n")
                f.write(f"  F1-Score:  {class_report['macro avg'].get('f1-score', 0):.4f}\n")

            if "weighted avg" in class_report:
                f.write("\nWeighted Average:\n")
                f.write(f"  Precision: {class_report['weighted avg'].get('precision', 0):.4f}\n")
                f.write(f"  Recall:    {class_report['weighted avg'].get('recall', 0):.4f}\n")
                f.write(f"  F1-Score:  {class_report['weighted avg'].get('f1-score', 0):.4f}\n")

        logger.info(f"Saved classification report to {filepath}")
        return str(filepath)


def run_sklearn_baseline() -> dict[str, Any]:
    """Run all sklearn baselines and return comprehensive results.

    Returns:
        Dict with results from all three models (LogReg, RF, SVC).
    """
    logger.info("Starting sklearn baseline benchmark...")

    benchmark = SklearnBaselineBenchmark()

    # Load data
    X_train, X_test, y_train, y_test = benchmark.load_financial_phrasebank()
    class_labels = sorted(set(y_train))

    results = {
        "dataset": "financial_phrasebank",
        "split": {"train": len(X_train), "test": len(X_test)},
        "models": {},
    }

    # Model 1: TF-IDF + LogisticRegression
    logger.info("Training TF-IDF + LogisticRegression...")
    start = time.time()
    model_lr = benchmark.build_tfidf_logreg_pipeline()
    model_lr.fit(X_train, y_train)
    train_time_lr = time.time() - start

    eval_lr = benchmark.evaluate_model(model_lr, X_test, y_test, "TF-IDF + LogisticRegression")
    latency_lr = benchmark.benchmark_inference_latency(model_lr, X_test)
    eval_lr["training_time_seconds"] = train_time_lr
    eval_lr["inference_latency_ms"] = latency_lr

    benchmark.save_confusion_matrix_plot(
        eval_lr["confusion_matrix"], class_labels, "tfidf_logreg"
    )
    benchmark.save_classification_report(
        eval_lr["classification_report"], "tfidf_logreg"
    )

    results["models"]["tfidf_logreg"] = eval_lr

    # Model 2: TF-IDF + RandomForest
    logger.info("Training TF-IDF + RandomForest...")
    start = time.time()
    model_rf = benchmark.build_tfidf_rf_pipeline()
    model_rf.fit(X_train, y_train)
    train_time_rf = time.time() - start

    eval_rf = benchmark.evaluate_model(model_rf, X_test, y_test, "TF-IDF + RandomForest")
    latency_rf = benchmark.benchmark_inference_latency(model_rf, X_test)
    eval_rf["training_time_seconds"] = train_time_rf
    eval_rf["inference_latency_ms"] = latency_rf

    benchmark.save_confusion_matrix_plot(
        eval_rf["confusion_matrix"], class_labels, "tfidf_rf"
    )
    benchmark.save_classification_report(eval_rf["classification_report"], "tfidf_rf")

    results["models"]["tfidf_rf"] = eval_rf

    # Model 3: TF-IDF + LinearSVC
    logger.info("Training TF-IDF + LinearSVC...")
    start = time.time()
    model_svc = benchmark.build_tfidf_svc_pipeline()
    model_svc.fit(X_train, y_train)
    train_time_svc = time.time() - start

    eval_svc = benchmark.evaluate_model(model_svc, X_test, y_test, "TF-IDF + LinearSVC")
    latency_svc = benchmark.benchmark_inference_latency(model_svc, X_test)
    eval_svc["training_time_seconds"] = train_time_svc
    eval_svc["inference_latency_ms"] = latency_svc

    benchmark.save_confusion_matrix_plot(
        eval_svc["confusion_matrix"], class_labels, "tfidf_svc"
    )
    benchmark.save_classification_report(eval_svc["classification_report"], "tfidf_svc")

    results["models"]["tfidf_svc"] = eval_svc

    logger.info("Baseline benchmarking complete")
    return results


def generate_benchmark_report() -> None:
    """Run all baselines, save results, and print summary table."""
    logger.info("Generating benchmark report...")

    # Run baselines
    results = run_sklearn_baseline()

    # Save results to JSON
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "benchmark_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved benchmark results to {results_file}")

    # Print summary table
    print("\n" + "=" * 90)
    print("SKLEARN BASELINE BENCHMARK SUMMARY")
    print("=" * 90)
    print(f"\nDataset: {results['dataset']}")
    print(f"Train samples: {results['split']['train']}, Test samples: {results['split']['test']}\n")

    print(f"{'Model':<30} {'Accuracy':<12} {'Macro F1':<12} {'Weighted F1':<12} {'Train Time (s)':<15}")
    print("-" * 90)

    for model_name, model_results in results["models"].items():
        print(
            f"{model_name:<30} {model_results['accuracy']:<12.4f} "
            f"{model_results['macro_f1']:<12.4f} {model_results['weighted_f1']:<12.4f} "
            f"{model_results['training_time_seconds']:<15.2f}"
        )

    print("\nInference Latency (milliseconds):")
    print("-" * 90)
    print(f"{'Model':<30} {'Mean':<12} {'Median':<12} {'Min':<12} {'Max':<12}")
    print("-" * 90)

    for model_name, model_results in results["models"].items():
        latency = model_results["inference_latency_ms"]
        print(
            f"{model_name:<30} {latency['mean_ms']:<12.4f} {latency['median_ms']:<12.4f} "
            f"{latency['min_ms']:<12.4f} {latency['max_ms']:<12.4f}"
        )

    print("\n" + "=" * 90)
    print("Reports saved to: outputs/")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Generate report
    generate_benchmark_report()
