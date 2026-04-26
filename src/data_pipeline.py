"""Data pipeline optimization for financial text processing.

Implements streaming data loading, intelligent caching, data quality
validation, and preprocessing optimizations for downstream ML tasks.
"""

import logging
import os
import pickle
import threading
import time
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psutil

logger = logging.getLogger(__name__)


class DataQualityValidator:
    """Validates data quality and generates comprehensive quality reports.

    Checks for:
    - Empty or null text
    - Duplicate sentences (using fuzzy matching)
    - Class imbalance (alert if ratio > 10:1)
    - Text length distribution anomalies
    """

    def __init__(self, similarity_threshold: float = 0.95):
        """Initialize the validator.

        Args:
            similarity_threshold: SequenceMatcher similarity threshold for duplicates.
        """
        self.similarity_threshold = similarity_threshold
        self.report: Dict[str, Any] = {}

    def check_empty_or_null(self, texts: List[str]) -> Tuple[int, List[int]]:
        """Check for empty or null text entries.

        Args:
            texts: List of text strings to validate.

        Returns:
            Tuple of (count of empty/null items, indices of problematic items).
        """
        empty_indices = [i for i, text in enumerate(texts) if not text or text.strip() == ""]
        return len(empty_indices), empty_indices

    def detect_duplicates(self, texts: List[str]) -> Tuple[int, Dict[int, List[int]]]:
        """Detect duplicate or near-duplicate sentences using fuzzy matching.

        Args:
            texts: List of text strings to check.

        Returns:
            Tuple of (number of duplicates found, dict mapping indices to duplicate indices).
        """
        duplicate_groups: Dict[int, List[int]] = defaultdict(list)
        seen_indices = set()

        for i in range(len(texts)):
            if i in seen_indices:
                continue

            for j in range(i + 1, len(texts)):
                if j in seen_indices:
                    continue

                ratio = SequenceMatcher(None, texts[i], texts[j]).ratio()
                if ratio >= self.similarity_threshold:
                    duplicate_groups[i].append(j)
                    seen_indices.add(j)

        total_duplicates = sum(len(v) for v in duplicate_groups.values())
        return total_duplicates, dict(duplicate_groups)

    def validate_label_distribution(
        self, labels: List[str], imbalance_threshold: float = 10.0
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check label distribution and alert on severe class imbalance.

        Args:
            labels: List of labels.
            imbalance_threshold: Maximum acceptable ratio between most and least common class.

        Returns:
            Tuple of (is_balanced, distribution_info_dict).
        """
        label_counts = Counter(labels)
        counts = list(label_counts.values())

        if not counts:
            return True, {"label_counts": {}, "imbalance_ratio": 0.0, "is_balanced": True}

        max_count = max(counts)
        min_count = min(counts)
        imbalance_ratio = max_count / min_count if min_count > 0 else float("inf")

        is_balanced = imbalance_ratio <= imbalance_threshold

        distribution_info = {
            "label_counts": dict(label_counts),
            "imbalance_ratio": imbalance_ratio,
            "is_balanced": is_balanced,
            "max_class": max(label_counts, key=label_counts.get),
            "min_class": min(label_counts, key=label_counts.get),
        }

        if not is_balanced:
            logger.warning(
                f"Class imbalance detected: ratio={imbalance_ratio:.2f}x "
                f"(threshold={imbalance_threshold}x)"
            )

        return is_balanced, distribution_info

    def check_text_length_distribution(
        self, texts: List[str], percentile_threshold: float = 0.95
    ) -> Dict[str, Any]:
        """Analyze text length distribution and flag outliers.

        Args:
            texts: List of text strings.
            percentile_threshold: Percentile above which to flag as outlier.

        Returns:
            Dict with length statistics and outlier indices.
        """
        lengths = [len(text) for text in texts]

        if not lengths:
            return {
                "mean_length": 0,
                "min_length": 0,
                "max_length": 0,
                "median_length": 0,
                "outlier_indices": [],
            }

        sorted_lengths = sorted(lengths)
        n = len(sorted_lengths)
        percentile_idx = int(n * percentile_threshold)
        percentile_value = sorted_lengths[percentile_idx]

        outlier_indices = [i for i, length in enumerate(lengths) if length > percentile_value]

        return {
            "mean_length": sum(lengths) / len(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "median_length": sorted_lengths[n // 2],
            f"{percentile_threshold*100:.0f}th_percentile": percentile_value,
            "outlier_count": len(outlier_indices),
            "outlier_indices": outlier_indices,
        }

    def generate_report(
        self, texts: List[str], labels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive data quality report.

        Args:
            texts: List of text strings.
            labels: Optional list of labels for classification tasks.

        Returns:
            Comprehensive quality report as dictionary.
        """
        report = {}

        # Empty/null check
        empty_count, empty_indices = self.check_empty_or_null(texts)
        report["empty_or_null"] = {
            "count": empty_count,
            "percentage": (empty_count / len(texts) * 100) if texts else 0,
            "sample_indices": empty_indices[:10],
        }

        # Duplicate check
        dup_count, dup_groups = self.detect_duplicates(texts)
        report["duplicates"] = {
            "count": dup_count,
            "percentage": (dup_count / len(texts) * 100) if texts else 0,
            "sample_groups": dict(list(dup_groups.items())[:5]),
        }

        # Text length distribution
        report["length_distribution"] = self.check_text_length_distribution(texts)

        # Label distribution
        if labels:
            is_balanced, label_dist = self.validate_label_distribution(labels)
            report["label_distribution"] = label_dist
            report["is_balanced"] = is_balanced
        else:
            report["label_distribution"] = None
            report["is_balanced"] = None

        report["total_records"] = len(texts)
        self.report = report

        return report


class DataPipelineOptimizer:
    """Optimizes data pipeline with cleaning, deduplication, and augmentation.

    Handles:
    - Text normalization and cleaning
    - Deduplication with configurable thresholds
    - Stratified sampling
    - Data augmentation stubs
    - Disk caching with pickle/parquet
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the optimizer.

        Args:
            cache_dir: Directory to cache preprocessed data. If None, disables caching.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.stats = {
            "records_processed": 0,
            "records_filtered": 0,
            "start_time": None,
            "end_time": None,
        }

    def clean_text(self, text: str) -> str:
        """Intelligently clean text.

        Handles:
        - Whitespace normalization
        - Control character removal
        - Encoding issue fixes
        - Case preservation

        Args:
            text: Raw text to clean.

        Returns:
            Cleaned text.
        """
        if not isinstance(text, str):
            return ""

        # Remove control characters
        text = "".join(char for char in text if ord(char) >= 32 or char in "\n\t\r")

        # Normalize whitespace
        text = " ".join(text.split())

        # Fix common encoding issues
        text = text.replace("â", "'")  # Smart quote
        text = text.replace("â", '"')  # Left double quote
        text = text.replace("â", '"')  # Right double quote

        return text.strip()

    def deduplicate_texts(
        self, texts: List[str], similarity_threshold: float = 0.95
    ) -> Tuple[List[str], List[int]]:
        """Remove duplicate texts using fuzzy matching.

        Args:
            texts: List of texts to deduplicate.
            similarity_threshold: Similarity threshold for considering items duplicates.

        Returns:
            Tuple of (deduplicated texts, indices of kept items from original).
        """
        deduplicated = []
        kept_indices = []

        for i, text in enumerate(texts):
            is_duplicate = False
            for existing in deduplicated:
                if SequenceMatcher(None, text, existing).ratio() >= similarity_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                deduplicated.append(text)
                kept_indices.append(i)

        return deduplicated, kept_indices

    def stratified_sample(
        self, texts: List[str], labels: List[str], sample_fraction: float = 0.8
    ) -> Tuple[List[str], List[str], List[int]]:
        """Perform stratified sampling for balanced training sets.

        Args:
            texts: List of texts.
            labels: Corresponding labels.
            sample_fraction: Fraction of data to sample (0 < fraction <= 1).

        Returns:
            Tuple of (sampled_texts, sampled_labels, sampled_indices).
        """
        from collections import defaultdict

        # Group by label
        label_groups: Dict[str, List[int]] = defaultdict(list)
        for i, label in enumerate(labels):
            label_groups[label].append(i)

        sampled_indices = []
        for label, indices in label_groups.items():
            sample_size = max(1, int(len(indices) * sample_fraction))
            sampled_indices.extend(sorted(indices)[: int(sample_size)])

        sampled_indices = sorted(sampled_indices)
        sampled_texts = [texts[i] for i in sampled_indices]
        sampled_labels = [labels[i] for i in sampled_indices]

        return sampled_texts, sampled_labels, sampled_indices

    def augment_with_synonym_replacement(self, text: str, num_replacements: int = 2) -> str:
        """Stub for synonym replacement augmentation.

        Args:
            text: Input text.
            num_replacements: Number of words to replace (placeholder).

        Returns:
            Augmented text (currently returns original).
        """
        logger.debug(f"Synonym replacement augmentation stub (would replace {num_replacements} words)")
        # Placeholder: actual implementation would use NLTK, spaCy, or similar
        return text

    def augment_with_back_translation(self, text: str, intermediate_lang: str = "fr") -> str:
        """Stub for back-translation augmentation.

        Args:
            text: Input text.
            intermediate_lang: Intermediate language code (placeholder).

        Returns:
            Augmented text (currently returns original).
        """
        logger.debug(f"Back-translation augmentation stub (intermediate_lang={intermediate_lang})")
        # Placeholder: actual implementation would use HuggingFace transformers or Google Translate API
        return text

    def cache_to_disk(self, data: Any, filename: str, format: str = "pickle") -> str:
        """Cache preprocessed data to disk.

        Args:
            data: Data to cache (dict or list).
            filename: Filename for cached data (without extension).
            format: Format to use ('pickle' or 'parquet').

        Returns:
            Path to cached file.

        Raises:
            ValueError: If format is not supported or cache_dir not set.
        """
        if not self.cache_dir:
            raise ValueError("cache_dir not set during initialization")

        if format == "pickle":
            filepath = self.cache_dir / f"{filename}.pkl"
            with open(filepath, "wb") as f:
                pickle.dump(data, f)
        elif format == "parquet":
            try:
                import pandas as pd

                filepath = self.cache_dir / f"{filename}.parquet"
                df = pd.DataFrame(data)
                df.to_parquet(filepath)
            except ImportError:
                logger.warning("pandas not installed, falling back to pickle")
                filepath = self.cache_dir / f"{filename}.pkl"
                with open(filepath, "wb") as f:
                    pickle.dump(data, f)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Cached data to {filepath}")
        return str(filepath)

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline execution statistics.

        Returns:
            Dict with records processed, filtered, and elapsed time.
        """
        stats = self.stats.copy()
        if self.stats["start_time"] and self.stats["end_time"]:
            stats["elapsed_seconds"] = self.stats["end_time"] - self.stats["start_time"]
        return stats

    def process_pipeline(
        self,
        texts: List[str],
        labels: Optional[List[str]] = None,
        clean: bool = True,
        deduplicate: bool = True,
        sample: bool = False,
        sample_fraction: float = 0.8,
        cache: bool = False,
        cache_filename: str = "processed_data",
    ) -> Tuple[List[str], Optional[List[str]]]:
        """Execute full preprocessing pipeline.

        Args:
            texts: Input texts.
            labels: Optional labels.
            clean: Whether to clean text.
            deduplicate: Whether to deduplicate.
            sample: Whether to use stratified sampling.
            sample_fraction: Fraction to sample if sample=True.
            cache: Whether to cache results.
            cache_filename: Filename for cache.

        Returns:
            Tuple of (processed_texts, processed_labels or None).
        """
        self.stats["start_time"] = time.time()
        self.stats["records_processed"] = len(texts)

        processed_texts = texts
        processed_labels = labels

        # Clean
        if clean:
            processed_texts = [self.clean_text(t) for t in processed_texts]

        # Deduplicate
        if deduplicate:
            processed_texts, kept_indices = self.deduplicate_texts(processed_texts)
            if processed_labels:
                processed_labels = [processed_labels[i] for i in kept_indices]

        # Sample
        if sample and processed_labels:
            processed_texts, processed_labels, _ = self.stratified_sample(
                processed_texts, processed_labels, sample_fraction
            )

        self.stats["records_filtered"] = len(texts) - len(processed_texts)
        self.stats["end_time"] = time.time()

        # Cache
        if cache:
            cache_data = {
                "texts": processed_texts,
                "labels": processed_labels,
                "stats": self.get_stats(),
            }
            self.cache_to_disk(cache_data, cache_filename, format="pickle")

        return processed_texts, processed_labels


class StreamingDataLoader:
    """Memory-efficient batch iterator with prefetching."""

    def __init__(
        self,
        texts: List[str],
        labels: Optional[List[str]] = None,
        batch_size: Optional[int] = None,
        auto_batch_size: bool = True,
    ):
        """Initialize streaming data loader.

        Args:
            texts: List of texts.
            labels: Optional list of labels.
            batch_size: Batch size (auto-detected if None and auto_batch_size=True).
            auto_batch_size: Whether to auto-detect optimal batch size.
        """
        self.texts = texts
        self.labels = labels
        self.auto_batch_size = auto_batch_size

        if batch_size is not None:
            self.batch_size = batch_size
        elif auto_batch_size:
            self.batch_size = self._calculate_optimal_batch_size()
        else:
            self.batch_size = 32

        self._prefetch_queue: Optional[Tuple] = None
        self._stop_prefetch = False

    def _calculate_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on available memory.

        Returns:
            Recommended batch size.
        """
        try:
            available_memory = psutil.virtual_memory().available
            # Assume ~1KB per text on average, use 10% of available memory
            estimated_text_size = (
                sum(len(t.encode("utf-8")) for t in self.texts) / len(self.texts)
            )
            recommended_batch = max(1, int(available_memory * 0.1 / estimated_text_size))
            logger.info(
                f"Auto-detected batch size: {recommended_batch} "
                f"(available memory: {available_memory / 1e9:.1f}GB)"
            )
            return recommended_batch
        except Exception as e:
            logger.warning(f"Failed to auto-detect batch size: {e}, using default=32")
            return 32

    def _prefetch_worker(self, batch_indices: List[int]) -> None:
        """Background thread worker for prefetching.

        Args:
            batch_indices: Indices of the next batch to fetch.
        """
        batch_texts = [self.texts[i] for i in batch_indices]
        batch_labels = (
            [self.labels[i] for i in batch_indices] if self.labels else None
        )
        self._prefetch_queue = (batch_texts, batch_labels)

    def __iter__(self):
        """Iterate over batches with prefetching."""
        n = len(self.texts)
        prefetch_thread = None

        for batch_start in range(0, n, self.batch_size):
            batch_end = min(batch_start + self.batch_size, n)
            batch_indices = list(range(batch_start, batch_end))

            # Wait for prefetch from previous iteration
            if prefetch_thread:
                prefetch_thread.join()
                if self._prefetch_queue:
                    batch_texts, batch_labels = self._prefetch_queue
                    self._prefetch_queue = None
                else:
                    batch_texts = [self.texts[i] for i in batch_indices]
                    batch_labels = (
                        [self.labels[i] for i in batch_indices] if self.labels else None
                    )
            else:
                batch_texts = [self.texts[i] for i in batch_indices]
                batch_labels = (
                    [self.labels[i] for i in batch_indices] if self.labels else None
                )

            # Prefetch next batch in background
            next_start = batch_end
            if next_start < n:
                next_end = min(next_start + self.batch_size, n)
                next_indices = list(range(next_start, next_end))
                prefetch_thread = threading.Thread(
                    target=self._prefetch_worker, args=(next_indices,)
                )
                prefetch_thread.daemon = True
                prefetch_thread.start()

            yield (batch_texts, batch_labels)

        # Clean up prefetch thread
        if prefetch_thread:
            prefetch_thread.join()

    def __len__(self) -> int:
        """Return number of batches."""
        return (len(self.texts) + self.batch_size - 1) // self.batch_size
