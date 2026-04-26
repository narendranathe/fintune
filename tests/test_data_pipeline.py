"""
Tests for data pipeline including validation, cleaning, sampling, and stats tracking.
"""

import unittest
import re
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class PipelineStats:
    """Statistics tracking for data pipeline."""
    total_samples: int = 0
    samples_after_validation: int = 0
    samples_after_dedup: int = 0
    samples_after_cleaning: int = 0
    samples_after_sampling: int = 0
    duplicates_removed: int = 0
    validation_failures: int = 0


class DataQualityValidator:
    """Validator for data quality checks."""

    @staticmethod
    def validate(text: str) -> bool:
        """Validate that text is not empty and meets basic criteria."""
        if not text or not isinstance(text, str):
            return False
        if text.strip() == "":
            return False
        return True

    @staticmethod
    def validate_batch(texts: List[str]) -> Tuple[List[str], int]:
        """Validate a batch of texts, return valid texts and failure count."""
        valid_texts = []
        failures = 0

        for text in texts:
            if DataQualityValidator.validate(text):
                valid_texts.append(text)
            else:
                failures += 1

        return valid_texts, failures


class DuplicateDetector:
    """Detect and remove duplicate texts."""

    @staticmethod
    def remove_duplicates(texts: List[str]) -> Tuple[List[str], int]:
        """Remove duplicates from text list."""
        seen = set()
        unique_texts = []
        duplicates = 0

        for text in texts:
            normalized = text.strip().lower()
            if normalized not in seen:
                seen.add(normalized)
                unique_texts.append(text)
            else:
                duplicates += 1

        return unique_texts, duplicates


class TextCleaner:
    """Clean and normalize text data."""

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Normalize whitespace: multiple spaces to single space, trim."""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    @staticmethod
    def remove_control_characters(text: str) -> str:
        """Remove control characters (except newline, tab)."""
        # Keep only printable characters and common whitespace
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
        return text

    @staticmethod
    def clean(text: str) -> str:
        """Apply all cleaning operations."""
        text = TextCleaner.remove_control_characters(text)
        text = TextCleaner.normalize_whitespace(text)
        return text

    @staticmethod
    def clean_batch(texts: List[str]) -> List[str]:
        """Clean a batch of texts."""
        return [TextCleaner.clean(text) for text in texts]


class StratifiedSampler:
    """Perform stratified sampling while preserving label ratios."""

    @staticmethod
    def stratified_sample(
        texts: List[str],
        labels: List[str],
        sample_size: int
    ) -> Tuple[List[str], List[str]]:
        """
        Perform stratified sampling to preserve label distribution.
        """
        if sample_size >= len(texts):
            return texts, labels

        # Group by label
        label_groups = {}
        for text, label in zip(texts, labels):
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(text)

        # Calculate samples per label based on ratio
        total = len(texts)
        sampled_texts = []
        sampled_labels = []

        for label, group_texts in label_groups.items():
            ratio = len(group_texts) / total
            samples_for_label = int(sample_size * ratio)
            samples_for_label = max(1, samples_for_label)  # At least 1

            # Random selection (simplified: take first N)
            selected = group_texts[:samples_for_label]
            sampled_texts.extend(selected)
            sampled_labels.extend([label] * len(selected))

        return sampled_texts, sampled_labels

    @staticmethod
    def verify_ratio_preservation(
        original_labels: List[str],
        sampled_labels: List[str],
        tolerance: float = 0.1
    ) -> bool:
        """Verify that label ratios are preserved within tolerance."""
        original_ratios = {}
        for label in original_labels:
            original_ratios[label] = original_labels.count(label) / len(original_labels)

        sampled_ratios = {}
        for label in sampled_labels:
            sampled_ratios[label] = sampled_labels.count(label) / len(sampled_labels)

        # Check that all labels are present
        if set(original_ratios.keys()) != set(sampled_ratios.keys()):
            return False

        # Check ratios within tolerance
        for label in original_ratios:
            diff = abs(original_ratios[label] - sampled_ratios[label])
            if diff > tolerance:
                return False

        return True


class DataPipeline:
    """Complete data pipeline with validation, cleaning, and sampling."""

    def __init__(self):
        self.stats = PipelineStats()

    def process(
        self,
        texts: List[str],
        labels: List[str] = None,
        sample_size: int = None
    ) -> Tuple[List[str], List[str], PipelineStats]:
        """
        Process data through complete pipeline:
        1. Validation
        2. Deduplication
        3. Cleaning
        4. Sampling (optional)
        """
        self.stats = PipelineStats()
        self.stats.total_samples = len(texts)

        # Validation
        texts, failures = DataQualityValidator.validate_batch(texts)
        self.stats.validation_failures = failures
        self.stats.samples_after_validation = len(texts)

        if labels:
            labels = [l for t, l in zip(
                [t for t in range(len(texts))],
                labels
            ) if t < len(texts)]

        # Deduplication
        texts, duplicates = DuplicateDetector.remove_duplicates(texts)
        self.stats.duplicates_removed = duplicates
        self.stats.samples_after_dedup = len(texts)

        # Cleaning
        texts = TextCleaner.clean_batch(texts)
        self.stats.samples_after_cleaning = len(texts)

        # Sampling
        if sample_size and sample_size < len(texts):
            if labels:
                texts, labels = StratifiedSampler.stratified_sample(
                    texts, labels, sample_size
                )
            else:
                texts = texts[:sample_size]
            self.stats.samples_after_sampling = len(texts)
        else:
            self.stats.samples_after_sampling = len(texts)

        return texts, labels, self.stats


class TestDataQualityValidator(unittest.TestCase):
    """Test DataQualityValidator."""

    def test_validate_valid_text(self):
        """Test validation of valid text."""
        text = "This is a valid sentence."
        self.assertTrue(DataQualityValidator.validate(text))

    def test_validate_empty_string(self):
        """Test that empty string fails validation."""
        self.assertFalse(DataQualityValidator.validate(""))

    def test_validate_whitespace_only(self):
        """Test that whitespace-only string fails validation."""
        self.assertFalse(DataQualityValidator.validate("   "))
        self.assertFalse(DataQualityValidator.validate("\t\n"))

    def test_validate_none(self):
        """Test that None fails validation."""
        self.assertFalse(DataQualityValidator.validate(None))

    def test_validate_batch(self):
        """Test batch validation."""
        texts = [
            "Valid text",
            "",
            "Another valid text",
            "   ",
            "Third valid text"
        ]

        valid, failures = DataQualityValidator.validate_batch(texts)

        self.assertEqual(len(valid), 3)
        self.assertEqual(failures, 2)


class TestDuplicateDetection(unittest.TestCase):
    """Test duplicate detection and removal."""

    def test_remove_exact_duplicates(self):
        """Test removal of exact duplicates."""
        texts = ["Hello", "World", "Hello", "Python"]

        unique, dup_count = DuplicateDetector.remove_duplicates(texts)

        self.assertEqual(len(unique), 3)
        self.assertEqual(dup_count, 1)
        self.assertIn("Hello", unique)
        self.assertIn("World", unique)
        self.assertIn("Python", unique)

    def test_remove_case_insensitive_duplicates(self):
        """Test removal of case-insensitive duplicates."""
        texts = ["Hello", "HELLO", "hello"]

        unique, dup_count = DuplicateDetector.remove_duplicates(texts)

        self.assertEqual(len(unique), 1)
        self.assertEqual(dup_count, 2)

    def test_no_duplicates(self):
        """Test with no duplicates."""
        texts = ["First", "Second", "Third"]

        unique, dup_count = DuplicateDetector.remove_duplicates(texts)

        self.assertEqual(len(unique), 3)
        self.assertEqual(dup_count, 0)


class TestTextCleaning(unittest.TestCase):
    """Test text cleaning operations."""

    def test_normalize_whitespace(self):
        """Test whitespace normalization."""
        text = "Hello    world  \t\n  python"
        cleaned = TextCleaner.normalize_whitespace(text)

        self.assertEqual(cleaned, "Hello world python")

    def test_normalize_leading_trailing_whitespace(self):
        """Test trim leading/trailing whitespace."""
        text = "   hello world   "
        cleaned = TextCleaner.normalize_whitespace(text)

        self.assertEqual(cleaned, "hello world")

    def test_remove_control_characters(self):
        """Test removal of control characters."""
        # Create text with control character
        text = "Hello\x00World"
        cleaned = TextCleaner.remove_control_characters(text)

        # Control character should be removed
        self.assertNotIn('\x00', cleaned)
        self.assertIn("Hello", cleaned)
        self.assertIn("World", cleaned)

    def test_preserve_newlines_and_tabs(self):
        """Test that newlines and tabs are preserved."""
        text = "Hello\nWorld\tPython"
        cleaned = TextCleaner.remove_control_characters(text)

        self.assertIn("\n", cleaned)
        self.assertIn("\t", cleaned)

    def test_full_cleaning_pipeline(self):
        """Test complete cleaning pipeline."""
        text = "  Hello\x00  \t\n  World  "
        cleaned = TextCleaner.clean(text)

        self.assertEqual(cleaned, "Hello World")

    def test_clean_batch(self):
        """Test batch cleaning."""
        texts = ["  Hello  ", "  World  ", "  Python  "]
        cleaned = TextCleaner.clean_batch(texts)

        self.assertEqual(cleaned, ["Hello", "World", "Python"])


class TestStratifiedSampling(unittest.TestCase):
    """Test stratified sampling."""

    def test_stratified_sample_preserves_ratios(self):
        """Test that stratified sampling preserves label ratios."""
        texts = ["text_" + str(i) for i in range(100)]
        labels = ["A"] * 70 + ["B"] * 30

        sampled_texts, sampled_labels = StratifiedSampler.stratified_sample(
            texts, labels, sample_size=20
        )

        # Should have 14 A's and 6 B's (approximately)
        a_count = sampled_labels.count("A")
        b_count = sampled_labels.count("B")

        # Verify ratio preservation with tolerance
        is_preserved = StratifiedSampler.verify_ratio_preservation(
            labels, sampled_labels, tolerance=0.15
        )
        self.assertTrue(is_preserved)

    def test_sample_size_larger_than_data(self):
        """Test sampling when sample size exceeds data size."""
        texts = ["text_1", "text_2", "text_3"]
        labels = ["A", "B", "A"]

        sampled_texts, sampled_labels = StratifiedSampler.stratified_sample(
            texts, labels, sample_size=10
        )

        self.assertEqual(len(sampled_texts), 3)
        self.assertEqual(len(sampled_labels), 3)

    def test_sample_single_label(self):
        """Test stratified sampling with single label."""
        texts = ["text_" + str(i) for i in range(10)]
        labels = ["A"] * 10

        sampled_texts, sampled_labels = StratifiedSampler.stratified_sample(
            texts, labels, sample_size=5
        )

        self.assertEqual(len(sampled_texts), 5)
        self.assertTrue(all(l == "A" for l in sampled_labels))


class TestPipelineStats(unittest.TestCase):
    """Test pipeline statistics tracking."""

    def test_stats_initialization(self):
        """Test PipelineStats initialization."""
        stats = PipelineStats()

        self.assertEqual(stats.total_samples, 0)
        self.assertEqual(stats.samples_after_validation, 0)
        self.assertEqual(stats.duplicates_removed, 0)

    def test_stats_tracking_full_pipeline(self):
        """Test stats tracking through full pipeline."""
        texts = [
            "Valid text 1",
            "",
            "Valid text 2",
            "Valid text 1",  # duplicate
            "Valid text 3"
        ]

        pipeline = DataPipeline()
        result_texts, _, stats = pipeline.process(texts)

        self.assertEqual(stats.total_samples, 5)
        self.assertEqual(stats.validation_failures, 1)  # empty string
        self.assertEqual(stats.duplicates_removed, 1)    # one duplicate
        self.assertEqual(stats.samples_after_cleaning, 4)

    def test_stats_with_sampling(self):
        """Test stats tracking with sampling."""
        texts = ["text_" + str(i) for i in range(20)]
        labels = ["A"] * 10 + ["B"] * 10

        pipeline = DataPipeline()
        result_texts, result_labels, stats = pipeline.process(
            texts, labels, sample_size=10
        )

        self.assertEqual(stats.total_samples, 20)
        self.assertEqual(stats.samples_after_sampling, 10)


if __name__ == '__main__':
    unittest.main()
