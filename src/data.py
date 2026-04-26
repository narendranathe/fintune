"""Dataset loading, preprocessing, and tokenization for financial text NLP."""

from __future__ import annotations

import logging
from typing import Any

from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

logger = logging.getLogger(__name__)

LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
NUM_LABELS = len(LABEL_MAP)


def load_financial_phrasebank(
    split_seed: int = 42,
    test_size: float = 0.2,
    agreement_level: str = "sentences_allagree",
) -> DatasetDict:
    """Load financial_phrasebank from HF Hub and create train/test split.

    Args:
        split_seed: Random seed for reproducible splits.
        test_size: Fraction of data reserved for evaluation.
        agreement_level: Annotator agreement level filter.

    Returns:
        DatasetDict with 'train' and 'test' splits.
    """
    dataset = load_dataset(
        "takala/financial_phrasebank",
        agreement_level,
    )

    # Dataset ships as a single 'train' split — we create our own test set
    splits = dataset["train"].train_test_split(
        test_size=test_size,
        seed=split_seed,
        stratify_by_column="label",
    )

    logger.info(
        "Loaded financial_phrasebank: train=%d, test=%d",
        len(splits["train"]),
        len(splits["test"]),
    )
    return splits


def build_tokenizer(model_name: str, max_length: int = 512) -> PreTrainedTokenizer:
    """Load and configure tokenizer with padding strategy.

    Args:
        model_name: HF model identifier.
        max_length: Maximum sequence length for truncation.

    Returns:
        Configured tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.model_max_length = max_length
    return tokenizer


def tokenize_dataset(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
    text_column: str = "sentence",
) -> DatasetDict:
    """Tokenize dataset with padding and truncation.

    Args:
        dataset: Raw dataset with text and labels.
        tokenizer: Pretrained tokenizer.
        max_length: Max token length.
        text_column: Name of the text field.

    Returns:
        Tokenized DatasetDict ready for Trainer.
    """

    def _tokenize(examples: dict[str, Any]) -> dict[str, Any]:
        tokens = tokenizer(
            examples[text_column],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        tokens["labels"] = examples["label"]
        return tokens

    tokenized = dataset.map(
        _tokenize,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing",
    )

    tokenized.set_format("torch")
    logger.info("Tokenization complete. Max length=%d", max_length)
    return tokenized
