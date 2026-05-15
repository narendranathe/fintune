"""Tests for data loading and tokenization."""

from src.data import LABEL_MAP, ID_TO_LABEL, NUM_LABELS


def test_label_map_consistency():
    """Label map and reverse map should be bijective."""
    assert len(LABEL_MAP) == NUM_LABELS
    assert len(ID_TO_LABEL) == NUM_LABELS
    for label, idx in LABEL_MAP.items():
        assert ID_TO_LABEL[idx] == label


def test_label_map_values():
    """All expected labels should be present."""
    assert set(LABEL_MAP.keys()) == {"negative", "neutral", "positive"}
    assert set(LABEL_MAP.values()) == {0, 1, 2}


def test_num_labels():
    """Financial phrasebank has exactly 3 sentiment classes."""
    assert NUM_LABELS == 3
