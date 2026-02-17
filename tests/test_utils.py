"""Tests for shared training utilities."""

import numpy as np
import pytest

from src import config
from src.utils import compute_metrics, get_device, get_training_args


def test_compute_metrics_perfect():
    """Perfect predictions should yield 1.0 across all metrics."""
    logits = np.array([[0.1, 0.9], [0.9, 0.1], [0.1, 0.9]])
    labels = np.array([1, 0, 1])
    result = compute_metrics((logits, labels))

    assert result["accuracy"] == 1.0
    assert result["f1"] == 1.0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0


def test_compute_metrics_worst():
    """All-wrong predictions should yield 0.0 F1."""
    logits = np.array([[0.9, 0.1], [0.1, 0.9], [0.9, 0.1]])
    labels = np.array([1, 0, 1])
    result = compute_metrics((logits, labels))

    assert result["accuracy"] == 0.0
    assert result["f1"] == 0.0


def test_compute_metrics_keys():
    logits = np.array([[0.1, 0.9], [0.9, 0.1]])
    labels = np.array([1, 0])
    result = compute_metrics((logits, labels))
    assert set(result.keys()) == {"accuracy", "f1", "precision", "recall"}


def test_get_device_returns_string():
    device = get_device()
    assert device in ("cuda", "mps", "cpu")


def test_get_training_args_defaults():
    args = get_training_args("/tmp/test_output")
    assert args.num_train_epochs == config.NUM_EPOCHS
    assert args.learning_rate == config.LEARNING_RATE
    assert args.per_device_train_batch_size == config.TRAIN_BATCH_SIZE
    assert args.seed == config.SEED
    assert args.metric_for_best_model == "f1"


def test_get_training_args_custom_epochs():
    args = get_training_args("/tmp/test_output", num_epochs=5)
    assert args.num_train_epochs == 5
