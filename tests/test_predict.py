"""Tests for the SentimentPredictor inference class."""

import pytest

from src.predict import SentimentPredictor
from src import config


@pytest.fixture
def model_path():
    """Return path to a trained model, or skip if none available."""
    # Try mBERT multilingual first, then any available model
    candidates = [
        config.MODEL_DIR / "mbert-multilingual",
        config.MODEL_DIR / "bert-en",
        config.MODEL_DIR / "camembert-fr",
        config.MODEL_DIR / "bertje-nl",
    ]
    for path in candidates:
        if path.exists() and (path / "config.json").exists():
            return str(path)
    pytest.skip("No trained model available — run training notebooks first")


def test_predict_returns_dict(model_path):
    predictor = SentimentPredictor(model_path)
    result = predictor.predict("This is a great movie!")
    assert isinstance(result, dict)
    assert "label" in result
    assert "score" in result


def test_predict_label_valid(model_path):
    predictor = SentimentPredictor(model_path)
    result = predictor.predict("Terrible film, waste of time.")
    assert result["label"] in config.LABEL_NAMES


def test_predict_score_range(model_path):
    predictor = SentimentPredictor(model_path)
    result = predictor.predict("An okay movie.")
    assert 0.0 <= result["score"] <= 1.0


def test_predict_batch(model_path):
    predictor = SentimentPredictor(model_path)
    texts = ["Great movie!", "Horrible experience.", "It was fine."]
    results = predictor.predict_batch(texts)
    assert len(results) == 3
    for r in results:
        assert r["label"] in config.LABEL_NAMES
        assert 0.0 <= r["score"] <= 1.0
