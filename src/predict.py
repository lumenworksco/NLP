"""Inference utilities for trained sentiment models."""

from __future__ import annotations

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from . import config


class SentimentPredictor:
    """Load a fine-tuned model and classify text."""

    def __init__(self, model_path: str, max_length: int = config.MAX_LENGTH):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.max_length = max_length

        # Move to best available device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.model.to(self.device)

    def predict(self, text: str) -> dict:
        """Classify a single text string.

        Returns:
            dict with keys 'label' ('Positive'/'Negative') and 'score' (float).
        """
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = torch.softmax(logits, dim=-1).squeeze()
        pred_id = probs.argmax().item()

        return {
            "label": config.LABEL_NAMES[pred_id],
            "score": probs[pred_id].item(),
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """Classify a list of texts.

        Returns:
            List of dicts, each with 'label' and 'score'.
        """
        inputs = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = torch.softmax(logits, dim=-1)
        pred_ids = probs.argmax(dim=-1)

        results = []
        for i, pid in enumerate(pred_ids):
            results.append({
                "label": config.LABEL_NAMES[pid.item()],
                "score": probs[i, pid].item(),
            })
        return results
