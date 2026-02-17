"""Shared training and evaluation utilities."""

import numpy as np
import torch
import evaluate
from transformers import AutoTokenizer, TrainingArguments

from . import config


def get_device() -> str:
    """Return the best available device string."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ── Metrics ────────────────────────────────────────────────────────────────

_accuracy = evaluate.load("accuracy")
_f1 = evaluate.load("f1")
_precision = evaluate.load("precision")
_recall = evaluate.load("recall")


def compute_metrics(eval_pred):
    """Compute accuracy, F1, precision, and recall from Trainer eval predictions."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        **_accuracy.compute(predictions=preds, references=labels),
        **_f1.compute(predictions=preds, references=labels),
        **_precision.compute(predictions=preds, references=labels),
        **_recall.compute(predictions=preds, references=labels),
    }


# ── Tokenization ──────────────────────────────────────────────────────────


def tokenize_dataset(dataset, tokenizer, max_length=config.MAX_LENGTH):
    """Tokenize a HuggingFace dataset, removing text/language columns."""

    def _tok(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    cols_to_remove = [c for c in dataset.column_names if c not in ("label",)]
    tokenized = dataset.map(_tok, batched=True, remove_columns=cols_to_remove)
    tokenized.set_format("torch")
    return tokenized


# ── Training arguments ────────────────────────────────────────────────────


def get_training_args(output_dir: str, num_epochs: int = config.NUM_EPOCHS) -> TrainingArguments:
    """Return a TrainingArguments object with the project defaults."""
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        warmup_ratio=config.WARMUP_RATIO,
        lr_scheduler_type=config.LR_SCHEDULER,
        max_grad_norm=config.MAX_GRAD_NORM,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=config.LOGGING_STEPS,
        report_to="none",
        seed=config.SEED,
        dataloader_num_workers=config.DATALOADER_WORKERS,
    )
