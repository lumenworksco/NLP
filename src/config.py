"""Centralized configuration for the multilingual sentiment analysis project."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"

# ── Training hyperparameters ───────────────────────────────────────────────
SEED = 42
MAX_LENGTH = 256
NUM_EPOCHS = 3
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
LR_SCHEDULER = "linear"
MAX_GRAD_NORM = 1.0
LOGGING_STEPS = 100
DATALOADER_WORKERS = 2

# ── Model identifiers ─────────────────────────────────────────────────────
MODELS = {
    "en": "bert-base-uncased",
    "nl": "GroNLP/bert-base-dutch-cased",
    "fr": "almanach/camembert-base",
    "multilingual": "bert-base-multilingual-cased",
}

# ── Dataset sources (Hugging Face) ────────────────────────────────────────
DATASETS = {
    "en": {"name": "stanfordnlp/imdb", "text_col": "text"},
    "fr": {"name": "tblard/allocine", "text_col": "review"},
    "nl": {"name": "benjaminvdb/dbrd", "text_col": "text"},
}

# ── Languages ──────────────────────────────────────────────────────────────
LANGUAGES = ["en", "fr", "nl"]
LABEL_NAMES = ["Negative", "Positive"]
