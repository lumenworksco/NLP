"""Tests for project configuration."""

from src import config


def test_paths_are_absolute():
    assert config.PROJECT_ROOT.is_absolute()
    assert config.DATA_DIR.is_absolute()
    assert config.MODEL_DIR.is_absolute()


def test_data_dir_under_project_root():
    assert str(config.DATA_DIR).startswith(str(config.PROJECT_ROOT))


def test_languages_defined():
    assert len(config.LANGUAGES) == 3
    assert set(config.LANGUAGES) == {"en", "fr", "nl"}


def test_models_cover_all_languages():
    for lang in config.LANGUAGES:
        assert lang in config.MODELS, f"Missing model for {lang}"
    assert "multilingual" in config.MODELS


def test_datasets_cover_all_languages():
    for lang in config.LANGUAGES:
        assert lang in config.DATASETS
        assert "name" in config.DATASETS[lang]
        assert "text_col" in config.DATASETS[lang]


def test_label_names():
    assert config.LABEL_NAMES == ["Negative", "Positive"]


def test_hyperparameters_reasonable():
    assert 0 < config.LEARNING_RATE < 1
    assert config.MAX_LENGTH > 0
    assert config.NUM_EPOCHS > 0
    assert config.TRAIN_BATCH_SIZE > 0
    assert config.SEED == 42
