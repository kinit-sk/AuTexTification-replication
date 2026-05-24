"""Configuration and shared types for MULTITuDE training."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray

from feature_extraction.probabilistic_features import MULTILINGUAL_LARGE_MODELS
from feature_extraction.style_features import get_style_feature_names
from utils.constants import (
    BATCH_SIZE,
    CHECKPOINTS_DIR,
    ENCODER_MULTILINGUAL,
    EPOCHS,
    FEATURES_DIR,
    FREEZE_EPOCHS,
)

PROJECT_ROOT: str = str(Path(__file__).resolve().parents[2])

RANDOM_SEED: int = 42
EXPECTED_NUM_CLASSES: int = 8
VALIDATION_SIZE: float = 0.05

DATASET_PATH: Path = Path(PROJECT_ROOT) / "data" / "multitude_v3_mAA.csv.gz"
MULTITUDE_FEATURE_DIR: Path = FEATURES_DIR / "multitude"
MULTITUDE_CHECKPOINT_DIR: Path = CHECKPOINTS_DIR / "multitude"

TEXT_COLUMN: str = "text"
LABEL_COLUMN: str = "multi_label"
SPLIT_COLUMN: str = "split"
LANGUAGE_COLUMN: str = "language"
SOURCE_COLUMN: str = "source"
ROW_ID_COLUMN: str = "row_id"
TARGET_COLUMN: str = "label_id"

HYBRID_VARIANT: str = "hybrid_multilingual"
LINGRF_VARIANT: str = "lingrf_predout_multilingual"

STYLE_FEATURE_NAMES: tuple[str, ...] = tuple(get_style_feature_names())

FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int_]


@dataclass(frozen=True)
class MultitudeSplits:
    train_frame: pd.DataFrame
    dev_frame: pd.DataFrame
    test_frame: pd.DataFrame
    train_texts: list[str]
    dev_texts: list[str]
    test_texts: list[str]
    train_y: IntArray
    dev_y: IntArray
    test_y: IntArray
    label_to_id: dict[str, int]


@dataclass(frozen=True)
class ProbabilitySplits:
    train: FloatArray
    dev: FloatArray
    test: FloatArray
    metadata: dict[str, object]


@dataclass(frozen=True)
class VariantResult:
    variant: str
    dev_f1: float
    test_f1: float
    best_epoch: int
    prediction_path: Path
    extra: dict[str, str]


def parse_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Train Multitude multilingual Hybrid and LingRF+PredOut models."
    )
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument(
        "--variant",
        choices=["all", HYBRID_VARIANT, LINGRF_VARIANT],
        default="all",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--freeze-epochs", type=int, default=FREEZE_EPOCHS)
    parser.add_argument("--lstm-epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--prob-batch-size", type=int, default=8)
    parser.add_argument("--rf-estimators", type=int, default=200)
    parser.add_argument("--rf-max-depth", type=int, default=60)
    parser.add_argument("--encoder-id", type=str, default=ENCODER_MULTILINGUAL)
    parser.add_argument(
        "--prob-models",
        type=str,
        default=",".join(MULTILINGUAL_LARGE_MODELS),
        help="Comma-separated causal LM ids for multilingual probabilistic features.",
    )
    parser.add_argument("--force-recompute-prob", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_prob_models(raw_models: str) -> list[str]:
    model_ids: list[str] = [model.strip() for model in raw_models.split(",") if model.strip()]
    if not model_ids:
        raise ValueError("--prob-models must contain at least one model id.")
    return model_ids
