"""Shared constants, encoder maps, label maps, and directory layout."""

from __future__ import annotations

from pathlib import Path

import torch

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DEVICE: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LOCAL_DEVICE: torch.device = torch.device("cpu")

DEFAULT_SEED: int = 10
EPOCHS: int = 20
FREEZE_EPOCHS: int = 5
BATCH_SIZE: int = 16

ENCODER_MAP_BASELINE: dict[str, str] = {
    "en": "roberta-base",
    "es": "bertin-project/bertin-roberta-base-spanish",
}
ENCODER_MULTILINGUAL: str = "microsoft/mdeberta-v3-base"

BASELINE_ENCODERS: frozenset[str] = frozenset({
    "roberta-base",
    "bertin-project/bertin-roberta-base-spanish",
})

SUBTASK_1_LABELS: dict[int, str] = {0: "human", 1: "generated"}
SUBTASK_2_LABELS: dict[int, str] = {i: chr(ord("A") + i) for i in range(6)}

MODEL_VARIANTS: tuple[str, ...] = ("pred", "flm", "pred_flm", "pred_flm_add")

DATA_DIR: Path = PROJECT_ROOT / "data" / "data"
FEATURES_DIR: Path = PROJECT_ROOT / "data" / "features"
OUT_DIR: Path = PROJECT_ROOT / "data" / "out"
CHECKPOINTS_DIR: Path = OUT_DIR / "checkpoints"
RESULTS_DIR: Path = OUT_DIR / "results"
SHAP_DIR: Path = OUT_DIR / "shap"
LOG_DIR: Path = PROJECT_ROOT / "logs"


def is_baseline_encoder(encoder_id: str) -> bool:
    return encoder_id in BASELINE_ENCODERS
