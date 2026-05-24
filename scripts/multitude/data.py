"""MULTITuDE dataset loading and probability feature caching."""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from feature_extraction.probabilistic_features import ConfigurableProbFeatures, fixed_len
from scripts.multitude.config import (
    EXPECTED_NUM_CLASSES,
    LABEL_COLUMN,
    LANGUAGE_COLUMN,
    MULTITUDE_FEATURE_DIR,
    ProbabilitySplits,
    ROW_ID_COLUMN,
    SOURCE_COLUMN,
    SPLIT_COLUMN,
    TARGET_COLUMN,
    TEXT_COLUMN,
    VALIDATION_SIZE,
    FloatArray,
    IntArray,
    MultitudeSplits,
)
from utils.constants import DEVICE, LOCAL_DEVICE


def dataset_fingerprint(dataset_path: Path) -> dict[str, object]:
    resolved_path: Path = dataset_path.resolve()
    stat_result: os.stat_result = resolved_path.stat()
    return {
        "path": str(resolved_path),
        "size_bytes": int(stat_result.st_size),
        "mtime_ns": int(stat_result.st_mtime_ns),
    }


def class_counts(labels: IntArray) -> dict[str, int]:
    counts: Counter[int] = Counter(int(label) for label in labels)
    return {str(label_id): int(count) for label_id, count in sorted(counts.items())}


def validate_columns(frame: pd.DataFrame, dataset_path: Path) -> None:
    required_columns: set[str] = {
        TEXT_COLUMN,
        LABEL_COLUMN,
        SPLIT_COLUMN,
        LANGUAGE_COLUMN,
        SOURCE_COLUMN,
    }
    missing_columns: set[str] = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            f"Dataset {dataset_path} is missing required columns: {sorted(missing_columns)}"
        )


def load_multitude_splits(dataset_path: Path, seed: int) -> MultitudeSplits:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Multitude dataset not found: {dataset_path}")

    frame: pd.DataFrame = pd.read_csv(dataset_path)
    validate_columns(frame=frame, dataset_path=dataset_path)

    frame = frame.copy()
    frame[ROW_ID_COLUMN] = np.arange(len(frame), dtype=np.int64)
    frame[TEXT_COLUMN] = frame[TEXT_COLUMN].fillna("").astype(str)
    frame[LABEL_COLUMN] = frame[LABEL_COLUMN].astype(str)
    frame[SPLIT_COLUMN] = frame[SPLIT_COLUMN].astype(str)
    frame[LANGUAGE_COLUMN] = frame[LANGUAGE_COLUMN].astype(str)
    frame[SOURCE_COLUMN] = frame[SOURCE_COLUMN].fillna("").astype(str)

    split_values: set[str] = set(frame[SPLIT_COLUMN].unique().tolist())
    expected_splits: set[str] = {"train", "test"}
    if split_values != expected_splits:
        raise ValueError(f"Expected split values {expected_splits}, got {split_values}.")

    labels: list[str] = sorted(frame[LABEL_COLUMN].unique().tolist())
    if len(labels) != EXPECTED_NUM_CLASSES:
        raise ValueError(f"Expected {EXPECTED_NUM_CLASSES} labels, got {len(labels)}: {labels}")

    label_to_id: dict[str, int] = {label: idx for idx, label in enumerate(labels)}
    frame[TARGET_COLUMN] = frame[LABEL_COLUMN].map(label_to_id)
    if frame[TARGET_COLUMN].isna().any():
        bad_labels: list[str] = sorted(frame.loc[frame[TARGET_COLUMN].isna(), LABEL_COLUMN].unique())
        raise ValueError(f"Could not map labels: {bad_labels}")

    train_pool: pd.DataFrame = frame.loc[frame[SPLIT_COLUMN] == "train"].copy()
    test_frame: pd.DataFrame = frame.loc[frame[SPLIT_COLUMN] == "test"].copy()
    train_frame, dev_frame = train_test_split(
        train_pool,
        test_size=VALIDATION_SIZE,
        stratify=train_pool[TARGET_COLUMN],
        random_state=seed,
    )

    train_frame = train_frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    dev_frame = dev_frame.reset_index(drop=True)
    test_frame = test_frame.reset_index(drop=True)

    train_y: IntArray = train_frame[TARGET_COLUMN].astype(int).to_numpy()
    dev_y: IntArray = dev_frame[TARGET_COLUMN].astype(int).to_numpy()
    test_y: IntArray = test_frame[TARGET_COLUMN].astype(int).to_numpy()

    validate_split_coverage(train_y=train_y, dev_y=dev_y, test_y=test_y, num_classes=len(labels))

    return MultitudeSplits(
        train_frame=train_frame,
        dev_frame=dev_frame,
        test_frame=test_frame,
        train_texts=train_frame[TEXT_COLUMN].tolist(),
        dev_texts=dev_frame[TEXT_COLUMN].tolist(),
        test_texts=test_frame[TEXT_COLUMN].tolist(),
        train_y=train_y,
        dev_y=dev_y,
        test_y=test_y,
        label_to_id=label_to_id,
    )


def validate_split_coverage(
    train_y: IntArray,
    dev_y: IntArray,
    test_y: IntArray,
    num_classes: int,
) -> None:
    expected_ids: set[int] = set(range(num_classes))
    observed_by_split: dict[str, set[int]] = {
        "train": set(int(label) for label in train_y),
        "dev": set(int(label) for label in dev_y),
        "test": set(int(label) for label in test_y),
    }
    for split_name, observed_ids in observed_by_split.items():
        missing_ids: set[int] = expected_ids.difference(observed_ids)
        if missing_ids:
            raise ValueError(f"Split {split_name} is missing class ids: {sorted(missing_ids)}")


def print_startup_validation(splits: MultitudeSplits) -> None:
    print("\n[DATA] MULTITuDE mAA validation")
    print(f"  Train samples: {len(splits.train_y)}")
    print(f"  Dev samples:   {len(splits.dev_y)}")
    print(f"  Test samples:  {len(splits.test_y)}")
    print(f"  Languages:     {splits.train_frame[LANGUAGE_COLUMN].nunique()} train languages")
    print(f"  Labels:        {json.dumps(splits.label_to_id, ensure_ascii=False, sort_keys=True)}")
    print(f"  Train classes: {class_counts(splits.train_y)}")
    print(f"  Dev classes:   {class_counts(splits.dev_y)}")
    print(f"  Test classes:  {class_counts(splits.test_y)}")


def probability_cache_path(seed: int, model_ids: list[str]) -> Path:
    model_slug: str = re.sub(r"[^A-Za-z0-9]+", "_", "__".join(model_ids)).strip("_").lower()
    compact_slug: str = model_slug[:120]
    return MULTITUDE_FEATURE_DIR / f"prob_features_seed{seed}_{compact_slug}.npz"


def build_probability_metadata(
    dataset_path: Path,
    splits: MultitudeSplits,
    model_ids: list[str],
    seed: int,
    prob_batch_size: int,
) -> dict[str, object]:
    return {
        "feature_type": "multitude_multilingual_prob_features",
        "dataset": dataset_fingerprint(dataset_path=dataset_path),
        "seed": int(seed),
        "validation_size": float(VALIDATION_SIZE),
        "fixed_len": int(fixed_len),
        "prob_batch_size": int(prob_batch_size),
        "prob_models": list(model_ids),
        "label_to_id": dict(splits.label_to_id),
        "split_counts": {
            "train": int(len(splits.train_y)),
            "dev": int(len(splits.dev_y)),
            "test": int(len(splits.test_y)),
        },
        "class_counts": {
            "train": class_counts(splits.train_y),
            "dev": class_counts(splits.dev_y),
            "test": class_counts(splits.test_y),
        },
    }


def metadata_json(metadata: dict[str, object]) -> str:
    return json.dumps(metadata, ensure_ascii=False, sort_keys=True)


def load_or_compute_probability_features(
    dataset_path: Path,
    splits: MultitudeSplits,
    model_ids: list[str],
    seed: int,
    prob_batch_size: int,
    force_recompute: bool,
) -> ProbabilitySplits:
    MULTITUDE_FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path: Path = probability_cache_path(seed=seed, model_ids=model_ids)
    expected_metadata: dict[str, object] = build_probability_metadata(
        dataset_path=dataset_path,
        splits=splits,
        model_ids=model_ids,
        seed=seed,
        prob_batch_size=prob_batch_size,
    )
    expected_json: str = metadata_json(expected_metadata)

    if cache_path.exists() and not force_recompute:
        print(f"\n[FEATURES] Loading cached multilingual probability features: {cache_path}")
        with np.load(cache_path, allow_pickle=False) as cached:
            cached_json: str = str(cached["metadata_json"].item())
            if cached_json != expected_json:
                raise ValueError(
                    "Probability feature cache metadata mismatch. "
                    f"Cache path: {cache_path}. "
                    "Run again with --force-recompute-prob to rebuild it."
                )
            return ProbabilitySplits(
                train=cached["train"].astype(np.float32),
                dev=cached["dev"].astype(np.float32),
                test=cached["test"].astype(np.float32),
                metadata=json.loads(cached_json),
            )

    print("\n[FEATURES] Computing multilingual probabilistic token features")
    print(f"  Models: {model_ids}")
    extractor: ConfigurableProbFeatures = ConfigurableProbFeatures(
        device=DEVICE,
        local_device=LOCAL_DEVICE,
        model_ids=model_ids,
        disabled=False,
        batch_size=prob_batch_size,
    )

    train_prob: FloatArray = np.asarray(extractor.word_features(splits.train_texts), dtype=np.float32)
    dev_prob: FloatArray = np.asarray(extractor.word_features(splits.dev_texts), dtype=np.float32)
    test_prob: FloatArray = np.asarray(extractor.word_features(splits.test_texts), dtype=np.float32)

    print(f"[FEATURES] Saving probability cache: {cache_path}")
    np.savez_compressed(
        cache_path,
        train=train_prob,
        dev=dev_prob,
        test=test_prob,
        metadata_json=np.array(expected_json),
    )

    return ProbabilitySplits(
        train=train_prob,
        dev=dev_prob,
        test=test_prob,
        metadata=expected_metadata,
    )
