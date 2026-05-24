"""Train multilingual Hybrid and LingRF+PredOut on MULTITuDE mAA."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import types
from datetime import datetime
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from sklearn.metrics import f1_score

SCRIPTS_DIR: Path = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _bootstrap import configure_project_root

PROJECT_ROOT: str = str(configure_project_root(__file__, remove_shadowing_utils=False))

for package_name in ("feature_extraction", "models", "utils"):
    package_dir: Path = Path(PROJECT_ROOT) / package_name
    package_module: types.ModuleType = types.ModuleType(package_name)
    package_module.__path__ = [str(package_dir)]  # type: ignore[attr-defined]
    package_module.__package__ = package_name
    sys.modules[package_name] = package_module

from feature_extraction.linguistic_features import LingRFPredOutClassifier
from feature_extraction.probabilistic_features import fixed_len
from scripts.multitude.config import (
    HYBRID_VARIANT,
    LANGUAGE_COLUMN,
    LINGRF_VARIANT,
    MULTITUDE_CHECKPOINT_DIR,
    ROW_ID_COLUMN,
    SOURCE_COLUMN,
    SPLIT_COLUMN,
    FloatArray,
    IntArray,
    MultitudeSplits,
    ProbabilitySplits,
    VariantResult,
    parse_args,
    parse_prob_models,
    set_seeds,
)
from scripts.multitude.data import (
    load_multitude_splits,
    load_or_compute_probability_features,
    print_startup_validation,
)
from scripts.multitude.features import (
    extract_hybrid_probabilities,
    extract_multilingual_style_features,
    get_tokenizer,
    train_predout_lstm,
)
from scripts.multitude.models import MultitudeHybridBiLSTMEncoder
from utils.constants import DEVICE, LOCAL_DEVICE, LOG_DIR, RESULTS_DIR
from utils.logging_utils import Tee
from utils.training_pipeline import create_dataloaders, tokenize_splits, train_and_evaluate


def id_to_label(label_to_id: dict[str, int]) -> dict[int, str]:
    return {idx: label for label, idx in label_to_id.items()}


def safe_label_name(label: str) -> str:
    safe: str = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_").lower()
    if not safe:
        raise ValueError(f"Could not create safe probability column name for label: {label}")
    return safe


def save_test_predictions(
    output_path: Path,
    test_frame: pd.DataFrame,
    test_y: IntArray,
    probabilities: FloatArray,
    label_to_id: dict[str, int],
) -> None:
    labels_by_id: dict[int, str] = id_to_label(label_to_id=label_to_id)
    pred_ids: IntArray = probabilities.argmax(axis=1).astype(int)
    confidences: FloatArray = probabilities.max(axis=1).astype(np.float32)

    output_frame: pd.DataFrame = pd.DataFrame(
        {
            ROW_ID_COLUMN: test_frame[ROW_ID_COLUMN].to_numpy(),
            SPLIT_COLUMN: test_frame[SPLIT_COLUMN].to_numpy(),
            LANGUAGE_COLUMN: test_frame[LANGUAGE_COLUMN].to_numpy(),
            SOURCE_COLUMN: test_frame[SOURCE_COLUMN].to_numpy(),
            "true_label_id": test_y.astype(int),
            "pred_label_id": pred_ids,
            "true_label": [labels_by_id[int(label_id)] for label_id in test_y],
            "pred_label": [labels_by_id[int(label_id)] for label_id in pred_ids],
            "pred_confidence": confidences,
            "correct": pred_ids == test_y,
        }
    )

    for class_id in sorted(labels_by_id):
        label: str = labels_by_id[class_id]
        output_frame[f"prob_{class_id}_{safe_label_name(label=label)}"] = probabilities[:, class_id]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_frame.to_csv(output_path, sep="\t", index=False)


def run_hybrid(
    splits: MultitudeSplits,
    prob_features: ProbabilitySplits,
    seed: int,
    epochs: int,
    freeze_epochs: int,
    batch_size: int,
    encoder_id: str,
    timestamp: str,
) -> VariantResult:
    print("\n" + "=" * 80)
    print(f"Training {HYBRID_VARIANT}")
    print("=" * 80)
    set_seeds(seed=seed)

    tokenizer = get_tokenizer(encoder_id=encoder_id)
    train_enc, dev_enc, test_enc = tokenize_splits(
        splits.train_texts,
        splits.dev_texts,
        splits.test_texts,
        tokenizer,
        fixed_len,
    )

    train_loader, dev_loader, test_loader = create_dataloaders(
        train_X=prob_features.train,
        dev_X=prob_features.dev,
        test_X=prob_features.test,
        train_enc=train_enc,
        dev_enc=dev_enc,
        test_enc=test_enc,
        train_Y=splits.train_y,
        dev_Y=splits.dev_y,
        test_Y=splits.test_y,
        batch_size=batch_size,
        num_workers=0,
    )

    model: MultitudeHybridBiLSTMEncoder = MultitudeHybridBiLSTMEncoder(
        seq_feature_len=prob_features.train.shape[2],
        num_classes=len(splits.label_to_id),
        local_device=LOCAL_DEVICE,
        encoder_id=encoder_id,
        lstm_hidden_size=64,
        lstm_bidirectional=True,
    ).to(DEVICE)

    checkpoint_prefix: str = f"multitude_{HYBRID_VARIANT}_seed{seed}"
    result = train_and_evaluate(
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=test_loader,
        model=model,
        model_variant="pred_flm",
        device=DEVICE,
        out_dir=MULTITUDE_CHECKPOINT_DIR,
        checkpoint_prefix=checkpoint_prefix,
        epochs=epochs,
        freeze_epochs=freeze_epochs,
        cleanup_non_best=True,
    )

    test_probabilities: FloatArray = extract_hybrid_probabilities(
        model=model,
        seq_features=prob_features.test,
        encodings=test_enc,
        labels=splits.test_y,
        batch_size=batch_size,
        device=DEVICE,
    )
    prediction_path: Path = RESULTS_DIR / f"multitude_{HYBRID_VARIANT}_{timestamp}_test_predictions.tsv"
    save_test_predictions(
        output_path=prediction_path,
        test_frame=splits.test_frame,
        test_y=splits.test_y,
        probabilities=test_probabilities,
        label_to_id=splits.label_to_id,
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return VariantResult(
        variant=HYBRID_VARIANT,
        dev_f1=result.dev_f1,
        test_f1=result.test_f1,
        best_epoch=result.best_epoch,
        prediction_path=prediction_path,
        extra={"encoder_id": encoder_id},
    )


def complete_probability_matrix(
    probabilities: FloatArray,
    classes: NDArray[np.int_],
    num_classes: int,
) -> FloatArray:
    complete: FloatArray = np.zeros((probabilities.shape[0], num_classes), dtype=np.float32)
    for column_idx, class_id in enumerate(classes.astype(int)):
        complete[:, int(class_id)] = probabilities[:, column_idx]
    return complete


def macro_f1(y_true: IntArray, y_pred: IntArray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def run_lingrf_predout(
    splits: MultitudeSplits,
    prob_features: ProbabilitySplits,
    seed: int,
    batch_size: int,
    lstm_epochs: int,
    rf_estimators: int,
    rf_max_depth: int,
    timestamp: str,
) -> VariantResult:
    print("\n" + "=" * 80)
    print(f"Training {LINGRF_VARIANT}")
    print("=" * 80)
    set_seeds(seed=seed)

    train_pred_probs, dev_pred_probs, test_pred_probs, predout_epoch, predout_dev_f1, predout_test_f1 = (
        train_predout_lstm(
            splits=splits,
            prob_features=prob_features,
            seed=seed,
            batch_size=batch_size,
            lstm_epochs=lstm_epochs,
            num_classes=len(splits.label_to_id),
        )
    )
    print(
        f"[PREDOUT] LSTM epoch={predout_epoch} | dev={predout_dev_f1:.4f} | "
        f"test={predout_test_f1:.4f}"
    )

    print("\n[STYLE] Extracting existing 26 stylometric features")
    train_style, style_feature_names = extract_multilingual_style_features(
        texts=splits.train_texts,
        languages=splits.train_frame[LANGUAGE_COLUMN].tolist(),
    )
    dev_style, _ = extract_multilingual_style_features(
        texts=splits.dev_texts,
        languages=splits.dev_frame[LANGUAGE_COLUMN].tolist(),
    )
    test_style, _ = extract_multilingual_style_features(
        texts=splits.test_texts,
        languages=splits.test_frame[LANGUAGE_COLUMN].tolist(),
    )

    clf: LingRFPredOutClassifier = LingRFPredOutClassifier(
        n_estimators=rf_estimators,
        max_depth=rf_max_depth,
        random_state=seed,
    )
    clf.fit(
        ling_features=train_style,
        pred_probs=train_pred_probs,
        y=splits.train_y,
        feature_names=style_feature_names,
    )

    train_preds: IntArray = clf.predict(train_style, train_pred_probs).astype(int)
    dev_preds: IntArray = clf.predict(dev_style, dev_pred_probs).astype(int)
    test_preds: IntArray = clf.predict(test_style, test_pred_probs).astype(int)

    train_f1: float = macro_f1(y_true=splits.train_y, y_pred=train_preds)
    dev_f1: float = macro_f1(y_true=splits.dev_y, y_pred=dev_preds)
    test_f1: float = macro_f1(y_true=splits.test_y, y_pred=test_preds)

    print("\n[RESULTS]")
    print(f"  Train F1: {train_f1:.4f}")
    print(f"  Dev F1:   {dev_f1:.4f}")
    print(f"  Test F1:  {test_f1:.4f}")

    top_features: list[tuple[str, float]] = clf.get_feature_importance(top_k=20)
    print("\n[FEATURE IMPORTANCE] Top 20")
    for rank, (name, importance) in enumerate(top_features, 1):
        print(f"  {rank:2d}. {name}: {importance:.6f}")

    raw_test_probabilities: FloatArray = clf.predict_proba(test_style, test_pred_probs).astype(np.float32)
    test_probabilities: FloatArray = complete_probability_matrix(
        probabilities=raw_test_probabilities,
        classes=clf.model.classes_,
        num_classes=len(splits.label_to_id),
    )

    prediction_path: Path = RESULTS_DIR / f"multitude_{LINGRF_VARIANT}_{timestamp}_test_predictions.tsv"
    save_test_predictions(
        output_path=prediction_path,
        test_frame=splits.test_frame,
        test_y=splits.test_y,
        probabilities=test_probabilities,
        label_to_id=splits.label_to_id,
    )

    return VariantResult(
        variant=LINGRF_VARIANT,
        dev_f1=dev_f1,
        test_f1=test_f1,
        best_epoch=predout_epoch,
        prediction_path=prediction_path,
        extra={
            "predout_dev_f1": f"{predout_dev_f1:.6f}",
            "predout_test_f1": f"{predout_test_f1:.6f}",
            "rf_train_f1": f"{train_f1:.6f}",
            "n_style_features": str(len(style_feature_names)),
            "n_predout_features": str(train_pred_probs.shape[1]),
        },
    )


def save_label_mapping(label_to_id: dict[str, int], timestamp: str) -> None:
    mapping_payload: dict[str, object] = {
        "label_to_id": label_to_id,
        "id_to_label": {str(idx): label for label, idx in label_to_id.items()},
    }
    timestamped_path: Path = RESULTS_DIR / f"multitude_label_mapping_{timestamp}.json"
    latest_path: Path = RESULTS_DIR / "multitude_label_mapping_latest.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for path in (timestamped_path, latest_path):
        path.write_text(json.dumps(mapping_payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_results(
    results: list[VariantResult],
    splits: MultitudeSplits,
    model_ids: list[str],
    seed: int,
    timestamp: str,
) -> None:
    rows: list[dict[str, object]] = []
    label_mapping: str = json.dumps(splits.label_to_id, ensure_ascii=False, sort_keys=True)
    prob_models: str = ",".join(model_ids)

    for result in results:
        row: dict[str, object] = {
            "variant": result.variant,
            "seed": seed,
            "n_train": len(splits.train_y),
            "n_dev": len(splits.dev_y),
            "n_test": len(splits.test_y),
            "n_classes": len(splits.label_to_id),
            "best_epoch": result.best_epoch,
            "dev_f1": f"{result.dev_f1:.6f}",
            "test_f1": f"{result.test_f1:.6f}",
            "prob_models": prob_models,
            "label_mapping": label_mapping,
            "prediction_path": str(result.prediction_path),
        }
        row.update(result.extra)
        rows.append(row)

    output_frame: pd.DataFrame = pd.DataFrame(rows)
    timestamped_path: Path = RESULTS_DIR / f"multitude_results_{timestamp}.tsv"
    latest_path: Path = RESULTS_DIR / "multitude_results_latest.tsv"
    output_frame.to_csv(timestamped_path, sep="\t", index=False)
    output_frame.to_csv(latest_path, sep="\t", index=False)
    print(f"\n[INFO] Saved results: {timestamped_path}")
    print(f"[INFO] Saved results: {latest_path}")


def selected_variants(variant_arg: str) -> list[str]:
    if variant_arg == "all":
        return [HYBRID_VARIANT, LINGRF_VARIANT]
    return [variant_arg]


def run() -> None:
    args: argparse.Namespace = parse_args()
    timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MULTITUDE_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    log_path: Path = LOG_DIR / f"multitude_{timestamp}.log"
    tee: Tee = Tee(log_path)
    sys.stdout = tee

    try:
        print("=" * 80)
        print("MULTITuDE mAA multilingual training")
        print("=" * 80)
        print(f"Dataset: {args.dataset}")
        print(f"Variant: {args.variant}")
        print(f"Seed: {args.seed}")
        print(f"Device: {DEVICE}")
        print(f"Log: {log_path}")
        print("=" * 80)

        set_seeds(seed=int(args.seed))
        model_ids: list[str] = parse_prob_models(raw_models=str(args.prob_models))
        splits: MultitudeSplits = load_multitude_splits(
            dataset_path=Path(args.dataset),
            seed=int(args.seed),
        )
        print_startup_validation(splits=splits)

        if bool(args.validate_only):
            print("\n[VALIDATE] Startup validation completed. No training was run.")
            return

        save_label_mapping(label_to_id=splits.label_to_id, timestamp=timestamp)

        variants: list[str] = selected_variants(variant_arg=str(args.variant))
        prob_features: ProbabilitySplits = load_or_compute_probability_features(
            dataset_path=Path(args.dataset),
            splits=splits,
            model_ids=model_ids,
            seed=int(args.seed),
            prob_batch_size=int(args.prob_batch_size),
            force_recompute=bool(args.force_recompute_prob),
        )

        results: list[VariantResult] = []
        if HYBRID_VARIANT in variants:
            results.append(
                run_hybrid(
                    splits=splits,
                    prob_features=prob_features,
                    seed=int(args.seed),
                    epochs=int(args.epochs),
                    freeze_epochs=int(args.freeze_epochs),
                    batch_size=int(args.batch_size),
                    encoder_id=str(args.encoder_id),
                    timestamp=timestamp,
                )
            )

        if LINGRF_VARIANT in variants:
            results.append(
                run_lingrf_predout(
                    splits=splits,
                    prob_features=prob_features,
                    seed=int(args.seed),
                    batch_size=int(args.batch_size),
                    lstm_epochs=int(args.lstm_epochs),
                    rf_estimators=int(args.rf_estimators),
                    rf_max_depth=int(args.rf_max_depth),
                    timestamp=timestamp,
                )
            )

        save_results(
            results=results,
            splits=splits,
            model_ids=model_ids,
            seed=int(args.seed),
            timestamp=timestamp,
        )

        print("\n" + "=" * 80)
        print("Training complete")
        print("=" * 80)
    finally:
        tee.close()


if __name__ == "__main__":
    run()
