"""Features-only: aggregate token-level features + linguistic (+ style) → RF / XGBoost / MLP."""

from __future__ import annotations

import argparse
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from feature_extraction.linguistic_features import LinguisticFeatures
from feature_extraction.style_features import StyleFeatures
from utils.constants import (
    DATA_DIR,
    DEVICE,
    FEATURES_DIR,
    LOCAL_DEVICE,
    LOG_DIR,
    OUT_DIR,
)
from utils.classifier_utils import run_classifiers
from utils.data_utils import load_train_dev_test
from utils.feature_utils import compute_all_features
from utils.logging_utils import Tee

RESOURCES_DIR: Path = Path(PROJECT_ROOT) / "resources"
FEATURES_OUT_DIR: Path = OUT_DIR / "tuning"

STATS: tuple[str, ...] = ("mean", "max", "min", "std")
STAT_FN: dict[str, callable] = {
    "mean": lambda x: np.mean(x, axis=0),
    "max": lambda x: np.max(x, axis=0),
    "min": lambda x: np.min(x, axis=0),
    "std": lambda x: np.std(x, axis=0),
}


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(0)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def aggregate_token_features(
    X: np.ndarray,
    mask_channel: int = 0,
) -> tuple[np.ndarray, list[str]]:
    """Pool 3D token features (samples, seq_len, channels) to 2D via mean/max/min/std.

    Channel 0 is a binary mask indicating valid tokens; remaining channels
    are the actual features.
    """
    n_samples, _, n_channels = X.shape
    n_feat = n_channels - 1

    blocks: list[np.ndarray] = []
    names: list[str] = []

    for stat in STATS:
        fn = STAT_FN[stat]
        buf = np.zeros((n_samples, n_feat), dtype=np.float32)

        for i in range(n_samples):
            valid_len = int(X[i, :, mask_channel].sum())
            if valid_len > 0:
                buf[i] = fn(X[i, :valid_len, 1:])

        blocks.append(buf)
        names.extend(f"prob_{j}_{stat}" for j in range(n_feat))

    return np.concatenate(blocks, axis=1), names


def main() -> None:
    parser = argparse.ArgumentParser(description="Features-only: aggregated token features + linguistic → RF / XGB / MLP")
    parser.add_argument("--subtask", choices=["subtask_1", "subtask_2"], default="subtask_1")
    parser.add_argument("--lang", choices=["en", "es"], default="en")
    parser.add_argument("--config", choices=["baseline", "multilingual"], default="baseline")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--style", action="store_true", help="Include style features")
    args = parser.parse_args()

    subtask, lang, seed = args.subtask, args.lang, args.seed
    multilingual = args.config == "multilingual"
    n_classes = 2 if subtask == "subtask_1" else 6

    for d in (FEATURES_OUT_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tee = Tee(LOG_DIR / f"features_{subtask}_{lang}_{args.config}_{timestamp}.log")
    sys.stdout = tee

    print("=" * 80)
    print(f"Features-only | {subtask} | {lang} | {args.config} | seed={seed} | style={args.style}")
    print("=" * 80)

    set_seeds(seed)

    train_texts, dev_texts, test_texts, train_Y, dev_Y, test_Y, train_idx, dev_idx = (
        load_train_dev_test(
            train_dir=DATA_DIR / "train" / subtask / lang,
            test_dir=DATA_DIR / "test" / subtask / lang,
            subtask=subtask, seed=seed,
        )
    )

    # ── Token-level features → aggregate ─────────────────────────────────
    print("\n[STEP] Computing token-level features (prob + freq + grammar)...")
    train_tok, dev_tok, test_tok, _ = compute_all_features(
        train_texts=train_texts, dev_texts=dev_texts, test_texts=test_texts,
        train_idx=train_idx, dev_idx=dev_idx,
        subtask=subtask, lang=lang,
        device=DEVICE, local_device=LOCAL_DEVICE,
        model_variant="pred_flm_add", features_dir=FEATURES_DIR,
        multilingual=multilingual,
    )
    print(f"  Token-level shape: {train_tok.shape}")

    print("\n[STEP] Aggregating to document-level (mean/max/min/std)...")
    train_agg, agg_names = aggregate_token_features(train_tok)
    dev_agg, _ = aggregate_token_features(dev_tok)
    test_agg, _ = aggregate_token_features(test_tok)
    print(f"  Aggregated shape: {train_agg.shape}")

    # ── Linguistic features (+ optional style) ───────────────────────────
    print("\n[STEP] Extracting linguistic features...")
    ling_ext = LinguisticFeatures(language=lang, resources_dir=RESOURCES_DIR)
    train_ling, feat_names = ling_ext.extract_features(train_texts)
    dev_ling, _ = ling_ext.extract_features(dev_texts, feature_names=feat_names)
    test_ling, _ = ling_ext.extract_features(test_texts, feature_names=feat_names)

    if args.style:
        print("[STEP] Extracting style features...")
        style_ext = StyleFeatures(language=lang)
        tr_s, s_names = style_ext.extract(train_texts, cache_key=f"train_{subtask}_{lang}")
        dv_s, _ = style_ext.extract(dev_texts, cache_key=f"dev_{subtask}_{lang}")
        te_s, _ = style_ext.extract(test_texts, cache_key=f"test_{subtask}_{lang}")
        train_ling = np.concatenate([train_ling, tr_s], axis=1)
        dev_ling = np.concatenate([dev_ling, dv_s], axis=1)
        test_ling = np.concatenate([test_ling, te_s], axis=1)
        feat_names = feat_names + s_names

    # ── Combine ──────────────────────────────────────────────────────────
    all_names = feat_names + agg_names
    train_X = np.concatenate([train_ling, train_agg], axis=1)
    dev_X = np.concatenate([dev_ling, dev_agg], axis=1)
    test_X = np.concatenate([test_ling, test_agg], axis=1)
    print(f"\n  Combined: {train_X.shape[1]} features ({len(feat_names)} ling + {len(agg_names)} agg)")

    npz_path = FEATURES_OUT_DIR / f"{subtask}_{lang}_{args.config}_features_stage2_data.npz"
    np.savez(
        npz_path,
        train_X=train_X, dev_X=dev_X, test_X=test_X,
        train_y=train_Y, dev_y=dev_Y, test_y=test_Y,
        subtask=subtask, lang=lang, n_classes=n_classes,
        seed=seed, source="agg", feature_names=np.array(all_names),
    )
    print(f"  Saved NPZ: {npz_path}")

    # ── Classifiers ──────────────────────────────────────────────────────
    print("\n[STEP] Training classifiers...")
    clf_results = run_classifiers(
        train_X, dev_X, test_X, train_Y, dev_Y, test_Y,
        n_classes=n_classes, seed=seed, device=DEVICE,
    )

    print("\n" + "=" * 72)
    print(f"{'Model':<12} {'Train F1':<10} {'Dev F1':<10} {'Test F1':<10}")
    print("-" * 42)
    for name, res in clf_results.items():
        print(f"{name.upper():<12} {res['train_f1']:<10.4f} {res['dev_f1']:<10.4f} {res['test_f1']:<10.4f}")
    print("=" * 72)

    print(f"\nFor Optuna-tuned classifiers: python scripts/tune_optuna.py --npz {npz_path} --models rf xgb mlp")

    tee.close()


if __name__ == "__main__":
    main()
