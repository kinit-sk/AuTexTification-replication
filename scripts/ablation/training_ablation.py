"""Train one RF ablation configuration and write a single-row result table."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

SCRIPTS_DIR: Path = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _bootstrap import configure_project_root

PROJECT_ROOT: str = str(configure_project_root(__file__, remove_shadowing_utils=True))

import numpy as np
from sklearn.metrics import f1_score

from feature_extraction.linguistic_features import (
    LingRFClassifier,
    LingRFPredOutClassifier,
    LinguisticFeatures,
)
from feature_extraction.style_features import StyleFeatures
from scripts.ablation.constants import STYLE_GROUPS, class_names_for_subtask
from utils.constants import OUT_DIR, RESULTS_DIR, SHAP_DIR
from utils.data_utils import load_train_dev_test

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("[WARNING] shap not installed; SHAP analysis will be skipped.")

ABLATION_PROBS_DIR: Path = OUT_DIR / "ablation_probs"
ABLATION_LING_CACHE_DIR: Path = OUT_DIR / "ablation_ling_cache"

SEED = 10
NUMPY_SEED = 0
CODE_SPLIT = False
USE_FOLD = 0

N_ESTIMATORS = 200
MAX_DEPTH = 60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single ablation RF run with optional style group exclusion."
    )
    parser.add_argument("--subtask", choices=["subtask_1", "subtask_2"], required=True)
    parser.add_argument("--lang", choices=["en", "es"], required=True)
    parser.add_argument(
        "--variant",
        choices=["lingrf_style", "lingrf_style_predout"],
        default="lingrf_style",
    )
    parser.add_argument(
        "--exclude-style-groups",
        type=str,
        default="",
        dest="exclude_style_groups",
        help="Comma-separated group names to exclude (empty = baseline, no exclusion).",
    )
    parser.add_argument(
        "--multilingual", action="store_true",
        help="Use multilingual LSTM probs for predout variant.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed for the RF model. The train/dev split remains fixed.",
    )
    parser.add_argument("--no-shap", action="store_true", dest="no_shap",
                        help="Skip SHAP analysis.")
    parser.add_argument("--shap-samples", type=int, default=100, dest="shap_samples")
    return parser.parse_args()


def _run_shap_analysis(
    clf,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    subtask: str,
    lang: str,
    variant: str,
    excluded_group_key: str,
    shap_samples: int,
    multilingual: bool,
    seed: int,
) -> None:
    suffix = f"_excl_{excluded_group_key}" if excluded_group_key != "baseline" else ""
    multilingual_suffix = "_multilingual" if multilingual else ""
    prefix = f"{subtask}_{lang}_{variant}{multilingual_suffix}_ablation{suffix}_seed{seed}"
    print(f"\n[SHAP] Computing SHAP values for {prefix}...")

    n_test = min(shap_samples, len(X_test))
    idx_test = np.random.choice(len(X_test), n_test, replace=False)
    X_test_sample = X_test[idx_test]
    y_test_sample = y_test[idx_test]
    pred_sample = clf.model.predict(X_test_sample)
    class_names = class_names_for_subtask(subtask)

    explainer = shap.TreeExplainer(clf.model)
    shap_values = explainer.shap_values(X_test_sample)

    SHAP_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SHAP_DIR / f"{prefix}_shap_values.npz"
    np.savez_compressed(
        out_path,
        shap_values=np.array(shap_values) if isinstance(shap_values, list) else shap_values,
        feature_names=np.array(feature_names),
        X_test_sample=X_test_sample,
        y_test_sample=y_test_sample,
        pred_sample=pred_sample,
        sample_indices=idx_test,
        class_names=np.array(class_names),
    )
    print(f"  Saved: {out_path}")


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(NUMPY_SEED + args.seed)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    excluded_groups: list[str] = (
        [g.strip() for g in args.exclude_style_groups.split(",") if g.strip()]
        if args.exclude_style_groups
        else []
    )
    for group in excluded_groups:
        if group not in STYLE_GROUPS:
            print(
                f"[ERROR] Unknown style group: '{group}'. "
                f"Valid groups: {list(STYLE_GROUPS.keys())}"
            )
            sys.exit(1)

    excluded_feature_names: set[str] = set()
    for group in excluded_groups:
        excluded_feature_names.update(STYLE_GROUPS.get(group, []))

    excluded_group_key = "_".join(sorted(excluded_groups)) if excluded_groups else "baseline"

    print("\n" + "=" * 80)
    print(f"Ablation RF | {args.variant} | subtask={args.subtask} | lang={args.lang}")
    print(f"Seed            : {args.seed}")
    print(f"Excluded groups : {excluded_groups if excluded_groups else '(none — baseline)'}")
    print(f"Excluded features: {sorted(excluded_feature_names) if excluded_feature_names else '(none)'}")
    print("=" * 80)

    data_dir = Path(PROJECT_ROOT) / "data" / "data"
    resources_dir = Path(PROJECT_ROOT) / "resources"

    try:
        (
            train_texts, dev_texts, test_texts,
            train_Y, dev_Y, test_Y,
            _train_idx, _dev_idx,
        ) = load_train_dev_test(
            train_dir=data_dir / "train" / args.subtask / args.lang,
            test_dir=data_dir / "test" / args.subtask / args.lang,
            subtask=args.subtask,
            seed=SEED,
            code_split=CODE_SPLIT,
            use_fold=USE_FOLD,
        )
    except FileNotFoundError as e:
        print(f"[ERROR] Data not found for {args.subtask}/{args.lang}: {e}")
        sys.exit(1)

    print(f"\n[DATA] Train: {len(train_texts)}, Dev: {len(dev_texts)}, Test: {len(test_texts)}")

    ABLATION_LING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ling_cache_path = ABLATION_LING_CACHE_DIR / f"{args.subtask}_{args.lang}_ling_features.npz"

    if ling_cache_path.exists():
        print(f"\n[STEP] Loading cached linguistic features from {ling_cache_path}")
        _ld = np.load(ling_cache_path, allow_pickle=True)
        train_ling_X = _ld["train_ling_X"]
        dev_ling_X = _ld["dev_ling_X"]
        test_ling_X = _ld["test_ling_X"]
        ling_feature_names = list(_ld["ling_feature_names"])
        print(f"  Linguistic features: {train_ling_X.shape[1]}")
    else:
        print("\n[STEP] Extracting linguistic features...")
        ling_extractor = LinguisticFeatures(language=args.lang, resources_dir=resources_dir)
        train_ling_X, ling_feature_names = ling_extractor.extract_features(train_texts)
        dev_ling_X, _ = ling_extractor.extract_features(dev_texts, feature_names=ling_feature_names)
        test_ling_X, _ = ling_extractor.extract_features(test_texts, feature_names=ling_feature_names)
        print(f"  Linguistic features: {train_ling_X.shape[1]}")
        np.savez_compressed(
            ling_cache_path,
            train_ling_X=train_ling_X,
            dev_ling_X=dev_ling_X,
            test_ling_X=test_ling_X,
            ling_feature_names=np.array(ling_feature_names),
        )
        print(f"  Saved ling cache: {ling_cache_path}")

    print("\n[STEP] Extracting style features...")
    style_extractor = StyleFeatures(language=args.lang)
    train_style_X, style_feature_names = style_extractor.extract(
        train_texts, cache_key=f"train_{args.subtask}_{args.lang}"
    )
    dev_style_X, _ = style_extractor.extract(
        dev_texts, cache_key=f"dev_{args.subtask}_{args.lang}"
    )
    test_style_X, _ = style_extractor.extract(
        test_texts, cache_key=f"test_{args.subtask}_{args.lang}"
    )
    print(f"  Style features (before exclusion): {train_style_X.shape[1]}")

    if excluded_feature_names:
        keep_mask = [
            i for i, n in enumerate(style_feature_names)
            if n not in excluded_feature_names
        ]
        removed = [n for n in style_feature_names if n in excluded_feature_names]
        train_style_X = train_style_X[:, keep_mask]
        dev_style_X = dev_style_X[:, keep_mask]
        test_style_X = test_style_X[:, keep_mask]
        style_feature_names = [style_feature_names[i] for i in keep_mask]
        print(f"  Removed {len(removed)} style features: {removed}")
        print(f"  Remaining style features: {train_style_X.shape[1]}")

    train_X = np.concatenate([train_ling_X, train_style_X], axis=1)
    dev_X = np.concatenate([dev_ling_X, dev_style_X], axis=1)
    test_X = np.concatenate([test_ling_X, test_style_X], axis=1)
    feature_names: list[str] = ling_feature_names + style_feature_names
    print(
        f"\n  Combined features: {train_X.shape[1]} "
        f"(ling={len(ling_feature_names)}, style={len(style_feature_names)})"
    )

    train_pred_probs = dev_pred_probs = test_pred_probs = None

    if args.variant == "lingrf_style_predout":
        ml_suffix = "_multilingual" if args.multilingual else ""
        probs_path = ABLATION_PROBS_DIR / f"{args.subtask}_{args.lang}{ml_suffix}_pred_probs.npz"
        if not probs_path.exists():
            print(f"\n[ERROR] Pre-computed LSTM probs not found: {probs_path}")
            print("  Run: python scripts/ablation/precompute_lstm_probs.py first")
            sys.exit(1)
        print(f"\n[STEP] Loading pre-computed LSTM probs from {probs_path}")
        probs_data = np.load(probs_path)
        train_pred_probs = probs_data["train_probs"]
        dev_pred_probs = probs_data["dev_probs"]
        test_pred_probs = probs_data["test_probs"]
        print(f"  Prob shape: {train_pred_probs.shape}")

    print("\n[STEP] Training Random Forest...")
    print(f"  n_estimators={N_ESTIMATORS}, max_depth={MAX_DEPTH}")

    if args.variant == "lingrf_style":
        clf = LingRFClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            random_state=args.seed,
        )
        clf.fit(train_X, train_Y, feature_names=feature_names)
        train_preds = clf.predict(train_X)
        dev_preds = clf.predict(dev_X)
        test_preds = clf.predict(test_X)

    else:
        clf = LingRFPredOutClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            random_state=args.seed,
        )
        clf.fit(train_X, train_pred_probs, train_Y, feature_names=feature_names)

        train_preds = clf.predict(train_X, train_pred_probs)
        dev_preds = clf.predict(dev_X, dev_pred_probs)
        test_preds = clf.predict(test_X, test_pred_probs)

        prob_names = [f"PRED_PROB_{i}" for i in range(train_pred_probs.shape[1])]
        feature_names = feature_names + prob_names
        train_X = np.concatenate([train_X, train_pred_probs], axis=1)
        dev_X = np.concatenate([dev_X, dev_pred_probs], axis=1)
        test_X = np.concatenate([test_X, test_pred_probs], axis=1)

    train_f1 = f1_score(train_Y, train_preds, average="macro")
    dev_f1 = f1_score(dev_Y, dev_preds, average="macro")
    test_f1 = f1_score(test_Y, test_preds, average="macro")

    print("\n[RESULTS]")
    print(f"  Train F1: {train_f1:.4f}")
    print(f"  Dev F1:   {dev_f1:.4f}")
    print(f"  Test F1:  {test_f1:.4f}")
    print(f"  Dev-Test gap: {dev_f1 - test_f1:+.4f}")

    print("\n[FEATURE IMPORTANCE] Top 15 features:")
    for i, (name, imp) in enumerate(clf.get_feature_importance(top_k=15), 1):
        print(f"  {i:2d}. {name}: {imp:.4f}")

    if HAS_SHAP and not args.no_shap:
        _run_shap_analysis(
            clf=clf,
            X_test=test_X,
            y_test=test_Y,
            feature_names=feature_names,
            subtask=args.subtask,
            lang=args.lang,
            variant=args.variant,
            excluded_group_key=excluded_group_key,
            shap_samples=args.shap_samples,
            multilingual=args.multilingual,
            seed=args.seed,
        )

    n_style_out = len(style_feature_names)
    n_total_out = len(feature_names)

    out_filename = (
        f"ablation_single_{args.subtask}_{args.lang}_{args.variant}_{excluded_group_key}_seed{args.seed}.tsv"
    )
    out_path = RESULTS_DIR / out_filename
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(
            "subtask\tlang\tvariant\texcluded_group\tseed\t"
            "train_f1\tdev_f1\ttest_f1\t"
            "n_ling_features\tn_style_features\tn_total_features\n"
        )
        f.write(
            f"{args.subtask}\t{args.lang}\t{args.variant}\t{excluded_group_key}\t{args.seed}\t"
            f"{train_f1:.4f}\t{dev_f1:.4f}\t{test_f1:.4f}\t"
            f"{len(ling_feature_names)}\t{n_style_out}\t{n_total_out}\n"
        )
    print(f"\n[INFO] Saved ablation result to {out_path}")


if __name__ == "__main__":
    main()
