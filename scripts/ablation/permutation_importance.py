"""Grouped permutation feature importance for LingRF style variants.

Trains the full model once per seed, then measures macro-F1 drop when each
feature group is permuted on the test set (each column independently).
Repeats the permutation N_REPEATS times per seed for stability.

Groups
------
  Linguistic    — all original linguistic features (one block)
  LexicalDiversity, SentenceStructure, RepetitionPatterns,
  WordLevelStatistics, FunctionalStylisticMarkers, ReadabilityMetrics,
  PunctuationUsage  — the 7 style feature groups
  LSTM_Probs    — BiLSTM output probabilities (predout variant only)

Output
------
  data/out/results/perm_importance_{subtask}_{lang}_{variant}.tsv
  data/out/results/perm_importance_latest.tsv  (all configs combined)
  data/out/perm_importance/plots/*.png

Usage
-----
  python scripts/ablation/permutation_importance.py
  python scripts/ablation/permutation_importance.py --subtask subtask_1 --lang en
  python scripts/ablation/permutation_importance.py --variants lingrf_style_predout
  python scripts/ablation/permutation_importance.py --multilingual
  python scripts/ablation/permutation_importance.py --seeds 10 11 12 --repeats 5
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

SCRIPTS_DIR: Path = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _bootstrap import configure_project_root

PROJECT_ROOT: str = str(configure_project_root(__file__, remove_shadowing_utils=True))

from feature_extraction.linguistic_features import (
    LingRFClassifier,
    LingRFPredOutClassifier,
    LinguisticFeatures,
)
from feature_extraction.style_features import StyleFeatures
from scripts.ablation.constants import GROUP_ORDER, STYLE_FEATURE_SET, STYLE_GROUPS
from utils.constants import OUT_DIR, RESULTS_DIR
from utils.data_utils import load_train_dev_test

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
ABLATION_PROBS_DIR: Path = OUT_DIR / "ablation_probs"
ABLATION_LING_CACHE_DIR: Path = OUT_DIR / "ablation_ling_cache"
PERM_PLOTS_DIR: Path = OUT_DIR / "perm_importance" / "plots"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CODE_SPLIT = False
USE_FOLD = 0
N_ESTIMATORS = 200
MAX_DEPTH = 60

def load_pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grouped permutation feature importance for LingRF style variants."
    )
    parser.add_argument("--subtask", choices=["subtask_1", "subtask_2", "all"], default="all")
    parser.add_argument("--lang", choices=["en", "es", "all"], default="all")
    parser.add_argument(
        "--variants", nargs="+",
        choices=["lingrf_style", "lingrf_style_predout"],
        default=["lingrf_style"],
        help="Variants to evaluate (default: lingrf_style).",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[10, 11, 12],
        help="Random seeds to train with (default: 10 11 12).",
    )
    parser.add_argument(
        "--repeats", type=int, default=5,
        help="Permutation repeats per seed (default: 5).",
    )
    parser.add_argument(
        "--multilingual", action="store_true",
        help="Use multilingual LSTM probs for predout variant.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Permutation helpers
# ---------------------------------------------------------------------------

def _permute_group_in_X(
    X: np.ndarray,
    feature_names: list[str],
    group_features: set[str],
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a copy of X with the columns belonging to group_features permuted."""
    X_perm = X.copy()
    for i, name in enumerate(feature_names):
        if name in group_features:
            X_perm[:, i] = rng.permutation(X_perm[:, i])
    return X_perm


def _permute_pred_probs(
    pred_probs: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a copy of pred_probs with each column independently permuted."""
    pp = pred_probs.copy()
    for col in range(pp.shape[1]):
        pp[:, col] = rng.permutation(pp[:, col])
    return pp


# ---------------------------------------------------------------------------
# Feature extraction (with ling cache)
# ---------------------------------------------------------------------------

def extract_features(
    subtask: str,
    lang: str,
    train_texts: list[str],
    dev_texts: list[str],
    test_texts: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Extract and concatenate ling + style features. Returns (train, dev, test, names)."""
    resources_dir = Path(PROJECT_ROOT) / "resources"

    # Linguistic features (cached)
    ABLATION_LING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ling_cache_path = ABLATION_LING_CACHE_DIR / f"{subtask}_{lang}_ling_features.npz"
    if ling_cache_path.exists():
        print(f"  [LING] Loading cache: {ling_cache_path}")
        _ld = np.load(ling_cache_path, allow_pickle=True)
        train_ling_X = _ld["train_ling_X"]
        dev_ling_X = _ld["dev_ling_X"]
        test_ling_X = _ld["test_ling_X"]
        ling_feature_names: list[str] = list(_ld["ling_feature_names"])
    else:
        print(f"  [LING] Extracting (will cache)...")
        ling_extractor = LinguisticFeatures(language=lang, resources_dir=resources_dir)
        train_ling_X, ling_feature_names = ling_extractor.extract_features(train_texts)
        dev_ling_X, _ = ling_extractor.extract_features(dev_texts, feature_names=ling_feature_names)
        test_ling_X, _ = ling_extractor.extract_features(test_texts, feature_names=ling_feature_names)
        np.savez_compressed(
            ling_cache_path,
            train_ling_X=train_ling_X, dev_ling_X=dev_ling_X, test_ling_X=test_ling_X,
            ling_feature_names=np.array(ling_feature_names),
        )

    # Style features (StyleFeatures has its own disk cache)
    style_extractor = StyleFeatures(language=lang)
    train_style_X, style_names = style_extractor.extract(
        train_texts, cache_key=f"train_{subtask}_{lang}"
    )
    dev_style_X, _ = style_extractor.extract(dev_texts, cache_key=f"dev_{subtask}_{lang}")
    test_style_X, _ = style_extractor.extract(test_texts, cache_key=f"test_{subtask}_{lang}")

    feature_names: list[str] = ling_feature_names + style_names
    train_X = np.concatenate([train_ling_X, train_style_X], axis=1)
    dev_X = np.concatenate([dev_ling_X, dev_style_X], axis=1)
    test_X = np.concatenate([test_ling_X, test_style_X], axis=1)
    return train_X, dev_X, test_X, feature_names


# ---------------------------------------------------------------------------
# Single seed run
# ---------------------------------------------------------------------------

def run_one_seed(
    seed: int,
    n_repeats: int,
    subtask: str,
    lang: str,
    variant: str,
    multilingual: bool,
    train_X: np.ndarray,
    test_X: np.ndarray,
    train_Y: np.ndarray,
    test_Y: np.ndarray,
    feature_names: list[str],
    test_pred_probs: np.ndarray | None,
    train_pred_probs: np.ndarray | None,
) -> dict[str, float]:
    """
    Train one model with `seed`, compute baseline test F1, then for each group
    run `n_repeats` permutations and record mean F1 drop.

    Returns dict: group_name → mean F1 drop across repeats.
    """
    random.seed(seed)
    np.random.seed(seed)

    print(f"\n  [Seed {seed}] Training {variant} on {subtask}/{lang}...")

    if variant == "lingrf_style":
        clf = LingRFClassifier(
            n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=seed,
        )
        clf.fit(train_X, train_Y, feature_names=feature_names)
        baseline_preds = clf.predict(test_X)
    else:  # lingrf_style_predout
        clf = LingRFPredOutClassifier(
            n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=seed,
        )
        clf.fit(train_X, train_pred_probs, train_Y, feature_names=feature_names)
        baseline_preds = clf.predict(test_X, test_pred_probs)

    baseline_f1 = f1_score(test_Y, baseline_preds, average="macro")
    print(f"  [Seed {seed}] Baseline test F1: {baseline_f1:.4f}")

    # Build group → feature-index mapping
    ling_features: set[str] = {n for n in feature_names if n not in STYLE_FEATURE_SET}
    group_feature_map: dict[str, set[str]] = {"Linguistic": ling_features}
    for gname, gfeats in STYLE_GROUPS.items():
        group_feature_map[gname] = set(gfeats)
    if variant == "lingrf_style_predout":
        group_feature_map["LSTM_Probs"] = set()  # handled separately via pred_probs

    drops: dict[str, list[float]] = {g: [] for g in group_feature_map}

    for repeat in range(n_repeats):
        rng = np.random.default_rng(seed * 1000 + repeat)
        for group_name in group_feature_map:
            if group_name == "LSTM_Probs":
                perm_probs = _permute_pred_probs(test_pred_probs, rng)
                perm_preds = clf.predict(test_X, perm_probs)
            else:
                X_perm = _permute_group_in_X(
                    test_X, feature_names, group_feature_map[group_name], rng
                )
                if variant == "lingrf_style":
                    perm_preds = clf.predict(X_perm)
                else:
                    perm_preds = clf.predict(X_perm, test_pred_probs)

            perm_f1 = f1_score(test_Y, perm_preds, average="macro")
            drop = baseline_f1 - perm_f1
            drops[group_name].append(drop)

    return {g: float(np.mean(v)) for g, v in drops.items()}, baseline_f1


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_perm_importance(
    group_names: list[str],
    means: np.ndarray,
    stds: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    plt = load_pyplot()
    # Sort ascending so most important is at top
    order = np.argsort(means)
    sorted_names = [group_names[i] for i in order]
    sorted_means = means[order]
    sorted_stds = stds[order]

    fig, ax = plt.subplots(figsize=(10, max(4, len(group_names) * 0.55)))
    y_pos = np.arange(len(sorted_names))
    colors = ["#d73027" if m > 0 else "#4575b4" for m in sorted_means]

    ax.barh(y_pos, sorted_means, xerr=sorted_stds, color=colors,
            height=0.65, capsize=4, error_kw={"linewidth": 1.5, "ecolor": "#555"})
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names, fontsize=11)
    ax.set_xlabel("Mean F1 drop when group is permuted (↑ = more important)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Value labels
    for i, (m, s) in enumerate(zip(sorted_means, sorted_stds)):
        ax.text(
            m + sorted_stds[i] + 0.0005, i,
            f"{m:+.4f} ±{s:.4f}", va="center", fontsize=8, color="#333"
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Saved plot: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    subtasks = ["subtask_1", "subtask_2"] if args.subtask == "all" else [args.subtask]
    languages = ["en", "es"] if args.lang == "all" else [args.lang]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PERM_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []

    for subtask in subtasks:
        for lang in languages:
            print(f"\n{'=' * 70}")
            print(f"Subtask: {subtask}  Lang: {lang}")
            print(f"{'=' * 70}")

            data_dir = Path(PROJECT_ROOT) / "data" / "data"
            try:
                (
                    train_texts, dev_texts, test_texts,
                    train_Y, dev_Y, test_Y,
                    _train_idx, _dev_idx,
                ) = load_train_dev_test(
                    train_dir=data_dir / "train" / subtask / lang,
                    test_dir=data_dir / "test" / subtask / lang,
                    subtask=subtask,
                    seed=args.seeds[0],
                    code_split=CODE_SPLIT,
                    use_fold=USE_FOLD,
                )
            except FileNotFoundError as e:
                print(f"[ERROR] Data not found: {e}")
                continue

            print(f"  Data — Train: {len(train_texts)}, Dev: {len(dev_texts)}, Test: {len(test_texts)}")

            # Extract features once per (subtask, lang)
            train_X, dev_X, test_X, feature_names = extract_features(
                subtask, lang, train_texts, dev_texts, test_texts
            )
            print(f"  Features: {train_X.shape[1]} ({len(feature_names)} names)")

            for variant in args.variants:
                print(f"\n{'─' * 70}")
                print(f"Variant: {variant}  Seeds: {args.seeds}  Repeats/seed: {args.repeats}")
                print(f"{'─' * 70}")

                # Load LSTM probs if needed
                train_pred_probs = test_pred_probs = None
                if variant == "lingrf_style_predout":
                    ml_suffix = "_multilingual" if args.multilingual else ""
                    probs_path = ABLATION_PROBS_DIR / f"{subtask}_{lang}{ml_suffix}_pred_probs.npz"
                    if not probs_path.exists():
                        print(f"  [ERROR] LSTM probs not found: {probs_path}")
                        print("  Run scripts/ablation/precompute_lstm_probs.py first.")
                        continue
                    probs_data = np.load(probs_path)
                    train_pred_probs = probs_data["train_probs"]
                    test_pred_probs = probs_data["test_probs"]
                    print(f"  Loaded LSTM probs: {test_pred_probs.shape}")

                # Collect per-seed mean drops
                seed_drops: dict[str, list[float]] = {}  # group → [drop_seed1, drop_seed2, ...]
                seed_baselines: list[float] = []

                for seed in args.seeds:
                    drops_this_seed, baseline_f1 = run_one_seed(
                        seed=seed,
                        n_repeats=args.repeats,
                        subtask=subtask,
                        lang=lang,
                        variant=variant,
                        multilingual=args.multilingual,
                        train_X=train_X,
                        test_X=test_X,
                        train_Y=train_Y,
                        test_Y=test_Y,
                        feature_names=feature_names,
                        test_pred_probs=test_pred_probs,
                        train_pred_probs=train_pred_probs,
                    )
                    seed_baselines.append(baseline_f1)
                    for g, d in drops_this_seed.items():
                        seed_drops.setdefault(g, []).append(d)

                # Compute mean ± std across seeds
                active_groups = [g for g in GROUP_ORDER if g in seed_drops]
                means = np.array([np.mean(seed_drops[g]) for g in active_groups])
                stds = np.array([np.std(seed_drops[g]) for g in active_groups])

                print(f"\n  Baseline test F1 across seeds: "
                      f"{np.mean(seed_baselines):.4f} ± {np.std(seed_baselines):.4f}")
                print(f"\n  {'Group':<28} {'Mean Drop':>10}  {'Std':>8}")
                print(f"  {'─'*50}")
                for g, m, s in sorted(
                    zip(active_groups, means, stds), key=lambda x: -x[1]
                ):
                    print(f"  {g:<28} {m:>+10.4f}  {s:>8.4f}")

                # Save per-config TSV
                tsv_path = RESULTS_DIR / f"perm_importance_{subtask}_{lang}_{variant}.tsv"
                with open(tsv_path, "w", encoding="utf-8", newline="") as f:
                    writer = csv.writer(f, delimiter="\t")
                    seed_cols = [f"seed_{s}" for s in args.seeds]
                    writer.writerow(
                        ["subtask", "lang", "variant", "group",
                         "mean_drop", "std_drop", "baseline_f1_mean"] + seed_cols
                    )
                    for g, m, s in zip(active_groups, means, stds):
                        writer.writerow([
                            subtask, lang, variant, g,
                            f"{m:.4f}", f"{s:.4f}",
                            f"{np.mean(seed_baselines):.4f}",
                        ] + [f"{seed_drops[g][i]:.4f}" for i in range(len(args.seeds))])
                print(f"\n  Saved: {tsv_path}")

                # Accumulate for combined output
                for g, m, s in zip(active_groups, means, stds):
                    all_rows.append({
                        "subtask": subtask, "lang": lang, "variant": variant, "group": g,
                        "mean_drop": f"{m:.4f}", "std_drop": f"{s:.4f}",
                        "baseline_f1_mean": f"{np.mean(seed_baselines):.4f}",
                        **{f"seed_{sv}": f"{seed_drops[g][i]:.4f}"
                           for i, sv in enumerate(args.seeds)},
                    })

                # Plot
                ml_tag = "_multilingual" if args.multilingual else ""
                plot_path = PERM_PLOTS_DIR / f"perm_importance_{subtask}_{lang}_{variant}{ml_tag}.png"
                plot_perm_importance(
                    group_names=active_groups,
                    means=means,
                    stds=stds,
                    title=f"Grouped Permutation Importance\n{subtask} / {lang} / {variant}",
                    output_path=plot_path,
                )

    # Combined TSV
    if all_rows:
        latest_path = RESULTS_DIR / "perm_importance_latest.tsv"
        fieldnames = list(all_rows[0].keys())
        with open(latest_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\n[INFO] Combined results: {latest_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
