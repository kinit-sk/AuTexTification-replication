"""Generates SHAP bar/beeswarm plots from saved SHAP .npz files."""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from utils.constants import SHAP_DIR

SHAP_PLOTS_DIR = SHAP_DIR / "plots"
SHAP_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

SHAP_DIR_STR = str(SHAP_DIR)
SHAP_PLOTS_DIR_STR = str(SHAP_PLOTS_DIR)


def plot_bar(feature_names, mean_shap, title, output_path, top_n=30):
    sorted_idx = np.argsort(mean_shap)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(12, max(8, top_n * 0.35)))

    y_pos = np.arange(len(sorted_idx))
    features = [feature_names[i] for i in sorted_idx][::-1]
    importance = mean_shap[sorted_idx][::-1]

    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(sorted_idx)))
    bars = ax.barh(y_pos, importance, color=colors, height=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=10)
    ax.set_xlabel("Mean |SHAP value| (feature importance)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    max_val = importance.max()
    for bar, val in zip(bars, importance):
        ax.text(
            val + max_val * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            fontsize=8,
            color="#333",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_beeswarm(feature_names, shap_values, X_test, title, output_path, top_n=25):
    """Save a beeswarm-style SHAP summary plot for top-N features."""
    n_samples = shap_values.shape[0]
    X_test = X_test[:n_samples]

    mean_shap = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_shap)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(12, max(8, top_n * 0.4)))

    for plot_idx, feat_idx in enumerate(sorted_idx[::-1]):
        sv = shap_values[:, feat_idx]
        fv = X_test[:, feat_idx]

        fv_min, fv_max = fv.min(), fv.max()
        if fv_max - fv_min > 1e-10:
            fv_norm = (fv - fv_min) / (fv_max - fv_min)
        else:
            fv_norm = np.full_like(fv, 0.5)

        y_jitter = np.random.normal(0, 0.15, len(sv))

        ax.scatter(
            sv,
            plot_idx + y_jitter,
            c=fv_norm,
            cmap="coolwarm",
            alpha=0.5,
            s=15,
            vmin=0,
            vmax=1,
            edgecolors="none",
        )

    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx[::-1]], fontsize=10)
    ax.axvline(x=0, color="#888", linestyle="-", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=25, pad=0.02)
    cbar.set_label("Feature value\n(Low → High)", fontsize=10)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["Low", "Mid", "High"])

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_style_features(feature_names, mean_shap, title, output_path):
    """Save a bar plot restricted to known style features."""
    style_list = [
        "ttr",
        "root_ttr",
        "log_ttr",
        "hapax_ratio",
        "dis_legomena_ratio",
        "avg_sentence_length",
        "sentence_length_std",
        "sentence_length_cv",
        "sentence_count",
        "bigram_repetition",
        "trigram_repetition",
        "avg_word_length",
        "word_length_std",
        "word_count",
        "function_word_ratio",
        "transition_word_ratio",
        "hedge_word_ratio",
        "flesch_reading_ease",
        "flesch_kincaid_grade",
        "punctuation_ratio",
        "comma_ratio",
        "rare_word_burstiness",
        "exclamation_ratio",
        "question_ratio",
        "first_person_ratio",
        "formal_word_ratio",
    ]

    style_data = []
    feature_names_list = list(feature_names)
    for f in style_list:
        if f in feature_names_list:
            idx = feature_names_list.index(f)
            style_data.append((f, mean_shap[idx]))

    if not style_data:
        print("No style features found")
        return

    style_data.sort(key=lambda x: x[1], reverse=True)
    features = [x[0] for x in style_data][::-1]
    importance = [x[1] for x in style_data][::-1]

    fig, ax = plt.subplots(figsize=(10, max(6, len(features) * 0.35)))

    y_pos = np.arange(len(features))
    colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(features)))
    ax.barh(y_pos, importance, color=colors, height=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=11)
    ax.set_xlabel("Mean |SHAP value|", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate SHAP plots from saved values")
    parser.add_argument("--subtask", default="subtask_1")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--variant", default="lingrf_style")
    parser.add_argument("--all", action="store_true", help="Process all available SHAP files")
    args = parser.parse_args()

    if args.all:
        import glob

        files = glob.glob(os.path.join(SHAP_DIR_STR, "*_shap_values.npz"))
        configs = []
        for f in files:
            basename = os.path.basename(f).replace("_shap_values.npz", "")
            parts = basename.split("_")
            if len(parts) >= 4:
                subtask = f"{parts[0]}_{parts[1]}"
                lang = parts[2]
                variant = "_".join(parts[3:])
                configs.append((subtask, lang, variant))
    else:
        configs = [(args.subtask, args.lang, args.variant)]

    for subtask, lang, variant in configs:
        prefix = f"{subtask}_{lang}_{variant}"
        npz_path = os.path.join(SHAP_DIR_STR, f"{prefix}_shap_values.npz")

        if not os.path.exists(npz_path):
            print(f"Not found: {npz_path}")
            continue

        print(f"\n{'='*60}")
        print(f"Plotting: {subtask} / {lang} / {variant}")
        print(f"{'='*60}")

        data = np.load(npz_path, allow_pickle=True)
        shap_values_raw = data["shap_values"]
        feature_names = list(data["feature_names"])
        X_test_sample = data["X_test_sample"]

        if len(shap_values_raw.shape) == 3:
            if shap_values_raw.shape[2] <= 10:
                n_classes = shap_values_raw.shape[2]
                if n_classes == 2:
                    sv = shap_values_raw[:, :, 1]
                else:
                    sv = np.mean(np.abs(shap_values_raw), axis=2)
            else:
                n_classes = shap_values_raw.shape[0]
                if n_classes == 2:
                    sv = shap_values_raw[1]
                else:
                    sv = np.mean(np.abs(shap_values_raw), axis=0)
        else:
            sv = shap_values_raw

        assert sv.shape[0] == X_test_sample.shape[0], f"Sample mismatch: {sv.shape[0]} vs {X_test_sample.shape[0]}"
        assert sv.shape[1] == len(feature_names), f"Feature mismatch: {sv.shape[1]} vs {len(feature_names)}"

        mean_shap = np.abs(sv).mean(axis=0)
        title_base = f"{subtask} / {lang} / {variant}"

        plot_bar(
            feature_names,
            mean_shap,
            f"SHAP Feature Importance - Top 30\n{title_base}",
            os.path.join(SHAP_PLOTS_DIR_STR, f"{prefix}_bar.png"),
            top_n=30,
        )

        plot_beeswarm(
            feature_names,
            sv,
            X_test_sample,
            f"SHAP Summary (Beeswarm) - Top 25\n{title_base}",
            os.path.join(SHAP_PLOTS_DIR_STR, f"{prefix}_beeswarm.png"),
            top_n=25,
        )

        plot_style_features(
            feature_names,
            mean_shap,
            f"SHAP - Style Features Only\n{title_base}",
            os.path.join(SHAP_PLOTS_DIR_STR, f"{prefix}_style_only.png"),
        )

    print(f"\nDone! Plots saved to: {SHAP_PLOTS_DIR_STR}")


if __name__ == "__main__":
    main()