"""Generate SHAP plots from saved .npz files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from types import ModuleType

import numpy as np

from _bootstrap import configure_project_root

configure_project_root(__file__, remove_shadowing_utils=False)

from scripts.ablation.constants import (
    STYLE_FEATURE_SET,
    STYLE_GROUPS,
    SUBTASK1_CLASS_NAMES,
    SUBTASK2_CLASS_NAMES,
)
from utils.constants import SHAP_DIR

SHAP_PLOTS_DIR: Path = SHAP_DIR / "plots"
SHAP_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

STYLE_FEATURE_NAMES: tuple[str, ...] = (
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
)


def load_pyplot() -> ModuleType:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def parse_shap_filename(npz_path: Path) -> tuple[str, str, str, str]:
    basename: str = npz_path.name.removesuffix("_shap_values.npz")
    parts: list[str] = basename.split("_")
    if len(parts) >= 4:
        return basename, f"{parts[0]}_{parts[1]}", parts[2], "_".join(parts[3:])
    return basename, "unknown", "unknown", basename


def collect_shap_files(args: argparse.Namespace) -> list[Path]:
    if args.all:
        return sorted(SHAP_DIR.glob("*_shap_values.npz"))

    prefix: str = f"{args.subtask}_{args.lang}_{args.variant}"
    npz_path: Path = SHAP_DIR / f"{prefix}_shap_values.npz"
    if npz_path.exists():
        return [npz_path]

    ablation_npz_path: Path = SHAP_DIR / f"{prefix}_ablation_shap_values.npz"
    if ablation_npz_path.exists():
        return [ablation_npz_path]

    print(f"Not found: {npz_path}")
    return []


def plot_bar(
    feature_names: list[str],
    mean_shap: np.ndarray,
    title: str,
    output_path: Path,
    top_n: int,
) -> None:
    plt = load_pyplot()
    sorted_idx: np.ndarray = np.argsort(mean_shap)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(12, max(8, top_n * 0.35)))

    y_pos: np.ndarray = np.arange(len(sorted_idx))
    features: list[str] = [feature_names[i] for i in sorted_idx][::-1]
    importance: np.ndarray = mean_shap[sorted_idx][::-1]
    colors: np.ndarray = plt.cm.Reds(np.linspace(0.3, 0.9, len(sorted_idx)))
    bars = ax.barh(y_pos, importance, color=colors, height=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=10)
    ax.set_xlabel("Mean |SHAP value| (feature importance)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    max_val: float = float(importance.max())
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


def plot_beeswarm(
    feature_names: list[str],
    shap_values: np.ndarray,
    x_test: np.ndarray,
    title: str,
    output_path: Path,
    top_n: int,
) -> None:
    plt = load_pyplot()
    n_samples: int = shap_values.shape[0]
    x_test = x_test[:n_samples]

    mean_shap: np.ndarray = np.abs(shap_values).mean(axis=0)
    sorted_idx: np.ndarray = np.argsort(mean_shap)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(12, max(8, top_n * 0.4)))

    for plot_idx, feat_idx in enumerate(sorted_idx[::-1]):
        shap_column: np.ndarray = shap_values[:, feat_idx]
        feature_column: np.ndarray = x_test[:, feat_idx]
        feature_min: float = float(feature_column.min())
        feature_max: float = float(feature_column.max())
        if feature_max - feature_min > 1e-10:
            feature_norm: np.ndarray = (feature_column - feature_min) / (feature_max - feature_min)
        else:
            feature_norm = np.full_like(feature_column, 0.5)

        y_jitter: np.ndarray = np.random.normal(0, 0.15, len(shap_column))
        ax.scatter(
            shap_column,
            plot_idx + y_jitter,
            c=feature_norm,
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
    cbar.set_label("Feature value\n(Low -> High)", fontsize=10)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["Low", "Mid", "High"])

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_style_features(
    feature_names: list[str],
    mean_shap: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    plt = load_pyplot()
    style_data: list[tuple[str, float]] = []
    for feature_name in STYLE_FEATURE_NAMES:
        if feature_name in feature_names:
            feature_idx: int = feature_names.index(feature_name)
            style_data.append((feature_name, float(mean_shap[feature_idx])))

    if not style_data:
        print("No style features found")
        return

    style_data.sort(key=lambda item: item[1], reverse=True)
    features: list[str] = [item[0] for item in style_data][::-1]
    importance: list[float] = [item[1] for item in style_data][::-1]

    fig, ax = plt.subplots(figsize=(10, max(6, len(features) * 0.35)))
    y_pos: np.ndarray = np.arange(len(features))
    colors: np.ndarray = plt.cm.Greens(np.linspace(0.3, 0.9, len(features)))
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


def standard_shap_matrix(shap_values_raw: np.ndarray) -> np.ndarray:
    if len(shap_values_raw.shape) == 3:
        if shap_values_raw.shape[2] <= 10:
            n_classes: int = shap_values_raw.shape[2]
            if n_classes == 2:
                return shap_values_raw[:, :, 1]
            return np.mean(np.abs(shap_values_raw), axis=2)

        n_classes = shap_values_raw.shape[0]
        if n_classes == 2:
            return shap_values_raw[1]
        return np.mean(np.abs(shap_values_raw), axis=0)

    return shap_values_raw


def process_standard_npz(npz_path: Path) -> None:
    basename, subtask, lang, variant = parse_shap_filename(npz_path)

    print(f"\n{'=' * 60}")
    print(f"Plotting: {subtask} / {lang} / {variant}")
    print(f"{'=' * 60}")

    data = np.load(npz_path, allow_pickle=True)
    shap_values_raw: np.ndarray = data["shap_values"]
    feature_names: list[str] = list(data["feature_names"])
    x_test_sample: np.ndarray = data["X_test_sample"]
    shap_values: np.ndarray = standard_shap_matrix(shap_values_raw)

    assert shap_values.shape[0] == x_test_sample.shape[0], (
        f"Sample mismatch: {shap_values.shape[0]} vs {x_test_sample.shape[0]}"
    )
    assert shap_values.shape[1] == len(feature_names), (
        f"Feature mismatch: {shap_values.shape[1]} vs {len(feature_names)}"
    )

    mean_shap: np.ndarray = np.abs(shap_values).mean(axis=0)
    title_base: str = f"{subtask} / {lang} / {variant}"

    plot_bar(
        feature_names,
        mean_shap,
        f"SHAP Feature Importance - Top 30\n{title_base}",
        SHAP_PLOTS_DIR / f"{basename}_bar.png",
        30,
    )
    plot_beeswarm(
        feature_names,
        shap_values,
        x_test_sample,
        f"SHAP Summary (Beeswarm) - Top 25\n{title_base}",
        SHAP_PLOTS_DIR / f"{basename}_beeswarm.png",
        25,
    )
    plot_style_features(
        feature_names,
        mean_shap,
        f"SHAP - Style Features Only\n{title_base}",
        SHAP_PLOTS_DIR / f"{basename}_style_only.png",
    )


def feature_group_for_name(feature_name: str) -> str:
    if feature_name.startswith("PRED_PROB_"):
        return "LSTM_Probs"
    for group_name, group_features in STYLE_GROUPS.items():
        if feature_name in group_features:
            return group_name
    return "Original Features"


def per_class_mean_abs(shap_values_raw: np.ndarray) -> np.ndarray | None:
    if shap_values_raw.ndim != 3:
        return None

    dim0, _dim1, dim2 = shap_values_raw.shape
    if 2 <= dim2 <= 20:
        return np.mean(np.abs(shap_values_raw), axis=0)
    if 2 <= dim0 <= 20:
        return np.mean(np.abs(shap_values_raw), axis=1).T
    return None


def sample_feature_class_shap(shap_values_raw: np.ndarray) -> np.ndarray | None:
    if shap_values_raw.ndim != 3:
        return None

    dim0, _dim1, dim2 = shap_values_raw.shape
    if 2 <= dim2 <= 20:
        return shap_values_raw
    if 2 <= dim0 <= 20:
        return np.transpose(shap_values_raw, (1, 2, 0))
    return None


def safe_pearsonr(x_values: np.ndarray, y_values: np.ndarray) -> float:
    if x_values.size < 2 or y_values.size < 2:
        return 0.0
    if float(np.std(x_values)) == 0.0 or float(np.std(y_values)) == 0.0:
        return 0.0
    return float(np.corrcoef(x_values, y_values)[0, 1])


def direction_label(correlation: float, low_mean: float, high_mean: float) -> str:
    if correlation >= 0.10 and high_mean > low_mean:
        return "high_values_support_class"
    if correlation <= -0.10 and low_mean > high_mean:
        return "low_values_support_class"
    return "no_clear_direction"


def direction_summary_rows(
    npz_path: Path,
    subtask: str,
    lang: str,
    variant: str,
    feature_names: list[str],
    shap_values_raw: np.ndarray,
    x_test_sample: np.ndarray,
    class_names: list[str],
    top_n: int,
) -> list[dict[str, str]]:
    shap_3d: np.ndarray | None = sample_feature_class_shap(shap_values_raw)
    if shap_3d is None:
        print("  [SKIP] Direction summary requires a 3-D SHAP array.")
        return []

    if x_test_sample.shape[1] != len(feature_names):
        raise ValueError(
            f"Feature-name mismatch for {npz_path}: "
            f"X has {x_test_sample.shape[1]} columns, names={len(feature_names)}"
        )

    n_classes: int = shap_3d.shape[2]
    if len(class_names) != n_classes:
        class_names = [f"Class {class_idx}" for class_idx in range(n_classes)]

    rows: list[dict[str, str]] = []
    for class_idx in range(n_classes):
        class_shap: np.ndarray = shap_3d[:, :, class_idx]
        mean_abs: np.ndarray = np.mean(np.abs(class_shap), axis=0)
        top_indices: np.ndarray = np.argsort(mean_abs)[::-1][:top_n]

        for rank, feature_idx in enumerate(top_indices, start=1):
            feature_values: np.ndarray = x_test_sample[:, feature_idx]
            shap_values: np.ndarray = class_shap[:, feature_idx]
            median_value: float = float(np.median(feature_values))
            low_mask: np.ndarray = feature_values <= median_value
            high_mask: np.ndarray = feature_values > median_value
            low_mean: float = float(np.mean(shap_values[low_mask])) if np.any(low_mask) else 0.0
            high_mean: float = float(np.mean(shap_values[high_mask])) if np.any(high_mask) else 0.0
            correlation: float = safe_pearsonr(feature_values, shap_values)
            feature_name: str = feature_names[feature_idx]

            rows.append(
                {
                    "source_file": str(npz_path),
                    "subtask": subtask,
                    "lang": lang,
                    "variant": variant,
                    "class_index": str(class_idx),
                    "class_name": class_names[class_idx],
                    "rank": str(rank),
                    "feature": feature_name,
                    "feature_group": feature_group_for_name(feature_name),
                    "mean_abs_shap": f"{float(mean_abs[feature_idx]):.8f}",
                    "mean_signed_shap": f"{float(np.mean(shap_values)):.8f}",
                    "corr_feature_shap": f"{correlation:.8f}",
                    "low_value_mean_shap": f"{low_mean:.8f}",
                    "high_value_mean_shap": f"{high_mean:.8f}",
                    "direction": direction_label(correlation, low_mean, high_mean),
                }
            )

    return rows


def aggregate_by_group(
    per_class: np.ndarray,
    feature_names: list[str],
) -> tuple[list[str], np.ndarray]:
    feature_to_group: dict[str, str] = {}
    for feature_name in feature_names:
        if feature_name.startswith("PRED_PROB_"):
            feature_to_group[feature_name] = "LSTM_Probs"
        elif feature_name in STYLE_FEATURE_SET:
            for group_name, members in STYLE_GROUPS.items():
                if feature_name in members:
                    feature_to_group[feature_name] = group_name
                    break
        else:
            feature_to_group[feature_name] = "Original Features"

    group_order: list[str] = ["Original Features", *STYLE_GROUPS.keys()]
    if any(group_name == "LSTM_Probs" for group_name in feature_to_group.values()):
        group_order.append("LSTM_Probs")

    n_classes: int = per_class.shape[1]
    group_shap: dict[str, np.ndarray] = {group_name: np.zeros(n_classes) for group_name in group_order}
    group_count: dict[str, int] = {group_name: 0 for group_name in group_order}

    for feature_idx, feature_name in enumerate(feature_names):
        group_name: str = feature_to_group.get(feature_name, "Original Features")
        if group_name not in group_shap:
            group_shap[group_name] = np.zeros(n_classes)
            group_count[group_name] = 0
        group_shap[group_name] += per_class[feature_idx]
        group_count[group_name] += 1

    for group_name, count in group_count.items():
        if count > 0:
            group_shap[group_name] /= count

    active_groups: list[str] = [
        group_name
        for group_name in group_order
        if group_name in group_shap and group_shap[group_name].sum() > 0
    ]
    data: np.ndarray = np.stack([group_shap[group_name] for group_name in active_groups], axis=0)
    return active_groups, data


def plot_summary_grouped(
    feature_names: list[str],
    shap_values_raw: np.ndarray,
    title: str,
    output_path: Path,
    class_names: list[str] | None,
) -> None:
    plt = load_pyplot()
    per_class: np.ndarray | None = per_class_mean_abs(shap_values_raw)
    if per_class is None:
        print(f"  [SKIP] '{title}' - not multi-class.")
        return

    n_classes: int = per_class.shape[1]
    if class_names is None or len(class_names) != n_classes:
        class_names = [f"Class {class_idx}" for class_idx in range(n_classes)]

    group_names, data = aggregate_by_group(per_class, feature_names)
    total: np.ndarray = data.sum(axis=1)
    order: np.ndarray = np.argsort(total)
    group_names = [group_names[group_idx] for group_idx in order]
    data = data[order]

    fig, ax = plt.subplots(figsize=(11, max(4, len(group_names) * 0.55)))
    cmap = plt.cm.get_cmap("tab10" if n_classes <= 10 else "tab20", n_classes)
    colors = [cmap(class_idx) for class_idx in range(n_classes)]
    y_pos: np.ndarray = np.arange(len(group_names))
    lefts: np.ndarray = np.zeros(len(group_names))

    for class_idx in range(n_classes):
        segment: np.ndarray = data[:, class_idx]
        ax.barh(y_pos, segment, left=lefts, color=colors[class_idx], label=class_names[class_idx], height=0.65)
        lefts += segment

    ax.set_yticks(y_pos)
    ax.set_yticklabels(group_names, fontsize=11)
    ax.set_xlabel("Mean of mean(|SHAP value|) per feature within group", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(title="Class", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9, title_fontsize=10, frameon=False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_summary_all_classes(
    feature_names: list[str],
    shap_values_raw: np.ndarray,
    title: str,
    output_path: Path,
    top_n: int,
    class_names: list[str] | None,
) -> None:
    plt = load_pyplot()
    per_class: np.ndarray | None = per_class_mean_abs(shap_values_raw)
    if per_class is None:
        print(f"  [SKIP] '{title}' - array shape {shap_values_raw.shape} is not multi-class.")
        return

    n_classes: int = per_class.shape[1]
    if class_names is None or len(class_names) != n_classes:
        class_names = [f"Class {class_idx}" for class_idx in range(n_classes)]

    total_importance: np.ndarray = per_class.sum(axis=1)
    sorted_idx: np.ndarray = np.argsort(total_importance)[::-1][:top_n][::-1]
    features_sorted: list[str] = [feature_names[feature_idx] for feature_idx in sorted_idx]
    data_sorted: np.ndarray = per_class[sorted_idx]

    fig, ax = plt.subplots(figsize=(12, max(6, len(sorted_idx) * 0.40)))
    cmap = plt.cm.get_cmap("tab10" if n_classes <= 10 else "tab20", n_classes)
    colors = [cmap(class_idx) for class_idx in range(n_classes)]
    y_pos: np.ndarray = np.arange(len(sorted_idx))
    lefts: np.ndarray = np.zeros(len(sorted_idx))

    for class_idx in range(n_classes):
        segment: np.ndarray = data_sorted[:, class_idx]
        ax.barh(
            y_pos,
            segment,
            left=lefts,
            color=colors[class_idx],
            label=class_names[class_idx],
            height=0.7,
        )
        lefts += segment

    ax.set_yticks(y_pos)
    ax.set_yticklabels(features_sorted, fontsize=10)
    ax.set_xlabel("mean(|SHAP value|) (average impact on model output magnitude)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(title="Class", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9, title_fontsize=10, frameon=False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved: {output_path}")


def class_names_for_npz(data: np.lib.npyio.NpzFile, subtask: str) -> list[str] | None:
    if "class_names" in data:
        return [str(value) for value in data["class_names"]]
    if "subtask_2" in subtask:
        return SUBTASK2_CLASS_NAMES
    if "subtask_1" in subtask:
        return SUBTASK1_CLASS_NAMES
    return None


def process_multiclass_npz(
    npz_path: Path,
    top_n: int,
    grouped: bool,
    feature_level: bool,
    direction_summary: bool,
) -> list[dict[str, str]]:
    basename, subtask, lang, variant = parse_shap_filename(npz_path)

    print(f"\n{'=' * 60}")
    print(f"Multi-class SHAP: {subtask} / {lang} / {variant}")
    print(f"{'=' * 60}")

    data = np.load(npz_path, allow_pickle=True)
    shap_values_raw: np.ndarray = data["shap_values"]
    feature_names: list[str] = list(data["feature_names"])

    if shap_values_raw.ndim < 3:
        print(f"  [SKIP] {basename} has {shap_values_raw.ndim}-D SHAP array (not multi-class).")
        return []

    class_names: list[str] | None = class_names_for_npz(data, subtask)

    if feature_level:
        plot_summary_all_classes(
            feature_names=feature_names,
            shap_values_raw=shap_values_raw,
            title=f"{subtask} / {lang} / {variant}",
            output_path=SHAP_PLOTS_DIR / f"{basename}_shap_summary_all_classes.png",
            top_n=top_n,
            class_names=class_names,
        )

    if grouped:
        plot_summary_grouped(
            feature_names=feature_names,
            shap_values_raw=shap_values_raw,
            title=f"{subtask} / {lang} / {variant}  [feature groups]",
            output_path=SHAP_PLOTS_DIR / f"{basename}_shap_summary_grouped.png",
            class_names=class_names,
        )

    if not direction_summary:
        return []

    if "X_test_sample" not in data:
        print("  [SKIP] Direction summary requires X_test_sample in the SHAP .npz file.")
        return []

    return direction_summary_rows(
        npz_path=npz_path,
        subtask=subtask,
        lang=lang,
        variant=variant,
        feature_names=feature_names,
        shap_values_raw=shap_values_raw,
        x_test_sample=data["X_test_sample"],
        class_names=class_names or [],
        top_n=top_n,
    )


def write_direction_summary(direction_rows: list[dict[str, str]]) -> None:
    out_path: Path = SHAP_DIR / "shap_direction_summary_latest.tsv"
    fieldnames: list[str] = list(direction_rows[0].keys())
    with open(out_path, "w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(direction_rows)
    print(f"Saved direction summary: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SHAP plots from saved values.")
    parser.add_argument("--subtask", default="subtask_1")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--variant", default="lingrf_style")
    parser.add_argument("--all", action="store_true", help="Process all available SHAP files.")
    parser.add_argument(
        "--mode",
        choices=["standard", "multiclass", "both"],
        default="standard",
        help="Plot type to generate.",
    )
    parser.add_argument("--top-n", type=int, default=20, help="Top features for multiclass plots.")
    parser.add_argument("--grouped", action="store_true", default=True, help="Produce grouped multiclass plot.")
    parser.add_argument("--no-grouped", dest="grouped", action="store_false", help="Skip grouped multiclass plot.")
    parser.add_argument("--no-feature-level", dest="feature_level", action="store_false", help="Skip per-feature plot.")
    parser.add_argument(
        "--direction-summary",
        action="store_true",
        default=True,
        help="Write signed SHAP direction summary TSV.",
    )
    parser.add_argument(
        "--no-direction-summary",
        dest="direction_summary",
        action="store_false",
        help="Skip signed SHAP direction summary TSV.",
    )
    parser.set_defaults(feature_level=True)
    return parser.parse_args()


def main() -> None:
    args: argparse.Namespace = parse_args()
    npz_paths: list[Path] = collect_shap_files(args)
    if not npz_paths:
        print(f"No SHAP .npz files found in {SHAP_DIR}")
        return

    direction_rows: list[dict[str, str]] = []
    for npz_path in npz_paths:
        if args.mode in {"standard", "both"}:
            process_standard_npz(npz_path)
        if args.mode in {"multiclass", "both"}:
            direction_rows.extend(
                process_multiclass_npz(
                    npz_path=npz_path,
                    top_n=args.top_n,
                    grouped=args.grouped,
                    feature_level=args.feature_level,
                    direction_summary=args.direction_summary,
                )
            )

    if direction_rows:
        write_direction_summary(direction_rows)

    print(f"\nDone! Plots saved to: {SHAP_PLOTS_DIR}")


if __name__ == "__main__":
    main()
