"""Runs inference on a saved text-classification checkpoint and exports error-analysis artifacts."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from utils.constants import (
    ENCODER_MAP_BASELINE,
    ENCODER_MULTILINGUAL,
    MODEL_VARIANTS,
    SUBTASK_1_LABELS,
    SUBTASK_2_LABELS,
)

VALID_SPLITS = ("train", "dev", "test")
VALID_VARIANTS = MODEL_VARIANTS
VALID_CONFIGS = ("baseline", "multilingual")


@dataclass(frozen=True)
class RunConfig:
    """Run settings for data loading, model rebuild, and inference."""

    subtask: str
    lang: str
    model_variant: str
    config: str
    seed: int
    encoder_id: str
    checkpoint_path: Path
    split: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run checkpoint error analysis.")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--subtask", choices=["subtask_1", "subtask_2"], default=None)
    parser.add_argument("--lang", choices=["en", "es"], default=None)
    parser.add_argument("--model_variant", choices=list(VALID_VARIANTS), default=None)
    parser.add_argument("--config", choices=list(VALID_CONFIGS), default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--encoder_id", type=str, default=None)
    parser.add_argument("--split", choices=list(VALID_SPLITS), default="test")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "out" / "error_analysis"),
    )
    parser.add_argument(
        "--compare_predictions",
        type=str,
        default=None,
        help="Comma-separated list of prediction TSV files to compare.",
    )
    return parser.parse_args()


def infer_from_checkpoint_name(checkpoint_path: Path) -> dict[str, str | int]:
    pattern = re.compile(
        r"^(subtask_[12])_(en|es)_(pred_flm_add|pred_flm|pred|flm)_(baseline|multilingual)"
        r"_seed(\d+)_epoch(\d+)\.pt$"
    )
    match = pattern.match(checkpoint_path.name)
    if match is None:
        return {}
    subtask, lang, variant, config, seed, _epoch = match.groups()
    return {
        "subtask": subtask,
        "lang": lang,
        "model_variant": variant,
        "config": config,
        "seed": int(seed),
    }


def resolve_config(args: argparse.Namespace) -> RunConfig:
    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    inferred = infer_from_checkpoint_name(checkpoint_path)
    subtask = args.subtask or inferred.get("subtask")
    lang = args.lang or inferred.get("lang")
    model_variant = args.model_variant or inferred.get("model_variant")
    config = args.config or inferred.get("config")
    seed = args.seed if args.seed is not None else inferred.get("seed")

    missing: list[str] = []
    if subtask is None:
        missing.append("--subtask")
    if lang is None:
        missing.append("--lang")
    if model_variant is None:
        missing.append("--model_variant")
    if config is None:
        missing.append("--config")
    if seed is None:
        missing.append("--seed")

    if missing:
        raise ValueError(
            "Could not infer all parameters from checkpoint name. "
            f"Please provide: {', '.join(missing)}"
        )

    if args.encoder_id is not None:
        encoder_id = args.encoder_id
    elif config == "multilingual":
        encoder_id = ENCODER_MULTILINGUAL
    else:
        encoder_id = ENCODER_MAP_BASELINE[lang]

    return RunConfig(
        subtask=subtask,
        lang=lang,
        model_variant=model_variant,
        config=config,
        seed=int(seed),
        encoder_id=encoder_id,
        checkpoint_path=checkpoint_path,
        split=args.split,
    )


def get_label_names(subtask: str) -> list[str]:
    label_map = SUBTASK_1_LABELS if subtask == "subtask_1" else SUBTASK_2_LABELS
    return [label_map[i] for i in sorted(label_map)]


def load_split_dataframe(
    train_dir: Path,
    test_dir: Path,
    subtask: str,
    split: str,
    seed: int,
) -> pd.DataFrame:
    if split == "test":
        test_df = pd.read_csv(test_dir / "test.tsv", sep="\t")
        test_df["text"] = test_df["text"].fillna("").astype(str)
        if subtask == "subtask_1":
            label_map = {"human": 0, "generated": 1}
            test_df["y"] = test_df["label"].str.lower().map(label_map).astype(int)
        else:
            test_df["y"] = (
                test_df["label"]
                .astype(str)
                .str.strip()
                .str.upper()
                .apply(lambda x: ord(x[0]) - ord("A"))
            ).astype(int)
        return test_df.reset_index(drop=True)

    train_df = pd.read_csv(train_dir / "train.tsv", sep="\t")
    if subtask == "subtask_1":
        label_map = {"human": 0, "generated": 1}
        train_df["y"] = train_df["label"].str.lower().map(label_map).astype(int)
    else:
        train_df["y"] = (
            train_df["label"].astype(str).str.strip().str.upper().apply(lambda x: ord(x[0]) - ord("A"))
        ).astype(int)

    shuffled = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    split_idx = int(len(shuffled) * 0.8)
    split_df = shuffled.iloc[:split_idx].copy() if split == "train" else shuffled.iloc[split_idx:].copy()
    return split_df.reset_index(drop=True)


def build_model(config: RunConfig, seq_feature_len: int, local_device: torch.device) -> torch.nn.Module:
    from models.hybrid import FLMRoBERTa, HybridBiLSTMRoBERTa, PredLSTM

    if config.model_variant == "pred":
        return PredLSTM(
            seq_feature_len=seq_feature_len,
            task=config.subtask,
            local_device=local_device,
        )
    if config.model_variant == "flm":
        return FLMRoBERTa(
            task=config.subtask,
            local_device=local_device,
            roberta_variant=config.encoder_id,
            baseline_compat_no_freeze=False,
        )
    return HybridBiLSTMRoBERTa(
        seq_feature_len=seq_feature_len,
        task=config.subtask,
        local_device=local_device,
        roberta_variant=config.encoder_id,
        disable_sequence=False,
    )


def build_dataloader_for_split(config: RunConfig, device: torch.device) -> tuple[DataLoader, pd.DataFrame]:
    from utils.data_utils import load_train_dev_test
    from utils.feature_utils import compute_all_features

    local_device = torch.device("cpu")
    data_dir = PROJECT_ROOT / "data" / "data"
    train_dir = data_dir / "train" / config.subtask / config.lang
    test_dir = data_dir / "test" / config.subtask / config.lang
    features_dir = PROJECT_ROOT / "data" / "features"

    (
        train_texts,
        dev_texts,
        test_texts,
        train_y,
        dev_y,
        test_y,
        train_idx,
        dev_idx,
    ) = load_train_dev_test(
        train_dir=train_dir,
        test_dir=test_dir,
        subtask=config.subtask,
        seed=config.seed,
        code_split=False,
        use_fold=0,
    )

    train_x, dev_x, test_x, fixed_len = compute_all_features(
        train_texts=train_texts,
        dev_texts=dev_texts,
        test_texts=test_texts,
        train_idx=train_idx,
        dev_idx=dev_idx,
        subtask=config.subtask,
        lang=config.lang,
        device=device,
        local_device=local_device,
        model_variant=config.model_variant,
        features_dir=features_dir,
        multilingual=config.config == "multilingual",
    )

    tokenizer = AutoTokenizer.from_pretrained(config.encoder_id, use_fast=True)

    def encode(texts: list[str]) -> dict[str, torch.Tensor]:
        return tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=fixed_len,
            return_tensors="pt",
        )

    if config.split == "train":
        split_texts, split_y, split_x = train_texts, train_y, train_x
    elif config.split == "dev":
        split_texts, split_y, split_x = dev_texts, dev_y, dev_x
    else:
        split_texts, split_y, split_x = test_texts, test_y, test_x

    enc = encode(split_texts)
    dataset = TensorDataset(
        torch.tensor(split_x).float(),
        enc["input_ids"],
        enc["attention_mask"],
        torch.tensor(split_y).long(),
    )
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=16,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    split_df = load_split_dataframe(
        train_dir=train_dir,
        test_dir=test_dir,
        subtask=config.subtask,
        split=config.split,
        seed=config.seed,
    )
    return dataloader, split_df


def run_inference(
    dataloader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []

    with torch.no_grad():
        for batch in dataloader:
            batch = [tensor.to(device, non_blocking=True) for tensor in batch]
            features, input_ids, attention_mask, labels = batch
            output = model(features, input_ids, attention_mask)
            probs = torch.exp(output).cpu().numpy()
            preds = probs.argmax(axis=1).astype(int)
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy().astype(int))
            all_probs.append(probs)

    return np.concatenate(all_preds), np.concatenate(all_labels), np.concatenate(all_probs)


def build_predictions_table(
    split_df: pd.DataFrame,
    predictions: np.ndarray,
    true_labels: np.ndarray,
    probabilities: np.ndarray,
    label_names: list[str],
) -> pd.DataFrame:
    if len(split_df) != len(predictions):
        raise ValueError(
            f"Split dataframe length {len(split_df)} does not match predictions length {len(predictions)}."
        )

    table = split_df.copy()
    table["true_label_id"] = true_labels
    table["pred_label_id"] = predictions
    table["true_label"] = [label_names[i] for i in true_labels]
    table["pred_label"] = [label_names[i] for i in predictions]
    table["pred_confidence"] = probabilities.max(axis=1)
    table["correct"] = table["true_label_id"] == table["pred_label_id"]
    table["text_length_chars"] = table["text"].astype(str).str.len()
    table["text_length_words"] = table["text"].astype(str).str.split().str.len()

    for class_idx, label in enumerate(label_names):
        safe_label = re.sub(r"[^a-zA-Z0-9_]+", "_", label)
        table[f"prob_{safe_label}"] = probabilities[:, class_idx]

    return table


def save_confusion_artifacts(
    predictions_table: pd.DataFrame,
    label_names: list[str],
    output_prefix: Path,
) -> None:
    matrix = confusion_matrix(
        predictions_table["true_label_id"],
        predictions_table["pred_label_id"],
        labels=list(range(len(label_names))),
    )
    matrix_df = pd.DataFrame(matrix, index=label_names, columns=label_names)
    matrix_df.to_csv(output_prefix.with_name(f"{output_prefix.name}_confusion_matrix.csv"))

    plt.figure(figsize=(7, 6))
    sns.heatmap(matrix_df, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(output_prefix.with_name(f"{output_prefix.name}_confusion_matrix.png"), dpi=180)
    plt.close()


def save_confidence_histogram(predictions_table: pd.DataFrame, output_prefix: Path) -> None:
    plot_df = predictions_table[["pred_confidence", "correct"]].copy()
    plot_df["status"] = np.where(plot_df["correct"], "correct", "error")

    plt.figure(figsize=(8, 5))
    sns.histplot(
        data=plot_df,
        x="pred_confidence",
        hue="status",
        bins=20,
        stat="density",
        common_norm=False,
        element="step",
    )
    plt.title("Prediction Confidence Distribution")
    plt.xlabel("Predicted class confidence")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(output_prefix.with_name(f"{output_prefix.name}_confidence_histogram.png"), dpi=180)
    plt.close()


def save_length_bucket_analysis(predictions_table: pd.DataFrame, output_prefix: Path) -> None:
    lengths = predictions_table["text_length_words"]
    if lengths.nunique() <= 1:
        bucketed = pd.Series(["all"] * len(lengths), index=predictions_table.index)
    else:
        bucketed = pd.qcut(lengths, q=min(5, int(lengths.nunique())), duplicates="drop")

    temp = predictions_table.copy()
    temp["length_bucket"] = bucketed.astype(str)

    grouped = temp.groupby("length_bucket", dropna=False).agg(
        n_samples=("correct", "size"),
        accuracy=("correct", "mean"),
        mean_confidence=("pred_confidence", "mean"),
        mean_word_length=("text_length_words", "mean"),
    )
    grouped["error_rate"] = 1.0 - grouped["accuracy"]
    grouped = grouped.sort_values("mean_word_length")
    grouped.to_csv(output_prefix.with_name(f"{output_prefix.name}_length_buckets.tsv"), sep="\t")


def save_reports(predictions_table: pd.DataFrame, label_names: list[str], output_prefix: Path) -> None:
    report = classification_report(
        predictions_table["true_label_id"],
        predictions_table["pred_label_id"],
        labels=list(range(len(label_names))),
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    output_prefix.with_name(f"{output_prefix.name}_classification_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    predictions_table.to_csv(output_prefix.with_name(f"{output_prefix.name}_predictions.tsv"), sep="\t", index=False)

    errors = predictions_table.loc[~predictions_table["correct"]].copy()
    errors.sort_values("pred_confidence", ascending=False).to_csv(
        output_prefix.with_name(f"{output_prefix.name}_misclassified.tsv"),
        sep="\t",
        index=False,
    )


def compare_prediction_files(compare_arg: str, output_dir: Path) -> None:
    files = [Path(raw.strip()).expanduser().resolve() for raw in compare_arg.split(",") if raw.strip()]
    if len(files) < 2:
        raise ValueError("--compare_predictions requires at least two files.")

    run_tables: list[tuple[str, pd.DataFrame]] = []
    for path in files:
        if not path.exists():
            raise FileNotFoundError(f"Prediction file not found: {path}")
        table = pd.read_csv(path, sep="\t")
        for needed_col in ("true_label_id", "pred_label_id", "correct"):
            if needed_col not in table.columns:
                raise ValueError(f"Missing column '{needed_col}' in file: {path}")
        run_tables.append((path.stem, table))

    base_len = len(run_tables[0][1])
    for stem, table in run_tables:
        if len(table) != base_len:
            raise ValueError(f"Prediction file length mismatch for {stem}.")

    summary_rows: list[dict[str, float | int | str]] = []
    disagreement_rows: list[dict[str, float | int | str]] = []

    for stem, table in run_tables:
        accuracy = float(table["correct"].mean())
        summary_rows.append(
            {"run": stem, "n_samples": len(table), "accuracy": accuracy, "error_rate": 1.0 - accuracy}
        )

    for i, (left_stem, left_table) in enumerate(run_tables):
        for j, (right_stem, right_table) in enumerate(run_tables):
            if i >= j:
                continue
            disagreement_rows.append(
                {
                    "run_left": left_stem,
                    "run_right": right_stem,
                    "prediction_disagreement_rate": float(
                        (left_table["pred_label_id"] != right_table["pred_label_id"]).mean()
                    ),
                    "both_wrong_rate": float((~left_table["correct"] & ~right_table["correct"]).mean()),
                    "one_wrong_one_right_rate": float((left_table["correct"] != right_table["correct"]).mean()),
                }
            )

    pd.DataFrame(summary_rows).to_csv(output_dir / "cross_run_summary.tsv", sep="\t", index=False)
    pd.DataFrame(disagreement_rows).to_csv(output_dir / "cross_run_disagreement.tsv", sep="\t", index=False)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.compare_predictions:
        compare_prediction_files(args.compare_predictions, output_dir)

    run_config = resolve_config(args)
    label_names = get_label_names(run_config.subtask)
    run_name = (
        f"{run_config.subtask}_{run_config.lang}_{run_config.model_variant}_{run_config.config}"
        f"_seed{run_config.seed}_{run_config.split}"
    )
    output_prefix = output_dir / run_name

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataloader, split_df = build_dataloader_for_split(run_config, device)

    model = build_model(
        config=run_config,
        seq_feature_len=int(dataloader.dataset.tensors[0].shape[2]),
        local_device=torch.device("cpu"),
    ).to(device)
    state = torch.load(run_config.checkpoint_path, map_location=device)
    state_dict = state["model_state"] if isinstance(state, dict) and "model_state" in state else state
    model.load_state_dict(state_dict, strict=True)

    predictions, true_labels, probabilities = run_inference(dataloader, model, device)
    predictions_table = build_predictions_table(
        split_df=split_df,
        predictions=predictions,
        true_labels=true_labels,
        probabilities=probabilities,
        label_names=label_names,
    )

    save_reports(predictions_table, label_names, output_prefix)
    save_confusion_artifacts(predictions_table, label_names, output_prefix)
    save_confidence_histogram(predictions_table, output_prefix)
    save_length_bucket_analysis(predictions_table, output_prefix)

    accuracy = float(predictions_table["correct"].mean())
    macro_report = classification_report(
        predictions_table["true_label_id"],
        predictions_table["pred_label_id"],
        output_dict=True,
        zero_division=0,
    )
    macro_f1 = float(macro_report["macro avg"]["f1-score"])
    n_errors = int((~predictions_table["correct"]).sum())

    print("=" * 80)
    print("ERROR ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Run name: {run_name}")
    print(f"Samples: {len(predictions_table)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Misclassified: {n_errors}")
    print(f"Artifacts directory: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()