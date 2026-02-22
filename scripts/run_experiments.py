"""Runs multilingual probabilistic-feature experiments across presets/variants and saves results."""

from __future__ import annotations

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import csv
import json
import random
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

np.seterr(invalid="ignore")
import torch
from transformers import AutoTokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from feature_extraction.grammar_features import GrammarFeatures, WordFrequency
from feature_extraction.probabilistic_features import ConfigurableProbFeatures
from utils.constants import (
    BATCH_SIZE,
    CHECKPOINTS_DIR,
    DEVICE,
    EPOCHS,
    FEATURES_DIR,
    FREEZE_EPOCHS,
    LOCAL_DEVICE,
    LOG_DIR,
    RESULTS_DIR,
)
from utils.data_utils import load_train_dev_test
from utils.logging_utils import Tee
from utils.training_pipeline import (
    build_model,
    create_dataloaders,
    tokenize_splits,
    train_and_evaluate,
)

SEED = 10
CODE_SPLIT = False
USE_FOLD = 0

PROB_FEATURE_BATCH_SIZE = 8
FIXED_LEN = 128

EXP_OUT_DIR = RESULTS_DIR / "experiments"
EXP_LOG_DIR = LOG_DIR / "experiments"


@dataclass(frozen=True)
class ProbModelSpec:
    model_id: str

    def __str__(self) -> str:
        return self.model_id.split("/")[-1]


@dataclass
class ExperimentConfig:
    name: str
    prob_models: list[ProbModelSpec] = field(default_factory=list)
    prob_models_en: list[ProbModelSpec] | None = None
    prob_models_es: list[ProbModelSpec] | None = None
    description: str = ""

    def get_prob_models(self, lang: str) -> list[ProbModelSpec]:
        if lang == "en" and self.prob_models_en is not None:
            return self.prob_models_en
        if lang == "es" and self.prob_models_es is not None:
            return self.prob_models_es
        return self.prob_models

    def get_encoder(self, subtask: str, lang: str) -> str:
        return "microsoft/mdeberta-v3-base"


MULTILINGUAL_XGLM = [
    ProbModelSpec("facebook/xglm-564M"),
    ProbModelSpec("facebook/xglm-1.7B"),
    ProbModelSpec("Qwen/Qwen2.5-1.5B"),
    ProbModelSpec("bigscience/bloom-1b1"),
]

MULTILINGUAL_MGPT = [
    ProbModelSpec("ai-forever/mGPT"),
    ProbModelSpec("meta-llama/Llama-3.2-3B"),
    ProbModelSpec("Qwen/Qwen2.5-1.5B"),
    ProbModelSpec("bigscience/bloom-1b7"),
]

MULTILINGUAL_LARGE = [
    ProbModelSpec("Qwen/Qwen2.5-3B"),
    ProbModelSpec("meta-llama/Llama-3.2-3B"),
    ProbModelSpec("facebook/xglm-2.9B"),
    ProbModelSpec("bigscience/bloom-1b7"),
]

EXPERIMENT_PRESETS: dict[str, ExperimentConfig] = {
    "multilingual_xglm": ExperimentConfig(
        name="multilingual_xglm",
        prob_models=MULTILINGUAL_XGLM,
        description="XGLM-focused cross-lingual (~5.8B): XGLM-564M, XGLM-1.7B, Qwen-1.5B, BLOOM-1.1B",
    ),
    "multilingual_mgpt": ExperimentConfig(
        name="multilingual_mgpt",
        prob_models=MULTILINGUAL_MGPT,
        description="mGPT-focused multilingual (~7B): mGPT, Llama-3B, Qwen-1.5B, BLOOM-1.7B",
    ),
    "multilingual_large": ExperimentConfig(
        name="multilingual_large",
        prob_models=MULTILINGUAL_LARGE,
        description="Large multilingual (~10.5B): Qwen-3B, Llama-3B, XGLM-2.9B, BLOOM-1.7B",
    ),
}


def get_all_experiment_names() -> list[str]:
    return list(EXPERIMENT_PRESETS.keys())


def get_experiment(name: str) -> ExperimentConfig:
    if name not in EXPERIMENT_PRESETS:
        raise ValueError(
            f"Unknown experiment '{name}'. Available: {', '.join(EXPERIMENT_PRESETS.keys())}"
        )
    return EXPERIMENT_PRESETS[name]


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(0)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_prob_features(
    train_texts: list[str],
    dev_texts: list[str],
    test_texts: list[str],
    model_ids: list[str],
    batch_size: int = PROB_FEATURE_BATCH_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prob_extractor = ConfigurableProbFeatures(
        device=DEVICE,
        local_device=LOCAL_DEVICE,
        model_ids=model_ids,
        disabled=False,
        batch_size=batch_size,
    )

    train_prob = prob_extractor.word_features(train_texts)
    dev_prob = prob_extractor.word_features(dev_texts)
    test_prob = prob_extractor.word_features(test_texts)

    return train_prob, dev_prob, test_prob


def compute_features_with_config(
    train_texts: list[str],
    dev_texts: list[str],
    test_texts: list[str],
    train_idx: np.ndarray,
    dev_idx: np.ndarray,
    config: ExperimentConfig,
    model_variant: str,
    subtask: str,
    lang: str,
    train_prob: np.ndarray | None = None,
    dev_prob: np.ndarray | None = None,
    test_prob: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    use_additional = model_variant == "pred_flm_add"
    prob_disabled = model_variant == "flm"

    prob_models = config.get_prob_models(lang)
    model_ids = [spec.model_id for spec in prob_models]

    if train_prob is None or dev_prob is None or test_prob is None:
        print(f"\n[FEATURES] Probabilistic models ({len(model_ids)}): {model_ids}")
        train_prob, dev_prob, test_prob = compute_prob_features(
            train_texts, dev_texts, test_texts, model_ids
        )
    else:
        print("\n[FEATURES] Reusing precomputed probabilistic features")
        print(f"[FEATURES] Probabilistic models ({len(model_ids)}): {model_ids}")

    if prob_disabled:
        train_prob = np.zeros_like(train_prob)
        dev_prob = np.zeros_like(dev_prob)
        test_prob = np.zeros_like(test_prob)

    if use_additional:
        print("\n[FEATURES] Computing frequency features...")
        freq_extractor = WordFrequency(DEVICE, LOCAL_DEVICE, lang)
        train_freq = np.array(freq_extractor.word_features(train_texts))
        dev_freq = np.array(freq_extractor.word_features(dev_texts))
        test_freq = np.array(freq_extractor.word_features(test_texts))

        print("[FEATURES] Computing grammar features...")
        grammar_train_path = FEATURES_DIR / f"train_{subtask}_{lang}_grammar.npy"
        grammar_test_path = FEATURES_DIR / f"test_{subtask}_{lang}_grammar.npy"

        if grammar_train_path.exists():
            train_gram_full = np.load(grammar_train_path)
            train_gram = train_gram_full[train_idx]
            dev_gram = train_gram_full[dev_idx]
        else:
            grammar_extractor = GrammarFeatures(DEVICE, LOCAL_DEVICE, lang)
            train_gram = np.array(grammar_extractor.word_features(train_texts))
            dev_gram = np.array(grammar_extractor.word_features(dev_texts))

        if grammar_test_path.exists():
            test_gram = np.load(grammar_test_path)
        else:
            grammar_extractor = GrammarFeatures(DEVICE, LOCAL_DEVICE, lang)
            test_gram = np.array(grammar_extractor.word_features(test_texts))

        train_X = np.concatenate([train_prob, train_freq, train_gram], axis=2)
        dev_X = np.concatenate([dev_prob, dev_freq, dev_gram], axis=2)
        test_X = np.concatenate([test_prob, test_freq, test_gram], axis=2)
    else:
        train_X = train_prob
        dev_X = dev_prob
        test_X = test_prob

    print(f"\n[FEATURES] Final feature tensor shape: {train_X.shape}")
    print(f"  - Samples: {train_X.shape[0]}")
    print(f"  - Sequence length: {train_X.shape[1]}")
    print(f"  - Features per token: {train_X.shape[2]}")

    return train_X, dev_X, test_X, FIXED_LEN


@dataclass
class ExperimentResult:
    experiment_name: str
    model_variant: str
    subtask: str
    lang: str
    prob_models: list[str]
    encoder: str
    best_epoch: int
    dev_f1: float
    test_f1: float
    n_prob_features: int
    n_total_features: int
    training_time_sec: float


def run_single_experiment(
    config: ExperimentConfig,
    model_variant: str,
    subtask: str,
    lang: str,
    train_prob: np.ndarray | None = None,
    dev_prob: np.ndarray | None = None,
    test_prob: np.ndarray | None = None,
    train_texts: list[str] | None = None,
    dev_texts: list[str] | None = None,
    test_texts: list[str] | None = None,
    train_Y: np.ndarray | None = None,
    dev_Y: np.ndarray | None = None,
    test_Y: np.ndarray | None = None,
    train_idx: np.ndarray | None = None,
    dev_idx: np.ndarray | None = None,
    feature_time_sec: float = 0.0,
) -> ExperimentResult | None:
    start_time = datetime.now()

    print("\n" + "=" * 80)
    print(f"EXPERIMENT: {config.name} | variant={model_variant} | {subtask}/{lang}")
    print(f"Description: {config.description}")
    print("=" * 80)

    set_seeds(SEED)

    if train_texts is None or dev_texts is None or test_texts is None:
        from utils.constants import DATA_DIR

        train_dir = DATA_DIR / "train" / subtask / lang
        test_dir = DATA_DIR / "test" / subtask / lang

        try:
            (
                train_texts,
                dev_texts,
                test_texts,
                train_Y,
                dev_Y,
                test_Y,
                train_idx,
                dev_idx,
            ) = load_train_dev_test(
                train_dir=train_dir,
                test_dir=test_dir,
                subtask=subtask,
                seed=SEED,
                code_split=CODE_SPLIT,
                use_fold=USE_FOLD,
            )
        except FileNotFoundError as e:
            print(f"[SKIP] Data not found: {e}")
            return None
    print(f"[DATA] Train: {len(train_texts)}, Dev: {len(dev_texts)}, Test: {len(test_texts)}")
    train_X, dev_X, test_X, fixed_len = compute_features_with_config(
        train_texts=train_texts,
        dev_texts=dev_texts,
        test_texts=test_texts,
        train_idx=train_idx,
        dev_idx=dev_idx,
        config=config,
        model_variant=model_variant,
        subtask=subtask,
        lang=lang,
        train_prob=train_prob,
        dev_prob=dev_prob,
        test_prob=test_prob,
    )

    encoder_id = config.get_encoder(subtask, lang)

    print(f"\n[STEP] Tokenizing with encoder: {encoder_id}")

    if "deberta" in encoder_id.lower():
        from transformers import DebertaV2Tokenizer

        tokenizer = DebertaV2Tokenizer.from_pretrained(encoder_id)
    elif "modernbert" in encoder_id.lower():
        tokenizer = AutoTokenizer.from_pretrained(encoder_id)
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(encoder_id)
        except (TypeError, ValueError):
            print(f"[WARN] Fast tokenizer failed for {encoder_id}, using slow tokenizer")
            tokenizer = AutoTokenizer.from_pretrained(encoder_id, use_fast=False)

    train_enc, dev_enc, test_enc = tokenize_splits(
        train_texts, dev_texts, test_texts, tokenizer, fixed_len,
    )

    train_loader, dev_loader, test_loader = create_dataloaders(
        train_X=train_X,
        dev_X=dev_X,
        test_X=test_X,
        train_enc=train_enc,
        dev_enc=dev_enc,
        test_enc=test_enc,
        train_Y=train_Y,
        dev_Y=dev_Y,
        test_Y=test_Y,
        batch_size=BATCH_SIZE,
        num_workers=0,
    )

    print(f"\n[STEP] Initializing model: {model_variant}")
    model = build_model(
        model_variant=model_variant,
        subtask=subtask,
        encoder_id=encoder_id,
        seq_feature_len=train_X.shape[2],
        device=DEVICE,
        local_device=LOCAL_DEVICE,
    )

    checkpoint_prefix = f"{config.name}_{model_variant}_{subtask}_{lang}"

    result = train_and_evaluate(
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=test_loader,
        model=model,
        model_variant=model_variant,
        device=DEVICE,
        out_dir=CHECKPOINTS_DIR / "experiments",
        checkpoint_prefix=checkpoint_prefix,
        epochs=EPOCHS,
        freeze_epochs=FREEZE_EPOCHS,
        cleanup_non_best=True,
    )

    training_elapsed = (datetime.now() - start_time).total_seconds()
    total_elapsed = training_elapsed + feature_time_sec

    prob_models = [spec.model_id for spec in config.get_prob_models(lang)]

    exp_result = ExperimentResult(
        experiment_name=config.name,
        model_variant=model_variant,
        subtask=subtask,
        lang=lang,
        prob_models=prob_models,
        encoder=encoder_id,
        best_epoch=result.best_epoch,
        dev_f1=result.dev_f1,
        test_f1=result.test_f1,
        n_prob_features=len(prob_models) * 3,
        n_total_features=train_X.shape[2],
        training_time_sec=total_elapsed,
    )

    print(
        f"\n[RESULT] dev={result.dev_f1:.4f} | test={result.test_f1:.4f} | "
        f"time={total_elapsed:.1f}s (train={training_elapsed:.1f}s + features={feature_time_sec:.1f}s)"
    )

    return exp_result


def save_results(results: list[ExperimentResult], output_path: Path) -> None:
    EXP_OUT_DIR.mkdir(parents=True, exist_ok=True)

    tsv_path = output_path.with_suffix(".tsv")
    csv_path = output_path.with_suffix(".csv")
    json_path = output_path.with_suffix(".json")

    headers = [
        "experiment",
        "variant",
        "subtask",
        "lang",
        "prob_models",
        "encoder",
        "best_epoch",
        "dev_f1",
        "test_f1",
        "n_features",
        "time_sec",
    ]

    with open(tsv_path, "w", encoding="utf-8", newline="") as f:
        f.write("\t".join(headers) + "\n")
        for r in results:
            prob_str = ",".join(m.split("/")[-1] for m in r.prob_models)
            enc_str = r.encoder.split("/")[-1]
            values = [
                r.experiment_name,
                r.model_variant,
                r.subtask,
                r.lang,
                prob_str,
                enc_str,
                str(r.best_epoch),
                f"{r.dev_f1:.4f}",
                f"{r.test_f1:.4f}",
                str(r.n_total_features),
                f"{r.training_time_sec:.1f}",
            ]
            f.write("\t".join(values) + "\n")

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for r in results:
            prob_str = ";".join(m.split("/")[-1] for m in r.prob_models)
            enc_str = r.encoder.split("/")[-1]
            writer.writerow(
                [
                    r.experiment_name,
                    r.model_variant,
                    r.subtask,
                    r.lang,
                    prob_str,
                    enc_str,
                    r.best_epoch,
                    f"{r.dev_f1:.4f}",
                    f"{r.test_f1:.4f}",
                    r.n_total_features,
                    f"{r.training_time_sec:.1f}",
                ]
            )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    print(f"\n[SAVED] {tsv_path}")
    print(f"[SAVED] {csv_path}")
    print(f"[SAVED] {json_path}")


def print_comparison_table(results: list[ExperimentResult]) -> None:
    print("\n" + "=" * 110)
    print("EXPERIMENT COMPARISON")
    print("=" * 110)

    print(
        f"\n{'Experiment':<22} {'Variant':<14} {'Task':<10} {'Lang':<5} "
        f"{'Encoder':<20} {'Dev F1':<10} {'Test F1':<10}"
    )
    print("-" * 110)

    for r in results:
        enc_short = r.encoder.split("/")[-1][:18]
        print(
            f"{r.experiment_name:<22} {r.model_variant:<14} {r.subtask:<10} {r.lang:<5} "
            f"{enc_short:<20} {r.dev_f1:<10.4f} {r.test_f1:<10.4f}"
        )


def main() -> None:
    """Orchestrate multi-experiment sweep."""
    parser = argparse.ArgumentParser(
        description="Run multilingual probabilistic model experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available experiment presets:
  {', '.join(get_all_experiment_names())}

Configurations:
  multilingual_xglm  - XGLM-564M, XGLM-1.7B, Qwen-1.5B, BLOOM-1.1B (~5.8B)
  multilingual_mgpt  - mGPT, Llama-3B, Qwen-1.5B, BLOOM-1.7B (~7B)
  multilingual_large - Qwen-3B, Llama-3B, XGLM-2.9B, BLOOM-1.7B (~10.5B)
        """,
    )

    parser.add_argument("--experiment", type=str, default="multilingual_xglm")
    parser.add_argument(
        "--variant",
        type=str,
        choices=["pred", "flm", "pred_flm", "pred_flm_add", "all"],
        default="pred_flm",
    )
    parser.add_argument(
        "--subtask",
        type=str,
        choices=["subtask_1", "subtask_2", "all"],
        default="subtask_1",
    )
    parser.add_argument("--lang", type=str, choices=["en", "es", "all"], default="en")
    parser.add_argument("--batch-size", type=int, default=PROB_FEATURE_BATCH_SIZE)

    args = parser.parse_args()
    batch_size = args.batch_size

    EXP_OUT_DIR.mkdir(parents=True, exist_ok=True)
    EXP_LOG_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = EXP_LOG_DIR / f"experiment_{timestamp}.log"
    tee = Tee(log_file)
    sys.stdout = tee

    print("=" * 80)
    print("MULTILINGUAL PROBABILISTIC MODEL EXPERIMENT PIPELINE")
    print("=" * 80)
    print(f"Timestamp: {timestamp}")
    print(f"Device: {DEVICE}")
    print(f"Arguments: {vars(args)}")
    print("=" * 80)

    config_names = (
        get_all_experiment_names()
        if args.experiment == "all"
        else [n.strip() for n in args.experiment.split(",")]
    )
    configs = [get_experiment(name) for name in config_names]

    variants = ["pred_flm", "pred_flm_add"] if args.variant == "all" else [args.variant]
    subtasks = ["subtask_1", "subtask_2"] if args.subtask == "all" else [args.subtask]
    languages = ["en", "es"] if args.lang == "all" else [args.lang]

    total_runs = len(configs) * len(variants) * len(subtasks) * len(languages)
    print(f"\nPlanned runs: {total_runs}")
    print(f"  Configs: {[c.name for c in configs]}")
    print(f"  Variants: {variants}")
    print(f"  Subtasks: {subtasks}")
    print(f"  Languages: {languages}")

    results: list[ExperimentResult] = []
    run_idx = 0

    from utils.constants import DATA_DIR

    for config in configs:
        for subtask in subtasks:
            for lang in languages:
                print(f"\n{'#' * 80}")
                print(f"# CONFIG GROUP: {config.name} | {subtask}/{lang}")
                print(f"# {config.description}")
                print(f"{'#' * 80}")

                train_dir = DATA_DIR / "train" / subtask / lang
                test_dir = DATA_DIR / "test" / subtask / lang

                try:
                    (
                        train_texts,
                        dev_texts,
                        test_texts,
                        train_Y,
                        dev_Y,
                        test_Y,
                        train_idx,
                        dev_idx,
                    ) = load_train_dev_test(
                        train_dir=train_dir,
                        test_dir=test_dir,
                        subtask=subtask,
                        seed=SEED,
                        code_split=CODE_SPLIT,
                        use_fold=USE_FOLD,
                    )
                except FileNotFoundError as e:
                    print(f"[SKIP] Data not found: {e}")
                    continue

                print(
                    f"[DATA] Train: {len(train_texts)}, Dev: {len(dev_texts)}, Test: {len(test_texts)}"
                )

                prob_models = config.get_prob_models(lang)
                model_ids = [spec.model_id for spec in prob_models]

                print(f"\n[FEATURES] Computing shared probabilistic features: {model_ids}")
                feature_start_time = datetime.now()
                train_prob, dev_prob, test_prob = compute_prob_features(
                    train_texts, dev_texts, test_texts, model_ids, batch_size
                )
                feature_time_sec = (datetime.now() - feature_start_time).total_seconds()
                print(
                    f"\n[FEATURES] Probabilistic feature computation took {feature_time_sec:.1f}s"
                )

                for variant in variants:
                    run_idx += 1
                    print(f"\n{'#' * 80}")
                    print(f"# RUN {run_idx}/{total_runs}")
                    print(f"# {config.name} | {variant} | {subtask}/{lang}")
                    print(f"{'#' * 80}")

                    result = run_single_experiment(
                        config,
                        variant,
                        subtask,
                        lang,
                        train_prob=train_prob,
                        dev_prob=dev_prob,
                        test_prob=test_prob,
                        train_texts=train_texts,
                        dev_texts=dev_texts,
                        test_texts=test_texts,
                        train_Y=train_Y,
                        dev_Y=dev_Y,
                        test_Y=test_Y,
                        train_idx=train_idx,
                        dev_idx=dev_idx,
                        feature_time_sec=feature_time_sec,
                    )

                    if result:
                        results.append(result)
                        save_results(
                            results, EXP_OUT_DIR / f"results_{timestamp}_partial"
                        )

    if results:
        print_comparison_table(results)
        save_results(results, EXP_OUT_DIR / f"results_{timestamp}")
        save_results(results, EXP_OUT_DIR / "results_latest")

    print("\n" + "=" * 80)
    print("EXPERIMENT PIPELINE COMPLETE")
    print("=" * 80)

    tee.close()


if __name__ == "__main__":
    main()
