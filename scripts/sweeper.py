"""Runs an in-process sweep of training variants/encoders/seeds and appends results to a TSV summary."""

from __future__ import annotations

import argparse
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from utils.constants import (
    BATCH_SIZE,
    CHECKPOINTS_DIR,
    DEVICE,
    EPOCHS,
    FREEZE_EPOCHS,
    LOCAL_DEVICE,
    LOG_DIR,
    RESULTS_DIR,
    is_baseline_encoder,
)
from utils.data_utils import load_train_dev_test
from utils.feature_utils import compute_all_features
from utils.logging_utils import Tee
from utils.training_pipeline import (
    build_model,
    create_dataloaders,
    tokenize_splits,
    train_and_evaluate,
)


def _make_tokenizer(encoder_id: str):
    if is_baseline_encoder(encoder_id):
        return AutoTokenizer.from_pretrained(encoder_id, use_fast=False)
    if "deberta" in encoder_id.lower():
        from transformers import DebertaV2Tokenizer

        return DebertaV2Tokenizer.from_pretrained(encoder_id)
    return AutoTokenizer.from_pretrained(encoder_id, use_fast=True)


def run_one(
    *,
    subtask: str,
    lang: str,
    model_variant: str,
    encoder_id: str,
    seed: int,
    device: torch.device,
    local_device: torch.device,
    data_dir: Path,
    features_dir: Path,
    out_dir: Path,
    log_dir: Path,
    freeze_epochs: int = FREEZE_EPOCHS,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 4,
    code_split: bool = False,
    use_fold: int = 0,
) -> dict[str, str | int | float]:
    assert model_variant in {"pred", "flm", "pred_flm", "pred_flm_add"}

    enc_tag = encoder_id.replace("/", "_")
    run_id = f"{subtask}_{lang}_{model_variant}_{enc_tag}_seed{seed}"
    log_file = log_dir / f"EXP_{run_id}.log"
    sys.stdout = Tee(log_file)

    print("=" * 90)
    print(f"RUN: {run_id}")
    print(f"timestamp={datetime.now().isoformat(timespec='seconds')}")
    print(f"device={device} | epochs={epochs} | freeze_epochs={freeze_epochs}")
    print("=" * 90)

    random.seed(seed)
    np.random.seed(0)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"[SEED] random/torch seed={seed} | numpy seed=0 (forced)")

    train_dir = data_dir / "train" / subtask / lang
    test_dir = data_dir / "test" / subtask / lang

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
        seed=seed,
        code_split=code_split,
        use_fold=use_fold,
    )

    train_X, dev_X, test_X, fixed_len = compute_all_features(
        train_texts=train_texts,
        dev_texts=dev_texts,
        test_texts=test_texts,
        train_idx=train_idx,
        dev_idx=dev_idx,
        subtask=subtask,
        lang=lang,
        device=device,
        local_device=local_device,
        model_variant=model_variant,
        features_dir=features_dir,
    )

    print(
        f"[INFO] Feature shapes: train={train_X.shape} dev={dev_X.shape} "
        f"test={test_X.shape} fixed_len={fixed_len}"
    )

    tokenizer = _make_tokenizer(encoder_id)
    print(
        "[TOK] Baseline encoder -> slow tokenizer"
        if is_baseline_encoder(encoder_id)
        else "[TOK] Non-baseline encoder -> fast tokenizer"
    )

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
        batch_size=batch_size,
        num_workers=num_workers,
    )

    baseline_mode = is_baseline_encoder(encoder_id)

    model = build_model(
        model_variant=model_variant,
        subtask=subtask,
        encoder_id=encoder_id,
        seq_feature_len=train_X.shape[2],
        device=device,
        local_device=local_device,
        baseline_compat_no_freeze=baseline_mode,
    )

    result = train_and_evaluate(
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=test_loader,
        model=model,
        model_variant=model_variant,
        device=device,
        out_dir=out_dir,
        checkpoint_prefix=run_id,
        epochs=epochs,
        freeze_epochs=freeze_epochs,
    )

    print(
        f"\nFINAL RESULT: {run_id} | best_epoch={result.best_epoch} "
        f"| dev={result.dev_f1:.4f} | test={result.test_f1:.4f}"
    )

    return {
        "subtask": subtask,
        "lang": lang,
        "variant": model_variant,
        "encoder_id": encoder_id,
        "seed": seed,
        "best_epoch": result.best_epoch,
        "dev_f1": result.dev_f1,
        "test_f1": result.test_f1,
        "run_id": run_id,
        "log_file": str(log_file),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subtask", choices=["subtask_1", "subtask_2", "all"], default="all")
    parser.add_argument("--seeds", type=int, nargs="+", default=[10])
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--freeze_epochs", type=int, default=FREEZE_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    data_dir = Path(PROJECT_ROOT) / "data" / "data"
    features_dir = Path(PROJECT_ROOT) / "data" / "features"

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    baseline = {
        "en": "roberta-base",
        "es": "bertin-project/bertin-roberta-base-spanish",
    }

    encoders_en = [
        "microsoft/deberta-v3-base",
        "FacebookAI/xlm-roberta-base",
        "answerdotai/ModernBERT-base",
    ]
    encoders_es = [
        "flax-community/bertin-roberta-large-spanish",
        "FacebookAI/xlm-roberta-base",
        "microsoft/mdeberta-v3-base",
    ]

    sweep_variants = ["flm", "pred_flm", "pred_flm_add"]
    subtasks = ["subtask_1", "subtask_2"] if args.subtask == "all" else [args.subtask]

    summary_path = RESULTS_DIR / "experiments_summary.tsv"

    if not summary_path.exists():
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(
                "subtask\tlang\tvariant\tencoder_id\tseed\tbest_epoch\tdev_f1\ttest_f1\trun_id\tlog_file\n"
            )

    def append_result_row(res: dict) -> None:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(
                f"{res['subtask']}\t{res['lang']}\t{res['variant']}\t{res['encoder_id']}\t"
                f"{res['seed']}\t{res['best_epoch']}\t{res['dev_f1']:.6f}\t{res['test_f1']:.6f}\t"
                f"{res['run_id']}\t{res['log_file']}\n"
            )

    all_runs: list[dict] = []

    for seed in args.seeds:
        for subtask in subtasks:
            for lang in ["en", "es"]:
                res_pred = run_one(
                    subtask=subtask,
                    lang=lang,
                    model_variant="pred",
                    encoder_id=baseline[lang],
                    seed=seed,
                    device=DEVICE,
                    local_device=LOCAL_DEVICE,
                    data_dir=data_dir,
                    features_dir=features_dir,
                    out_dir=CHECKPOINTS_DIR,
                    log_dir=LOG_DIR,
                    freeze_epochs=args.freeze_epochs,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                )
                all_runs.append(res_pred)
                append_result_row(res_pred)

                encoder_list = [baseline[lang]] + (encoders_en if lang == "en" else encoders_es)

                for encoder_id in encoder_list:
                    for variant in sweep_variants:
                        res = run_one(
                            subtask=subtask,
                            lang=lang,
                            model_variant=variant,
                            encoder_id=encoder_id,
                            seed=seed,
                            device=DEVICE,
                            local_device=LOCAL_DEVICE,
                            data_dir=data_dir,
                            features_dir=features_dir,
                            out_dir=CHECKPOINTS_DIR,
                            log_dir=LOG_DIR,
                            freeze_epochs=args.freeze_epochs,
                            epochs=args.epochs,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                        )
                        all_runs.append(res)
                        append_result_row(res)

    sys.stdout = sys.__stdout__
    print(f"Done. Appended {len(all_runs)} runs to {summary_path}")


if __name__ == "__main__":
    main()
