"""Train a selected hybrid/Predictability/FLM model variant and log dev/test F1 with checkpoints."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from _bootstrap import configure_project_root

PROJECT_ROOT: str = str(configure_project_root(__file__, remove_shadowing_utils=False))

from utils.constants import (
    BATCH_SIZE,
    CHECKPOINTS_DIR,
    DEVICE,
    ENCODER_MAP_BASELINE,
    ENCODER_MULTILINGUAL,
    EPOCHS,
    FREEZE_EPOCHS,
    LOCAL_DEVICE,
    LOG_DIR,
    RESULTS_DIR,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subtask", choices=["subtask_1", "subtask_2"], default="subtask_2")
    parser.add_argument("--lang", choices=["en", "es"], default="en")
    parser.add_argument(
        "--model_variant",
        choices=["pred", "flm", "pred_flm", "pred_flm_add"],
        default="flm",
    )
    parser.add_argument(
        "--config",
        choices=["baseline", "multilingual"],
        default="baseline",
        help="baseline: GPT-2 + roberta-base. multilingual: Qwen/Llama/XGLM/BLOOM + mdeberta-v3-base",
    )
    parser.add_argument(
        "--encoder_id",
        type=str,
        default=None,
        help="Optional HF model id for encoder backbone. If not set, uses config default.",
    )
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument(
        "--legacy_roberta_compat",
        action="store_true",
        help="If set: reproduce legacy RoBERTa-baseline behavior (numpy seed bug + slow RobertaTokenizer).",
    )
    args = parser.parse_args()

    subtask = args.subtask
    lang = args.lang
    model_variant = args.model_variant
    seed = int(args.seed)

    if args.encoder_id is not None:
        encoder_id = args.encoder_id
    elif args.config == "multilingual":
        encoder_id = ENCODER_MULTILINGUAL
    else:
        encoder_id = ENCODER_MAP_BASELINE[lang]

    multilingual = args.config == "multilingual"
    is_legacy = bool(args.legacy_roberta_compat and encoder_id == "roberta-base")

    data_dir = Path(PROJECT_ROOT) / "data" / "data"
    features_dir = Path(PROJECT_ROOT) / "data" / "features"

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    log_file = LOG_DIR / f"{subtask}_{lang}_{model_variant}_{args.config}_seed{seed}.log"
    sys.stdout = Tee(log_file)

    print("=" * 80)
    print(
        f"Training | subtask={subtask} | lang={lang} | variant={model_variant} | "
        f"config={args.config} | encoder={encoder_id} | seed={seed}"
    )
    print("=" * 80)

    random.seed(seed)
    np.random.seed(0)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
        code_split=False,
        use_fold=0,
    )

    train_X, dev_X, test_X, fixed_len = compute_all_features(
        train_texts=train_texts,
        dev_texts=dev_texts,
        test_texts=test_texts,
        train_idx=train_idx,
        dev_idx=dev_idx,
        subtask=subtask,
        lang=lang,
        device=DEVICE,
        local_device=LOCAL_DEVICE,
        model_variant=model_variant,
        features_dir=features_dir,
        multilingual=multilingual,
    )

    if multilingual:
        random.seed(seed)
        np.random.seed(0)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    if is_legacy:
        from transformers import RobertaTokenizer

        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        print("[TOK] Using legacy RobertaTokenizer (slow) for roberta-base baseline")
    elif "deberta" in encoder_id.lower():
        from transformers import DebertaV2Tokenizer

        tokenizer = DebertaV2Tokenizer.from_pretrained(encoder_id)
        print(f"[TOK] Using slow DebertaV2Tokenizer for {encoder_id}")
    else:
        tokenizer = AutoTokenizer.from_pretrained(encoder_id, use_fast=True)

    train_enc, dev_enc, test_enc = tokenize_splits(
        train_texts, dev_texts, test_texts, tokenizer, fixed_len,
    )

    num_workers = 0 if multilingual else 4

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
        num_workers=num_workers,
    )

    model = build_model(
        model_variant=model_variant,
        subtask=subtask,
        encoder_id=encoder_id,
        seq_feature_len=train_X.shape[2],
        device=DEVICE,
        local_device=LOCAL_DEVICE,
        baseline_compat_no_freeze=is_legacy,
    )
    checkpoint_prefix = f"{subtask}_{lang}_{model_variant}_{args.config}_seed{seed}"

    result = train_and_evaluate(
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=test_loader,
        model=model,
        model_variant=model_variant,
        device=DEVICE,
        out_dir=CHECKPOINTS_DIR,
        checkpoint_prefix=checkpoint_prefix,
        epochs=EPOCHS,
        freeze_epochs=FREEZE_EPOCHS,
    )

    print(
        f"\nFINAL RESULT: {model_variant} | encoder={encoder_id} | epoch={result.best_epoch} "
        f"| dev={result.dev_f1:.4f} | test={result.test_f1:.4f}"
    )

    results_path = RESULTS_DIR / "results_summary.tsv"
    write_header = not results_path.exists()

    with open(results_path, "a", encoding="utf-8") as f:
        if write_header:
            f.write("subtask\tlang\tvariant\tconfig\tencoder_id\tseed\tbest_epoch\tdev_f1\ttest_f1\n")
        f.write(
            f"{subtask}\t{lang}\t{model_variant}\t{args.config}\t{encoder_id}\t{seed}\t"
            f"{result.best_epoch}\t{result.dev_f1:.6f}\t{result.test_f1:.6f}\n"
        )

    print(f"[INFO] Appended results to {results_path}")


if __name__ == "__main__":
    main()
