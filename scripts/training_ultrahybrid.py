"""UltraHybrid: Hybrid+ (Stage 1) → RF / XGBoost / MLP on linguistic features + output probs (Stage 2)."""

from __future__ import annotations

import argparse
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from feature_extraction.linguistic_features import LinguisticFeatures
from feature_extraction.style_features import StyleFeatures
from utils.constants import (
    BATCH_SIZE,
    CHECKPOINTS_DIR,
    DATA_DIR,
    DEVICE,
    ENCODER_MAP_BASELINE,
    ENCODER_MULTILINGUAL,
    EPOCHS,
    FEATURES_DIR,
    FREEZE_EPOCHS,
    LOCAL_DEVICE,
    LOG_DIR,
    OUT_DIR,
)
from utils.data_utils import load_train_dev_test
from utils.feature_utils import compute_all_features
from utils.classifier_utils import run_classifiers
from utils.logging_utils import Tee
from utils.training_pipeline import (
    build_model,
    create_dataloaders,
    tokenize_splits,
    train_and_evaluate,
)

RESOURCES_DIR: Path = Path(PROJECT_ROOT) / "resources"
ULTRAHYBRID_DIR: Path = OUT_DIR / "tuning"


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(0)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_probabilities(
    model: torch.nn.Module,
    seq_features: np.ndarray,
    enc: dict[str, torch.Tensor],
    labels: np.ndarray,
    device: torch.device,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """Run ordered inference and return softmax probabilities."""
    dataset = TensorDataset(
        torch.tensor(seq_features).float(),
        enc["input_ids"],
        enc["attention_mask"],
        torch.tensor(labels).long(),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    all_probs: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = [b.to(device, non_blocking=True) for b in batch]
            log_probs = model(*batch[:-1])
            all_probs.append(torch.exp(log_probs).cpu().numpy())

    return np.concatenate(all_probs)


def main() -> None:
    parser = argparse.ArgumentParser(description="UltraHybrid: Hybrid+ → RF / XGB / MLP")
    parser.add_argument("--subtask", choices=["subtask_1", "subtask_2"], default="subtask_1")
    parser.add_argument("--lang", choices=["en", "es"], default="en")
    parser.add_argument("--config", choices=["baseline", "multilingual"], default="baseline")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--style", action="store_true", help="Include style features in Stage 2")
    args = parser.parse_args()

    subtask, lang, seed = args.subtask, args.lang, args.seed
    multilingual = args.config == "multilingual"
    encoder_id = ENCODER_MULTILINGUAL if multilingual else ENCODER_MAP_BASELINE[lang]
    n_classes = 2 if subtask == "subtask_1" else 6

    for d in (ULTRAHYBRID_DIR, CHECKPOINTS_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tee = Tee(LOG_DIR / f"ultrahybrid_{subtask}_{lang}_{args.config}_{timestamp}.log")
    sys.stdout = tee

    print("=" * 80)
    print(f"UltraHybrid | {subtask} | {lang} | {args.config} | encoder={encoder_id} | seed={seed}")
    print("=" * 80)

    set_seeds(seed)

    train_texts, dev_texts, test_texts, train_Y, dev_Y, test_Y, train_idx, dev_idx = (
        load_train_dev_test(
            train_dir=DATA_DIR / "train" / subtask / lang,
            test_dir=DATA_DIR / "test" / subtask / lang,
            subtask=subtask, seed=seed,
        )
    )

    # ── Stage 1: Train Hybrid+ and extract output probabilities ──────────
    train_X, dev_X, test_X, fixed_len = compute_all_features(
        train_texts=train_texts, dev_texts=dev_texts, test_texts=test_texts,
        train_idx=train_idx, dev_idx=dev_idx,
        subtask=subtask, lang=lang,
        device=DEVICE, local_device=LOCAL_DEVICE,
        model_variant="pred_flm_add", features_dir=FEATURES_DIR,
        multilingual=multilingual,
    )

    if multilingual:
        set_seeds(seed)

    if "deberta" in encoder_id.lower():
        from transformers import DebertaV2Tokenizer
        tokenizer = DebertaV2Tokenizer.from_pretrained(encoder_id)
    else:
        tokenizer = AutoTokenizer.from_pretrained(encoder_id, use_fast=True)

    train_enc, dev_enc, test_enc = tokenize_splits(
        train_texts, dev_texts, test_texts, tokenizer, fixed_len,
    )

    num_workers = 0 if multilingual else 4
    train_loader, dev_loader, test_loader = create_dataloaders(
        train_X=train_X, dev_X=dev_X, test_X=test_X,
        train_enc=train_enc, dev_enc=dev_enc, test_enc=test_enc,
        train_Y=train_Y, dev_Y=dev_Y, test_Y=test_Y,
        batch_size=BATCH_SIZE, num_workers=num_workers,
    )

    model = build_model(
        model_variant="pred_flm_add", subtask=subtask,
        encoder_id=encoder_id, seq_feature_len=train_X.shape[2],
        device=DEVICE, local_device=LOCAL_DEVICE,
    )

    prefix = f"{subtask}_{lang}_ultrahybrid_{args.config}_seed{seed}"
    hybrid_result = train_and_evaluate(
        train_loader=train_loader, dev_loader=dev_loader, test_loader=test_loader,
        model=model, model_variant="pred_flm_add", device=DEVICE,
        out_dir=CHECKPOINTS_DIR, checkpoint_prefix=prefix,
        epochs=EPOCHS, freeze_epochs=FREEZE_EPOCHS, cleanup_non_best=True,
    )
    print(f"\n[HYBRID+] epoch={hybrid_result.best_epoch} | dev={hybrid_result.dev_f1:.4f} | test={hybrid_result.test_f1:.4f}")

    print("\n[STAGE 1] Extracting Hybrid+ output probabilities...")
    train_probs = extract_probabilities(model, train_X, train_enc, train_Y, DEVICE)
    dev_probs = extract_probabilities(model, dev_X, dev_enc, dev_Y, DEVICE)
    test_probs = extract_probabilities(model, test_X, test_enc, test_Y, DEVICE)
    print(f"  Shape: {train_probs.shape}")

    del model
    torch.cuda.empty_cache()

    # ── Stage 2: Linguistic features + Hybrid+ probs → classifiers ───────
    print("\n[STAGE 2] Extracting linguistic features...")
    ling_ext = LinguisticFeatures(language=lang, resources_dir=RESOURCES_DIR)
    train_ling, feat_names = ling_ext.extract_features(train_texts)
    dev_ling, _ = ling_ext.extract_features(dev_texts, feature_names=feat_names)
    test_ling, _ = ling_ext.extract_features(test_texts, feature_names=feat_names)

    if args.style:
        style_ext = StyleFeatures(language=lang)
        tr_s, s_names = style_ext.extract(train_texts, cache_key=f"train_{subtask}_{lang}")
        dv_s, _ = style_ext.extract(dev_texts, cache_key=f"dev_{subtask}_{lang}")
        te_s, _ = style_ext.extract(test_texts, cache_key=f"test_{subtask}_{lang}")
        train_ling = np.concatenate([train_ling, tr_s], axis=1)
        dev_ling = np.concatenate([dev_ling, dv_s], axis=1)
        test_ling = np.concatenate([test_ling, te_s], axis=1)
        feat_names = feat_names + s_names

    prob_names = [f"HYBRID_PROB_{i}" for i in range(train_probs.shape[1])]
    all_names = feat_names + prob_names

    train_combined = np.concatenate([train_ling, train_probs], axis=1)
    dev_combined = np.concatenate([dev_ling, dev_probs], axis=1)
    test_combined = np.concatenate([test_ling, test_probs], axis=1)
    print(f"  Stage 2 input: {train_combined.shape[1]} features ({len(feat_names)} ling + {len(prob_names)} prob)")

    npz_path = ULTRAHYBRID_DIR / f"{subtask}_{lang}_{args.config}_stage2_data.npz"
    np.savez(
        npz_path,
        train_X=train_combined, dev_X=dev_combined, test_X=test_combined,
        train_y=train_Y, dev_y=dev_Y, test_y=test_Y,
        subtask=subtask, lang=lang, n_classes=n_classes,
        seed=seed, source="ultrahybrid", feature_names=np.array(all_names),
    )
    print(f"  Saved NPZ: {npz_path}")

    print("\n[STAGE 2] Training classifiers...")
    stage2 = run_classifiers(
        train_combined, dev_combined, test_combined,
        train_Y, dev_Y, test_Y, n_classes=n_classes, seed=seed, device=DEVICE,
    )

    print("\n" + "=" * 72)
    print(f"{'Model':<12} {'Train F1':<10} {'Dev F1':<10} {'Test F1':<10} {'Δ Dev':<10} {'Δ Test':<10}")
    print("-" * 72)
    print(f"{'Hybrid+':<12} {'-':<10} {hybrid_result.dev_f1:<10.4f} {hybrid_result.test_f1:<10.4f} {'-':<10} {'-':<10}")
    for name, res in stage2.items():
        dd = res["dev_f1"] - hybrid_result.dev_f1
        dt = res["test_f1"] - hybrid_result.test_f1
        print(f"{name.upper():<12} {res['train_f1']:<10.4f} {res['dev_f1']:<10.4f} {res['test_f1']:<10.4f} {dd:<+10.4f} {dt:<+10.4f}")
    print("=" * 72)

    print(f"\nFor Optuna-tuned Stage 2: python scripts/tune_optuna.py --npz {npz_path} --models rf xgb mlp")

    tee.close()


if __name__ == "__main__":
    main()
