"""Pre-compute BiLSTM output probabilities once per (subtask, lang) for ablation reuse.

The BiLSTM operates on probabilistic (GPT-2) features which are independent of any
style feature group — its output is therefore identical for every ablation config.
Train once, cache to disk, load cheaply in training_ablation.py.

Output
------
data/out/ablation_probs/{subtask}_{lang}[_multilingual]_pred_probs.npz
  Keys: train_probs, dev_probs, test_probs  (float32 arrays)

Usage
-----
  python scripts/ablation/precompute_lstm_probs.py
  python scripts/ablation/precompute_lstm_probs.py --subtask subtask_1 --lang en
  python scripts/ablation/precompute_lstm_probs.py --force
"""

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
import torch
from sklearn.metrics import f1_score
from torch.nn import NLLLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from feature_extraction.probabilistic_features import MultilingualProbFeatures, ProbabilisticFeatures
from models.hybrid import PredLSTM
from utils.constants import DEVICE, LOCAL_DEVICE, OUT_DIR
from utils.data_utils import load_train_dev_test

# ---------------------------------------------------------------------------
# Paths & hyper-parameters (mirror training_lingrf.py)
# ---------------------------------------------------------------------------

ABLATION_PROBS_DIR: Path = OUT_DIR / "ablation_probs"

SEED = 10
NUMPY_SEED = 0
CODE_SPLIT = False
USE_FOLD = 0

LSTM_EPOCHS = 20
LSTM_BATCH_SIZE = 16
LSTM_LR = 1e-3
EARLY_STOP_TOLERANCE = 0.01


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-compute BiLSTM probabilities for ablation reuse."
    )
    parser.add_argument(
        "--subtask", choices=["subtask_1", "subtask_2"], default=None,
        help="Single subtask (default: run both)",
    )
    parser.add_argument(
        "--lang", choices=["en", "es"], default=None,
        help="Single language (default: run both)",
    )
    parser.add_argument(
        "--multilingual", action="store_true",
        help="Use multilingual models instead of baseline GPT-2",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-compute even if the output file already exists",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# LSTM training (self-contained copy; does NOT import training_lingrf.py)
# ---------------------------------------------------------------------------

def _train_lstm_and_get_probs(
    train_texts: list[str],
    dev_texts: list[str],
    test_texts: list[str],
    train_Y: np.ndarray,
    dev_Y: np.ndarray,
    lang: str,
    subtask: str,
    use_multilingual: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if use_multilingual:
        from feature_extraction.probabilistic_features import MULTILINGUAL_LARGE_MODELS
        print(f"\n[LSTM] Using MULTILINGUAL models: {MULTILINGUAL_LARGE_MODELS}")
        prob_extractor = MultilingualProbFeatures(device=DEVICE)
    else:
        print(f"\n[LSTM] Using baseline GPT-2 models ({lang})...")
        prob_extractor = ProbabilisticFeatures(
            device=DEVICE,
            local_device=LOCAL_DEVICE,
            language=lang,
            disabled=False,
        )

    train_features = np.array(prob_extractor.word_features(train_texts))
    dev_features = np.array(prob_extractor.word_features(dev_texts))
    test_features = np.array(prob_extractor.word_features(test_texts))
    print(f"[LSTM] Predictability features shape: {train_features.shape}")

    model = PredLSTM(
        seq_feature_len=train_features.shape[2],
        task=subtask,
        local_device=DEVICE,
    ).to(DEVICE)

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(train_features, dtype=torch.float32),
            torch.tensor(train_Y, dtype=torch.long),
        ),
        batch_size=LSTM_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    dev_loader = DataLoader(
        TensorDataset(torch.tensor(dev_features, dtype=torch.float32)),
        batch_size=LSTM_BATCH_SIZE,
        shuffle=False,
    )

    optimizer = Adam(model.parameters(), lr=LSTM_LR)
    criterion = NLLLoss()

    print(f"[LSTM] Training for up to {LSTM_EPOCHS} epochs on {DEVICE}...")

    best_f1 = 0.0
    best_epoch = 0
    best_state: dict | None = None

    for epoch in range(1, LSTM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(features), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        model.eval()
        dev_preds_list: list[np.ndarray] = []
        with torch.no_grad():
            for (batch,) in dev_loader:
                preds = torch.argmax(model(batch.to(DEVICE)), dim=1)
                dev_preds_list.append(preds.cpu().numpy())

        dev_preds = np.concatenate(dev_preds_list)
        dev_f1 = f1_score(dev_Y, dev_preds, average="macro")

        if epoch % 5 == 0 or epoch == 1:
            print(f"[LSTM] Epoch {epoch}/{LSTM_EPOCHS} | Loss: {avg_loss:.4f} | Dev F1: {dev_f1:.4f}")

        if dev_f1 > best_f1:
            best_f1 = dev_f1
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif dev_f1 >= best_f1 - EARLY_STOP_TOLERANCE and epoch > best_epoch:
            print(f"[LSTM EARLY STOP] Epoch {epoch}: Dev F1={dev_f1:.4f}")
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    print(f"[LSTM] Selected epoch {best_epoch} with dev F1={best_f1:.4f}")

    def _get_probs(feats: np.ndarray) -> np.ndarray:
        model.eval()
        all_probs: list[np.ndarray] = []
        loader = DataLoader(
            TensorDataset(torch.tensor(feats, dtype=torch.float32)),
            batch_size=LSTM_BATCH_SIZE,
            shuffle=False,
        )
        with torch.no_grad():
            for (batch,) in loader:
                probs = torch.exp(model(batch.to(DEVICE)))
                all_probs.append(probs.cpu().numpy())
        return np.concatenate(all_probs, axis=0)

    train_probs = _get_probs(train_features)
    dev_probs = _get_probs(dev_features)
    test_probs = _get_probs(test_features)
    print(f"[LSTM] Output probabilities shape: {train_probs.shape}")

    return train_probs, dev_probs, test_probs


# ---------------------------------------------------------------------------
# Per-config runner
# ---------------------------------------------------------------------------

def run_for_config(subtask: str, lang: str, args: argparse.Namespace) -> None:
    suffix = "_multilingual" if args.multilingual else ""
    out_path = ABLATION_PROBS_DIR / f"{subtask}_{lang}{suffix}_pred_probs.npz"

    if out_path.exists() and not args.force:
        print(f"[SKIP] Already exists (use --force to recompute): {out_path}")
        return

    print("\n" + "=" * 80)
    print(f"Pre-computing LSTM probs | subtask={subtask} | lang={lang} | multilingual={args.multilingual}")
    print("=" * 80)

    random.seed(SEED)
    np.random.seed(NUMPY_SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

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
            seed=SEED,
            code_split=CODE_SPLIT,
            use_fold=USE_FOLD,
        )
    except FileNotFoundError as e:
        print(f"[SKIP] Data not found for {subtask}/{lang}: {e}")
        return

    print(f"\n[DATA] Train: {len(train_texts)}, Dev: {len(dev_texts)}, Test: {len(test_texts)}")

    train_probs, dev_probs, test_probs = _train_lstm_and_get_probs(
        train_texts=train_texts,
        dev_texts=dev_texts,
        test_texts=test_texts,
        train_Y=train_Y,
        dev_Y=dev_Y,
        lang=lang,
        subtask=subtask,
        use_multilingual=args.multilingual,
    )

    ABLATION_PROBS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        train_probs=train_probs,
        dev_probs=dev_probs,
        test_probs=test_probs,
    )
    print(f"\n[SAVED] {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    subtasks = ["subtask_1", "subtask_2"] if args.subtask is None else [args.subtask]
    languages = ["en", "es"] if args.lang is None else [args.lang]

    for subtask in subtasks:
        for lang in languages:
            run_for_config(subtask, lang, args)

    print("\nPre-computation complete.")


if __name__ == "__main__":
    main()
