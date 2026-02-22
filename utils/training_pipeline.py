"""Shared training pipeline: model construction, dataloaders, and train-evaluate loop."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from models.hybrid import FLMRoBERTa, HybridBiLSTMRoBERTa, PredLSTM
from utils.train_utils import eval_loop, train_loop


@dataclass(frozen=True)
class TrainingResult:
    best_epoch: int
    dev_f1: float
    test_f1: float
    best_checkpoint: Path


def build_model(
    *,
    model_variant: str,
    subtask: str,
    encoder_id: str,
    seq_feature_len: int,
    device: torch.device,
    local_device: torch.device,
    baseline_compat_no_freeze: bool = False,
) -> PredLSTM | FLMRoBERTa | HybridBiLSTMRoBERTa:
    """Instantiate the correct model architecture and move it to *device*."""
    if model_variant == "pred":
        return PredLSTM(
            seq_feature_len=seq_feature_len,
            task=subtask,
            local_device=local_device,
        ).to(device)

    if model_variant == "flm":
        return FLMRoBERTa(
            task=subtask,
            local_device=local_device,
            roberta_variant=encoder_id,
            baseline_compat_no_freeze=baseline_compat_no_freeze,
        ).to(device)

    return HybridBiLSTMRoBERTa(
        seq_feature_len=seq_feature_len,
        task=subtask,
        local_device=local_device,
        roberta_variant=encoder_id,
        disable_sequence=False,
    ).to(device)


def tokenize_splits(
    train_texts: list[str],
    dev_texts: list[str],
    test_texts: list[str],
    tokenizer,
    max_length: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Tokenize three text splits with the same tokenizer and max-length."""

    def _encode(texts: list[str]) -> dict[str, torch.Tensor]:
        return tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    return _encode(train_texts), _encode(dev_texts), _encode(test_texts)


def create_dataloaders(
    *,
    train_X: np.ndarray,
    dev_X: np.ndarray,
    test_X: np.ndarray,
    train_enc: dict[str, torch.Tensor],
    dev_enc: dict[str, torch.Tensor],
    test_enc: dict[str, torch.Tensor],
    train_Y: np.ndarray,
    dev_Y: np.ndarray,
    test_Y: np.ndarray,
    batch_size: int = 16,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train / dev / test DataLoader instances."""
    pin = torch.cuda.is_available()

    def _make_dataset(
        X: np.ndarray,
        enc: dict[str, torch.Tensor],
        Y: np.ndarray,
    ) -> TensorDataset:
        return TensorDataset(
            torch.tensor(X).float(),
            enc["input_ids"],
            enc["attention_mask"],
            torch.tensor(Y).long(),
        )

    return (
        DataLoader(
            _make_dataset(train_X, train_enc, train_Y),
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin,
        ),
        DataLoader(
            _make_dataset(dev_X, dev_enc, dev_Y),
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin,
        ),
        DataLoader(
            _make_dataset(test_X, test_enc, test_Y),
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin,
        ),
    )


def train_and_evaluate(
    *,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    test_loader: DataLoader,
    model: torch.nn.Module,
    model_variant: str,
    device: torch.device,
    out_dir: Path,
    checkpoint_prefix: str,
    epochs: int = 20,
    freeze_epochs: int = 5,
    cleanup_non_best: bool = False,
) -> TrainingResult:
    """Train, select best epoch (earliest within 0.01 of max dev-F1), evaluate on test."""

    def _make_lstm_optimizer() -> Adam:
        params = [
            p
            for name, p in model.named_parameters()
            if ("lstm" in name or "linear_layer" in name) and p.requires_grad
        ]
        return Adam(params, lr=1e-3)

    if model_variant == "pred":
        optimizer = Adam(model.parameters(), lr=1e-3)
    elif model_variant == "flm":
        optimizer = Adam(model.parameters(), lr=2e-5)
    else:
        model.freeze_llm()
        optimizer = _make_lstm_optimizer()

    dev_history: list[float] = []
    ckpt_paths: list[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        print(f"\n===== EPOCH {epoch + 1}/{epochs} =====")

        if model_variant in ("pred_flm", "pred_flm_add") and epoch == freeze_epochs:
            print("[PHASE 2] Unfreezing LM and switching LR to 2e-5")
            model.unfreeze_llm()
            optimizer = Adam(model.parameters(), lr=2e-5)

        train_loop(train_loader, model, optimizer, device)
        dev_f1 = eval_loop(dev_loader, model, device)
        dev_history.append(dev_f1)

        ckpt = out_dir / f"{checkpoint_prefix}_epoch{epoch + 1}.pt"
        ckpt_paths.append(ckpt)
        torch.save(
            {"epoch": epoch + 1, "model_state": model.state_dict(), "dev_f1": dev_f1},
            ckpt,
        )
        print(f"Saved checkpoint epoch {epoch + 1}")

    dev_arr = np.array(dev_history)
    f_max = float(dev_arr.max())
    threshold = f_max - 0.01
    best_idx = int(np.argmax(dev_arr >= threshold))
    best_epoch = best_idx + 1
    best_dev = float(dev_arr[best_idx])
    best_ckpt = ckpt_paths[best_idx]

    print(f"\nBest dev F1 = {best_dev:.4f} | selecting epoch {best_epoch}")

    state = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(state["model_state"])

    test_f1 = eval_loop(test_loader, model, device, test=True)

    if cleanup_non_best:
        for ckpt in ckpt_paths:
            if ckpt != best_ckpt and ckpt.exists():
                ckpt.unlink()

    return TrainingResult(
        best_epoch=best_epoch,
        dev_f1=best_dev,
        test_f1=float(test_f1),
        best_checkpoint=best_ckpt,
    )
