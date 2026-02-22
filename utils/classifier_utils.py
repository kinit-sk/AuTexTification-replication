"""Shared Stage-2 classifiers: PyTorch MLP, Random Forest, XGBoost."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

RF_N_ESTIMATORS: int = 200
RF_MAX_DEPTH: int = 60

XGB_N_ESTIMATORS: int = 300
XGB_MAX_DEPTH: int = 8
XGB_LEARNING_RATE: float = 0.1
XGB_EARLY_STOPPING: int = 50

MLP_HIDDEN: list[int] = [256, 128]
MLP_DROPOUT: float = 0.5
MLP_LR: float = 5e-4
MLP_WEIGHT_DECAY: float = 1e-4
MLP_LABEL_SMOOTHING: float = 0.1
MLP_BATCH_SIZE: int = 64
MLP_MAX_EPOCHS: int = 200
MLP_PATIENCE: int = 20


class MLPNet(nn.Module):
    """MLP with BatchNorm, ReLU, and Dropout."""

    def __init__(
        self,
        input_size: int,
        n_classes: int,
        hidden_sizes: list[int],
        dropout: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_mlp(
    train_X: np.ndarray,
    train_y: np.ndarray,
    dev_X: np.ndarray,
    dev_y: np.ndarray,
    n_classes: int,
    device: torch.device,
    *,
    hidden_sizes: list[int] | None = None,
    dropout: float = MLP_DROPOUT,
    lr: float = MLP_LR,
    weight_decay: float = MLP_WEIGHT_DECAY,
    label_smoothing: float = MLP_LABEL_SMOOTHING,
    batch_size: int = MLP_BATCH_SIZE,
    max_epochs: int = MLP_MAX_EPOCHS,
    patience: int = MLP_PATIENCE,
) -> tuple[MLPNet, StandardScaler, int]:
    """Train MLP with dev-set early stopping. Returns (model, scaler, best_epoch)."""
    if hidden_sizes is None:
        hidden_sizes = list(MLP_HIDDEN)

    scaler = StandardScaler()
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(scaler.fit_transform(train_X), dtype=torch.float32),
            torch.tensor(train_y, dtype=torch.long),
        ),
        batch_size=batch_size, shuffle=True,
    )
    dev_loader = DataLoader(
        TensorDataset(
            torch.tensor(scaler.transform(dev_X), dtype=torch.float32),
            torch.tensor(dev_y, dtype=torch.long),
        ),
        batch_size=batch_size, shuffle=False,
    )

    model = MLPNet(
        input_size=train_X.shape[1], n_classes=n_classes,
        hidden_sizes=hidden_sizes, dropout=dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    best_f1 = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    bad = 0

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

        model.eval()
        preds_list: list[np.ndarray] = []
        gold_list: list[np.ndarray] = []
        with torch.no_grad():
            for xb, yb in dev_loader:
                preds_list.append(model(xb.to(device)).argmax(dim=1).cpu().numpy())
                gold_list.append(yb.numpy())

        dev_f1 = float(f1_score(np.concatenate(gold_list), np.concatenate(preds_list), average="macro"))
        scheduler.step(dev_f1)

        if dev_f1 > best_f1:
            best_f1 = dev_f1
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, scaler, best_epoch


def predict_mlp(
    model: MLPNet,
    scaler: StandardScaler,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = MLP_BATCH_SIZE,
) -> np.ndarray:
    loader = DataLoader(
        TensorDataset(torch.tensor(scaler.transform(X), dtype=torch.float32)),
        batch_size=batch_size, shuffle=False,
    )
    model.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for (xb,) in loader:
            out.append(model(xb.to(device)).argmax(dim=1).cpu().numpy())
    return np.concatenate(out)


type ClassifierResults = dict[str, dict[str, float]]


def run_classifiers(
    train_X: np.ndarray,
    dev_X: np.ndarray,
    test_X: np.ndarray,
    train_y: np.ndarray,
    dev_y: np.ndarray,
    test_y: np.ndarray,
    n_classes: int,
    seed: int,
    device: torch.device,
) -> ClassifierResults:
    """Train RF, XGBoost, and PyTorch MLP. Returns per-classifier F1 scores."""
    macro = lambda yt, yp: float(f1_score(yt, yp, average="macro"))
    results: ClassifierResults = {}

    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH,
        random_state=seed, n_jobs=-1,
    )
    rf.fit(train_X, train_y)
    results["rf"] = {
        "train_f1": macro(train_y, rf.predict(train_X)),
        "dev_f1": macro(dev_y, rf.predict(dev_X)),
        "test_f1": macro(test_y, rf.predict(test_X)),
    }
    print(f"  RF  → dev={results['rf']['dev_f1']:.4f}  test={results['rf']['test_f1']:.4f}")

    eval_metric = "mlogloss" if n_classes > 2 else "logloss"
    xgb_clf = xgb.XGBClassifier(
        n_estimators=XGB_N_ESTIMATORS, max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=seed, n_jobs=-1,
        eval_metric=eval_metric,
        early_stopping_rounds=XGB_EARLY_STOPPING,
    )
    xgb_clf.fit(train_X, train_y, eval_set=[(dev_X, dev_y)], verbose=False)
    results["xgb"] = {
        "train_f1": macro(train_y, xgb_clf.predict(train_X)),
        "dev_f1": macro(dev_y, xgb_clf.predict(dev_X)),
        "test_f1": macro(test_y, xgb_clf.predict(test_X)),
    }
    print(f"  XGB → dev={results['xgb']['dev_f1']:.4f}  test={results['xgb']['test_f1']:.4f}  iter={xgb_clf.best_iteration}")

    mlp_model, mlp_scaler, best_epoch = train_mlp(
        train_X, train_y, dev_X, dev_y, n_classes, device,
    )
    results["mlp"] = {
        "train_f1": macro(train_y, predict_mlp(mlp_model, mlp_scaler, train_X, device)),
        "dev_f1": macro(dev_y, predict_mlp(mlp_model, mlp_scaler, dev_X, device)),
        "test_f1": macro(test_y, predict_mlp(mlp_model, mlp_scaler, test_X, device)),
    }
    print(f"  MLP → dev={results['mlp']['dev_f1']:.4f}  test={results['mlp']['test_f1']:.4f}  epoch={best_epoch}")

    return results
