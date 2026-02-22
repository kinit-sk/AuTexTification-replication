"""
Optuna hyperparameter tuning for aggregated features (RF / XGBoost / MLP).
tune_optuna_agg.py
"""

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

import optuna
from optuna.samplers import TPESampler

from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

import xgboost as xgb

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from utils.constants import LOG_DIR, OUT_DIR

DATA_OUT_DIR = OUT_DIR / "tuning"
DATA_OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro"))


@dataclass
class AggData:
    subtask: str
    lang: str
    n_classes: int
    train_X: np.ndarray
    dev_X: np.ndarray
    test_X: np.ndarray
    train_y: np.ndarray
    dev_y: np.ndarray
    test_y: np.ndarray
    seed: int
    source: str
    npz_path: Path


def load_agg_npz(npz_path: Path) -> AggData:
    z = np.load(npz_path, allow_pickle=True)
    subtask = str(z["subtask"]) if "subtask" in z else "unknown"
    lang = str(z["lang"]) if "lang" in z else "unknown"
    n_classes = int(z["n_classes"]) if "n_classes" in z else 2
    seed = int(z["seed"]) if "seed" in z else 10
    source = str(z["source"]) if "source" in z else "agg"

    return AggData(
        subtask=subtask,
        lang=lang,
        n_classes=n_classes,
        train_X=z["train_X"],
        dev_X=z["dev_X"],
        test_X=z["test_X"],
        train_y=z["train_y"],
        dev_y=z["dev_y"],
        test_y=z["test_y"],
        seed=seed,
        source=source,
        npz_path=npz_path,
    )


def find_all_npz(out_dir: Path, source: str) -> list[Path]:
    pattern = "*_features_stage2_data.npz" if source == "agg" else "*_stage2_data.npz"
    return sorted(out_dir.glob(pattern))


class MLPClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        n_classes: int,
        hidden_sizes: list[int],
        dropout: float,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_mlp_with_earlystop(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    n_classes: int,
    device: torch.device,
    *,
    hidden_sizes: list[int],
    dropout: float,
    lr: float,
    weight_decay: float,
    label_smoothing: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
) -> tuple[MLPClassifier, StandardScaler, float, int]:
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_dev_s = scaler.transform(X_dev)

    train_ds = TensorDataset(
        torch.tensor(X_train_s, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    dev_ds = TensorDataset(
        torch.tensor(X_dev_s, dtype=torch.float32),
        torch.tensor(y_dev, dtype=torch.long),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False)

    model = MLPClassifier(
        input_size=X_train.shape[1],
        n_classes=n_classes,
        hidden_sizes=hidden_sizes,
        dropout=float(dropout),
        use_batchnorm=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=float(label_smoothing))
    optimizer = Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
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
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        preds = []
        gold = []
        with torch.no_grad():
            for xb, yb in dev_loader:
                xb = xb.to(device)
                p = model(xb).argmax(dim=1).cpu().numpy()
                preds.append(p)
                gold.append(yb.numpy())
        preds = np.concatenate(preds)
        gold = np.concatenate(gold)
        dev_f1 = macro_f1(gold, preds)

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

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = 1
        best_f1 = -1.0

    model.load_state_dict(best_state)
    return model, scaler, float(best_f1), int(best_epoch)


def predict_mlp(
    model: MLPClassifier,
    scaler: StandardScaler,
    X: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    Xs = scaler.transform(X)
    ds = TensorDataset(torch.tensor(Xs, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()
    out = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            out.append(model(xb).argmax(dim=1).cpu().numpy())
    return np.concatenate(out)


def tune_rf(data: AggData, trials: int, seed: int) -> dict[str, Any]:
    Xtr, ytr, Xdv, ydv = data.train_X, data.train_y, data.dev_X, data.dev_y

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 2000, step=100),
            "max_depth": trial.suggest_categorical("max_depth", [None, 10, 20, 40, 60, 80]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, 0.7, 0.9]),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
            "bootstrap": True,
            "random_state": seed,
            "n_jobs": -1,
        }
        clf = RandomForestClassifier(**params)
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xdv)
        return macro_f1(ydv, pred)

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed))
    study.optimize(objective, n_trials=trials, catch=(Exception,))

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError("RF tuning: no completed trials.")

    best_params = dict(study.best_trial.params)
    final_params = {**best_params, "bootstrap": True, "random_state": seed, "n_jobs": -1}

    best_model = RandomForestClassifier(**final_params)
    best_model.fit(Xtr, ytr)

    dev_pred = best_model.predict(Xdv)
    test_pred = best_model.predict(data.test_X)

    return {
        "model": "rf",
        "best_params": final_params,
        "best_dev_f1": macro_f1(ydv, dev_pred),
        "test_f1": macro_f1(data.test_y, test_pred),
        "best_model_obj": best_model,
    }


def _xgb_predict_labels(booster: xgb.Booster, dmat: xgb.DMatrix, n_classes: int) -> np.ndarray:
    preds = booster.predict(dmat)
    if n_classes <= 2:
        return (preds >= 0.5).astype(int)
    return np.argmax(preds, axis=1).astype(int)


def tune_xgb(data: AggData, trials: int, seed: int) -> dict[str, Any]:
    Xtr, ytr, Xdv, ydv = data.train_X, data.train_y, data.dev_X, data.dev_y
    n_classes = data.n_classes

    dtrain = xgb.DMatrix(Xtr, label=ytr)
    ddev = xgb.DMatrix(Xdv, label=ydv)
    dtest = xgb.DMatrix(data.test_X, label=data.test_y)

    if n_classes > 2:
        objective_name = "multi:softprob"
        eval_metric = "mlogloss"
    else:
        objective_name = "binary:logistic"
        eval_metric = "logloss"

    def objective(trial: optuna.Trial) -> float:
        num_boost_round = trial.suggest_int("n_estimators", 200, 3000, step=100)

        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "eta": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "objective": objective_name,
            "eval_metric": eval_metric,
            "seed": seed,
        }
        if n_classes > 2:
            params["num_class"] = n_classes

        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=[(ddev, "dev")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )

        pred = _xgb_predict_labels(booster, ddev, n_classes=n_classes)
        return macro_f1(ydv, pred)

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed))
    study.optimize(objective, n_trials=trials, catch=(Exception,))

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError("XGB tuning: no completed trials.")

    best = dict(study.best_trial.params)
    best_num_boost_round = int(best["n_estimators"])

    final_params = {
        "max_depth": int(best["max_depth"]),
        "eta": float(best["learning_rate"]),
        "subsample": float(best["subsample"]),
        "colsample_bytree": float(best["colsample_bytree"]),
        "min_child_weight": int(best["min_child_weight"]),
        "gamma": float(best["gamma"]),
        "alpha": float(best["reg_alpha"]),
        "lambda": float(best["reg_lambda"]),
        "objective": objective_name,
        "eval_metric": eval_metric,
        "seed": seed,
    }
    if n_classes > 2:
        final_params["num_class"] = n_classes

    best_booster = xgb.train(
        params=final_params,
        dtrain=dtrain,
        num_boost_round=best_num_boost_round,
        evals=[(ddev, "dev")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )

    dev_pred = _xgb_predict_labels(best_booster, ddev, n_classes=n_classes)
    test_pred = _xgb_predict_labels(best_booster, dtest, n_classes=n_classes)

    best_iter = int(getattr(best_booster, "best_iteration", -1)) if hasattr(best_booster, "best_iteration") else -1

    return {
        "model": "xgb",
        "best_params": {**final_params, "n_estimators": best_num_boost_round},
        "best_dev_f1": macro_f1(ydv, dev_pred),
        "test_f1": macro_f1(data.test_y, test_pred),
        "best_model_obj": best_booster,
        "best_iteration": best_iter,
    }


def tune_mlp(data: AggData, trials: int, seed: int, device: torch.device) -> dict[str, Any]:
    Xtr, ytr, Xdv, ydv = data.train_X, data.train_y, data.dev_X, data.dev_y

    hidden_space = [
        [128],
        [256],
        [512],
        [256, 128],
        [512, 256],
        [512, 256, 128],
        [1024, 512],
    ]

    def objective(trial: optuna.Trial) -> float:
        trial_seed = seed + trial.number
        set_seeds(trial_seed)

        hidden_sizes = hidden_space[trial.suggest_int("hidden_idx", 0, len(hidden_space) - 1)]
        dropout = trial.suggest_float("dropout", 0.1, 0.6)
        lr = trial.suggest_float("lr", 1e-5, 3e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.2)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        max_epochs = trial.suggest_int("max_epochs", 80, 250, step=10)
        patience = trial.suggest_int("patience", 10, 30)

        model, _, best_dev_f1, _ = train_mlp_with_earlystop(
            X_train=Xtr,
            y_train=ytr,
            X_dev=Xdv,
            y_dev=ydv,
            n_classes=data.n_classes,
            device=device,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            label_smoothing=label_smoothing,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
        )
        del model
        torch.cuda.empty_cache()
        return float(best_dev_f1)

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed))
    study.optimize(objective, n_trials=trials, catch=(Exception,))

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError("MLP tuning: no completed trials.")

    best = dict(study.best_trial.params)
    hidden_sizes = hidden_space[int(best["hidden_idx"])]

    set_seeds(seed)
    final_model, final_scaler, best_dev_f1, best_epoch = train_mlp_with_earlystop(
        X_train=Xtr,
        y_train=ytr,
        X_dev=Xdv,
        y_dev=ydv,
        n_classes=data.n_classes,
        device=device,
        hidden_sizes=hidden_sizes,
        dropout=float(best["dropout"]),
        lr=float(best["lr"]),
        weight_decay=float(best["weight_decay"]),
        label_smoothing=float(best["label_smoothing"]),
        batch_size=int(best["batch_size"]),
        max_epochs=int(best["max_epochs"]),
        patience=int(best["patience"]),
    )

    dev_pred = predict_mlp(final_model, final_scaler, Xdv, device=device, batch_size=int(best["batch_size"]))
    test_pred = predict_mlp(final_model, final_scaler, data.test_X, device=device, batch_size=int(best["batch_size"]))

    return {
        "model": "mlp",
        "best_params": {
            "hidden_sizes": hidden_sizes,
            "dropout": float(best["dropout"]),
            "lr": float(best["lr"]),
            "weight_decay": float(best["weight_decay"]),
            "label_smoothing": float(best["label_smoothing"]),
            "batch_size": int(best["batch_size"]),
            "max_epochs": int(best["max_epochs"]),
            "patience": int(best["patience"]),
        },
        "best_dev_f1": macro_f1(ydv, dev_pred),
        "test_f1": macro_f1(data.test_y, test_pred),
        "best_epoch": int(best_epoch),
        "best_model_obj": final_model,
        "best_scaler_obj": final_scaler,
    }


def save_results(out_dir: Path, tag: str, rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"tuning_{tag}_{ts}.json"
    tsv_path = out_dir / f"tuning_{tag}_{ts}.tsv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    headers = ["subtask", "lang", "model", "best_dev_f1", "test_f1", "best_params", "npz", "source"]
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("\t".join(headers) + "\n")
        for r in rows:
            f.write(
                "\t".join([
                    str(r["subtask"]),
                    str(r["lang"]),
                    str(r["model"]),
                    f"{float(r['best_dev_f1']):.6f}",
                    f"{float(r['test_f1']):.6f}",
                    json.dumps(r["best_params"], ensure_ascii=False),
                    str(r["npz"]),
                    str(r.get("source", "")),
                ]) + "\n"
            )

    return json_path, tsv_path


def save_best_models(out_dir: Path, data: AggData, result: dict[str, Any], prefix: str) -> list[Path]:
    paths: list[Path] = []
    base = f"{prefix}_{data.subtask}_{data.lang}"

    if result["model"] == "rf":
        p = out_dir / f"{base}_rf.joblib"
        joblib.dump(result["best_model_obj"], p)
        paths.append(p)

    elif result["model"] == "xgb":
        p = out_dir / f"{base}_xgb.json"
        result["best_model_obj"].save_model(str(p))
        paths.append(p)

    elif result["model"] == "mlp":
        model_p = out_dir / f"{base}_mlp.pt"
        scaler_p = out_dir / f"{base}_mlp_scaler.joblib"
        meta_p = out_dir / f"{base}_mlp_meta.json"

        torch.save(result["best_model_obj"].state_dict(), model_p)
        joblib.dump(result["best_scaler_obj"], scaler_p)
        with open(meta_p, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "subtask": data.subtask,
                    "lang": data.lang,
                    "n_classes": data.n_classes,
                    "input_dim": int(data.train_X.shape[1]),
                    "best_params": result["best_params"],
                    "best_dev_f1": float(result["best_dev_f1"]),
                    "test_f1": float(result["test_f1"]),
                    "best_epoch": int(result.get("best_epoch", -1)),
                    "source": data.source,
                    "npz": str(data.npz_path),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        paths.extend([model_p, scaler_p, meta_p])

    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna tuning for aggregated features (RF/XGB/MLP)")
    parser.add_argument("--npz", type=str, default=None, help="Path to one NPZ file.")
    parser.add_argument("--trials", type=int, default=60, help="Trials per model per dataset.")
    parser.add_argument("--models", nargs="+", default=["rf", "xgb", "mlp"], choices=["rf", "xgb", "mlp"], help="Models to tune.")
    parser.add_argument("--seed", type=int, default=10, help="Global seed.")
    parser.add_argument("--device", type=str, default=None, help="cuda:0 / cpu (default: auto).")
    parser.add_argument("--save_best", action="store_true", help="Save best models.")
    parser.add_argument("--source", type=str, default="agg", choices=["agg", "ultrahybrid"], help="Which NPZ pattern to load.")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seeds(args.seed)

    if args.npz:
        npz_paths = [Path(args.npz)]
    else:
        npz_paths = find_all_npz(DATA_OUT_DIR, source=args.source)

    if not npz_paths:
        raise SystemExit(f"No NPZ found in: {DATA_OUT_DIR} (source={args.source})")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"optuna_tune_{args.source}_{ts}.log"
    with open(log_path, "w", encoding="utf-8") as logf:
        logf.write(f"Started: {ts}\n")
        logf.write(f"Device: {device}\n")
        logf.write(f"Source: {args.source}\n")
        logf.write(f"NPZ files: {len(npz_paths)}\n")
        logf.write(f"Models: {args.models}\n")
        logf.write(f"Trials: {args.trials}\n")

    all_rows: list[dict[str, Any]] = []
    prefix = "best_agg" if args.source == "agg" else "best_ultrahybrid"

    for npz_path in npz_paths:
        data = load_agg_npz(npz_path)
        print("\n" + "=" * 90)
        print(f"[DATA] {data.subtask}/{data.lang} | n_classes={data.n_classes} | Xdim={data.train_X.shape[1]}")
        print(f"       train={data.train_X.shape} dev={data.dev_X.shape} test={data.test_X.shape}")
        print(f"       file={data.npz_path}")
        print(f"       source={data.source}")
        print("=" * 90)

        per_dataset_rows: list[dict[str, Any]] = []

        if "rf" in args.models:
            print("\n[TUNE] RF")
            rf_res = tune_rf(data, trials=args.trials, seed=args.seed)
            print(f"  best_dev_f1={rf_res['best_dev_f1']:.4f} | test_f1={rf_res['test_f1']:.4f}")
            row = {
                "subtask": data.subtask,
                "lang": data.lang,
                "model": "rf",
                "best_dev_f1": rf_res["best_dev_f1"],
                "test_f1": rf_res["test_f1"],
                "best_params": rf_res["best_params"],
                "npz": str(data.npz_path),
                "source": data.source,
            }
            per_dataset_rows.append(row)
            if args.save_best:
                saved = save_best_models(DATA_OUT_DIR, data, rf_res, prefix=prefix)
                print(f"  saved: {[p.name for p in saved]}")

        if "xgb" in args.models:
            print("\n[TUNE] XGB")
            xgb_res = tune_xgb(data, trials=args.trials, seed=args.seed)
            bi = xgb_res.get("best_iteration", -1)
            print(f"  best_dev_f1={xgb_res['best_dev_f1']:.4f} | test_f1={xgb_res['test_f1']:.4f} | best_iter={bi}")
            row = {
                "subtask": data.subtask,
                "lang": data.lang,
                "model": "xgb",
                "best_dev_f1": xgb_res["best_dev_f1"],
                "test_f1": xgb_res["test_f1"],
                "best_params": xgb_res["best_params"],
                "best_iteration": bi,
                "npz": str(data.npz_path),
                "source": data.source,
            }
            per_dataset_rows.append(row)
            if args.save_best:
                saved = save_best_models(DATA_OUT_DIR, data, xgb_res, prefix=prefix)
                print(f"  saved: {[p.name for p in saved]}")

        if "mlp" in args.models:
            print("\n[TUNE] MLP")
            mlp_res = tune_mlp(data, trials=args.trials, seed=args.seed, device=device)
            print(f"  best_dev_f1={mlp_res['best_dev_f1']:.4f} | test_f1={mlp_res['test_f1']:.4f} | best_epoch={mlp_res.get('best_epoch', -1)}")
            row = {
                "subtask": data.subtask,
                "lang": data.lang,
                "model": "mlp",
                "best_dev_f1": mlp_res["best_dev_f1"],
                "test_f1": mlp_res["test_f1"],
                "best_params": mlp_res["best_params"],
                "best_epoch": mlp_res.get("best_epoch", -1),
                "npz": str(data.npz_path),
                "source": data.source,
            }
            per_dataset_rows.append(row)
            if args.save_best:
                saved = save_best_models(DATA_OUT_DIR, data, mlp_res, prefix=prefix)
                print(f"  saved: {[p.name for p in saved]}")

        tag = f"{args.source}_{data.subtask}_{data.lang}"
        json_path, tsv_path = save_results(DATA_OUT_DIR, tag=tag, rows=per_dataset_rows)
        print(f"\n[RESULTS] saved: {json_path.name}, {tsv_path.name}")

        all_rows.extend(per_dataset_rows)

    json_path, tsv_path = save_results(DATA_OUT_DIR, tag=f"{args.source}_ALL", rows=all_rows)
    print("\n" + "=" * 90)
    print(f"[DONE] Combined results saved: {json_path.name}, {tsv_path.name}")
    print("=" * 90)
    print(f"Log file: {log_path}")


if __name__ == "__main__":
    main()
