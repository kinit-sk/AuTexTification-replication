# utils/data_utils.py

import numpy as np
import pandas as pd
from pathlib import Path


def load_train_dev_test(
    train_dir: Path,
    test_dir: Path,
    subtask: str,
    seed: int = 10,
    code_split: bool = False,
    use_fold: int = 0,
):
    """Load train/dev/test data. Returns texts, labels, and indices."""

    # train
    train_path = train_dir / "train.tsv"
    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")

    train_df = pd.read_csv(train_path, sep="\t")

    if subtask == "subtask_1":
        label_map = {"human": 0, "generated": 1}
        train_df["y"] = train_df["label"].str.lower().map(label_map)
    else:
        train_df["y"] = train_df["label"].astype(str).str.strip().str.upper().apply(
            lambda x: ord(x[0]) - ord("A")
        )

    if code_split:
        print("[INFO] Using predefined fold split")
        folds_path = train_dir / "train_5folds.tsv"
        folds_df = pd.read_csv(folds_path, sep="\t", header=None, names=["id", "fold"])
        fold_map = dict(zip(folds_df["id"].astype(str), folds_df["fold"].astype(str)))

        train_df["id_str"] = train_df["id"].astype(str)
        train_df["fold"] = train_df["id_str"].map(lambda x: fold_map.get(x, "1"))

        dev_mask = train_df["fold"] == str(use_fold)
        train_mask = ~dev_mask
    else:
        print("[INFO] Using random 80/20 split")
        train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
        split_idx = int(len(train_df) * 0.8)

        train_mask = np.zeros(len(train_df), dtype=bool)
        train_mask[:split_idx] = True
        dev_mask = ~train_mask

    train_texts = train_df.loc[train_mask, "text"].fillna("").astype(str).tolist()
    dev_texts = train_df.loc[dev_mask, "text"].fillna("").astype(str).tolist()

    train_Y = train_df.loc[train_mask, "y"].astype(int).to_numpy()
    dev_Y = train_df.loc[dev_mask, "y"].astype(int).to_numpy()

    train_idx = train_df.loc[train_mask].index.to_numpy()
    dev_idx = train_df.loc[dev_mask].index.to_numpy()

    # test
    test_path = test_dir / "test.tsv"
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    test_df = pd.read_csv(test_path, sep="\t")
    test_df["text"] = test_df["text"].fillna("").astype(str)

    if subtask == "subtask_1":
        label_map = {"human": 0, "generated": 1}
        test_df["y"] = test_df["label"].str.lower().map(label_map)
    else:
        test_df["y"] = test_df["label"].astype(str).str.upper().apply(
            lambda x: ord(x[0]) - ord("A")
        )

    test_texts = test_df["text"].tolist()
    test_Y = test_df["y"].astype(int).to_numpy()

    print(f"[INFO] Train samples={len(train_Y)} | Dev samples={len(dev_Y)}")
    print(f"[INFO] Test samples={len(test_Y)}")

    return (
        train_texts,
        dev_texts,
        test_texts,
        train_Y,
        dev_Y,
        test_Y,
        train_idx,
        dev_idx,
    )
