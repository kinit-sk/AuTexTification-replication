"""LingRF training with linguistic + style features and SHAP interpretation."""

from __future__ import annotations

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import sys
from datetime import datetime
from pathlib import Path

from _bootstrap import configure_project_root

PROJECT_ROOT: str = str(configure_project_root(__file__, remove_shadowing_utils=False))

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.nn import NLLLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from feature_extraction.linguistic_features import (
    LingRFClassifier,
    LingRFPredOutClassifier,
    LinguisticFeatures,
)
from feature_extraction.probabilistic_features import MultilingualProbFeatures, ProbabilisticFeatures
from feature_extraction.style_features import StyleFeatures
from models.hybrid import PredLSTM
from utils.constants import (
    DEVICE,
    LOCAL_DEVICE,
    LOG_DIR,
    OUT_DIR,
    RESULTS_DIR,
    SHAP_DIR,
)
from utils.data_utils import load_train_dev_test
from utils.env_fingerprint import log_env_fingerprint, set_determinism
from utils.logging_utils import Tee

try:
    import shap

    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("[WARNING] shap not installed. Install with: pip install shap matplotlib")


# arguments
parser = argparse.ArgumentParser(description="Train LingRF models with style features and SHAP")
parser.add_argument("--subtask", choices=["subtask_1", "subtask_2"], default=None,
                    help="Run single subtask (default: run all)")
parser.add_argument("--lang", choices=["en", "es"], default=None,
                    help="Run single language (default: run all)")
parser.add_argument("--variant", choices=["lingrf", "lingrf_style", "lingrf_predout", "lingrf_style_predout", "all"], default="all",
                    help="Model variant (default: run all)")
parser.add_argument("--multilingual", action="store_true",
                    help="Use multilingual_large models for predout variants (default: use baseline GPT-2)")
parser.add_argument("--no-shap", action="store_true",
                    help="Skip SHAP analysis")
parser.add_argument("--shap-samples", type=int, default=100,
                    help="Number of samples for SHAP analysis (default: 100)")
parser.add_argument("--seed", type=int, default=10,
                    help="Seed for RF/LSTM init and data split (default: 10)")
args = parser.parse_args()

SEED = args.seed
NUMPY_SEED = 0
CODE_SPLIT = False
USE_FOLD = 0

N_ESTIMATORS = 200
MAX_DEPTH = 60

LSTM_EPOCHS = 20
LSTM_BATCH_SIZE = 16
LSTM_LR = 1e-3

SUBTASKS = ["subtask_1", "subtask_2"] if args.subtask is None else [args.subtask]
LANGUAGES = ["en", "es"] if args.lang is None else [args.lang]

if args.variant == "all":
    VARIANTS = ["lingrf", "lingrf_style", "lingrf_predout", "lingrf_style_predout"]
else:
    VARIANTS = [args.variant]

DATA_DIR = Path(PROJECT_ROOT) / "data" / "data"
RESOURCES_DIR = Path(PROJECT_ROOT) / "resources"
FEATURES_DIR = Path(PROJECT_ROOT) / "data" / "features"

TUNING_DIR: Path = OUT_DIR / "tuning"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SHAP_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
TUNING_DIR.mkdir(parents=True, exist_ok=True)

DO_SHAP = HAS_SHAP and not args.no_shap
SHAP_SAMPLES = args.shap_samples
USE_MULTILINGUAL = args.multilingual

EARLY_STOP_TOLERANCE = 0.01


def train_lstm_and_get_probs(
    train_texts: list[str],
    dev_texts: list[str],
    test_texts: list[str],
    train_Y: np.ndarray,
    dev_Y: np.ndarray,
    lang: str,
    subtask: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if USE_MULTILINGUAL:
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
    
    train_dataset = TensorDataset(
        torch.tensor(train_features, dtype=torch.float32),
        torch.tensor(train_Y, dtype=torch.long),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=LSTM_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    
    dev_tensor = torch.tensor(dev_features, dtype=torch.float32)
    dev_loader = DataLoader(
        TensorDataset(dev_tensor),
        batch_size=LSTM_BATCH_SIZE,
        shuffle=False,
    )
    
    optimizer = Adam(model.parameters(), lr=LSTM_LR)
    criterion = NLLLoss()
    
    print(f"[LSTM] Training for up to {LSTM_EPOCHS} epochs on {DEVICE}...")
    
    best_f1 = 0.0
    best_epoch = 0
    best_state = None
    
    for epoch in range(1, LSTM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        
        for features, labels in train_loader:
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            log_probs = model(features)
            loss = criterion(log_probs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        model.eval()
        dev_preds_list = []
        with torch.no_grad():
            for (batch,) in dev_loader:
                log_probs = model(batch.to(DEVICE))
                preds = torch.argmax(log_probs, dim=1)
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
    
    print("[LSTM] Extracting output probabilities...")
    model.eval()
    
    def get_probs_batched(features: np.ndarray) -> np.ndarray:
        all_probs = []
        dataset = TensorDataset(torch.tensor(features, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=LSTM_BATCH_SIZE, shuffle=False)
        
        with torch.no_grad():
            for (batch_features,) in loader:
                batch_features = batch_features.to(DEVICE)
                log_probs = model(batch_features)
                probs = torch.exp(log_probs)
                all_probs.append(probs.cpu().numpy())
        
        return np.concatenate(all_probs, axis=0)
    
    train_probs = get_probs_batched(train_features)
    dev_probs = get_probs_batched(dev_features)
    test_probs = get_probs_batched(test_features)
    
    print(f"[LSTM] Output probabilities shape: {train_probs.shape}")
    
    return train_probs, dev_probs, test_probs


def run_shap_analysis(
    clf,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str],
    subtask: str,
    lang: str,
    variant: str,
    class_names: list[str] = None,
) -> None:
    if not DO_SHAP:
        return
    
    print(f"\n[SHAP] Computing SHAP values for {subtask}/{lang}/{variant}...")
    
    # Sample for faster computation
    n_test = min(SHAP_SAMPLES, len(X_test))
    idx_test = np.random.choice(len(X_test), n_test, replace=False)
    X_test_sample = X_test[idx_test]
    
    # Create SHAP explainer and compute values
    explainer = shap.TreeExplainer(clf.model)
    shap_values = explainer.shap_values(X_test_sample)
    
    prefix = f"{subtask}_{lang}_{variant}_seed{SEED}"

    # Save SHAP values (plots generated separately via plot_shap.py)
    np.savez_compressed(
        SHAP_DIR / f"{prefix}_shap_values.npz",
        shap_values=np.array(shap_values) if isinstance(shap_values, list) else shap_values,
        feature_names=np.array(feature_names),
        X_test_sample=X_test_sample,
    )
    print(f"  Saved: {SHAP_DIR / f'{prefix}_shap_values.npz'}")
    print(f"  To generate plots, run: python scripts/plot_shap.py --subtask {subtask} --lang {lang} --variant {variant}")


def train_single_config(
    subtask: str,
    lang: str,
    model_variant: str,
) -> dict | None:
    print("\n" + "=" * 80)
    print(f"Training {model_variant.upper()} | subtask={subtask} | lang={lang} | seed={SEED}")
    print("=" * 80)

    set_determinism(SEED, numpy_seed=NUMPY_SEED)

    train_dir = DATA_DIR / "train" / subtask / lang
    test_dir = DATA_DIR / "test" / subtask / lang

    try:
        (
            train_texts, dev_texts, test_texts,
            train_Y, dev_Y, test_Y,
            train_idx, dev_idx
        ) = load_train_dev_test(
            train_dir=train_dir,
            test_dir=test_dir,
            subtask=subtask,
            seed=SEED,
            code_split=CODE_SPLIT,
            use_fold=USE_FOLD,
        )
    except FileNotFoundError as e:
        print(f"[SKIP] Data not found for {subtask}/{lang}: {e}")
        return None

    print(f"\n[DATA] Train: {len(train_texts)}, Dev: {len(dev_texts)}, Test: {len(test_texts)}")

    # Extract linguistic features
    print("\n[STEP] Extracting linguistic features...")
    ling_extractor = LinguisticFeatures(language=lang, resources_dir=RESOURCES_DIR)
    train_ling_X, ling_feature_names = ling_extractor.extract_features(train_texts)
    dev_ling_X, _ = ling_extractor.extract_features(dev_texts, feature_names=ling_feature_names)
    test_ling_X, _ = ling_extractor.extract_features(test_texts, feature_names=ling_feature_names)
    print(f"  Linguistic features: {train_ling_X.shape[1]}")

    # Extract style features (for lingrf_style and lingrf_style_predout variants)
    train_style_X = None
    dev_style_X = None
    test_style_X = None
    style_feature_names = []
    
    if model_variant in ["lingrf_style", "lingrf_style_predout"]:
        print("\n[STEP] Extracting style features...")
        style_extractor = StyleFeatures(language=lang)
        train_style_X, style_feature_names = style_extractor.extract(
            train_texts, cache_key=f"train_{subtask}_{lang}"
        )
        dev_style_X, _ = style_extractor.extract(
            dev_texts, cache_key=f"dev_{subtask}_{lang}"
        )
        test_style_X, _ = style_extractor.extract(
            test_texts, cache_key=f"test_{subtask}_{lang}"
        )
        print(f"  Style features: {train_style_X.shape[1]}")
        
        # Combine linguistic + style features
        train_X = np.concatenate([train_ling_X, train_style_X], axis=1)
        dev_X = np.concatenate([dev_ling_X, dev_style_X], axis=1)
        test_X = np.concatenate([test_ling_X, test_style_X], axis=1)
        feature_names = ling_feature_names + style_feature_names
        print(f"  Combined features: {train_X.shape[1]}")
    else:
        train_X = train_ling_X
        dev_X = dev_ling_X
        test_X = test_ling_X
        feature_names = ling_feature_names

    print(f"\nFeatures shape: Train={train_X.shape}, Dev={dev_X.shape}, Test={test_X.shape}")

    # Get LSTM probs for predout variants
    train_pred_probs = None
    dev_pred_probs = None
    test_pred_probs = None

    if model_variant in ["lingrf_predout", "lingrf_style_predout"]:
        print("\n[STEP] Training LSTM on predictability features...")
        train_pred_probs, dev_pred_probs, test_pred_probs = train_lstm_and_get_probs(
            train_texts=train_texts,
            dev_texts=dev_texts,
            test_texts=test_texts,
            train_Y=train_Y,
            dev_Y=dev_Y,
            lang=lang,
            subtask=subtask,
        )

    print("\n[STEP] Training Random Forest...")
    print(f"  n_estimators={N_ESTIMATORS}, max_depth={MAX_DEPTH}")

    if model_variant in ["lingrf", "lingrf_style"]:
        clf = LingRFClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            random_state=SEED,
        )
        clf.fit(train_X, train_Y, feature_names=feature_names)

        train_preds = clf.predict(train_X)
        dev_preds = clf.predict(dev_X)
        test_preds = clf.predict(test_X)

    else:  # lingrf_predout or lingrf_style_predout
        clf = LingRFPredOutClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            random_state=SEED,
        )
        clf.fit(train_X, train_pred_probs, train_Y, feature_names=feature_names)

        # Combine features for prediction
        train_X_full = np.concatenate([train_X, train_pred_probs], axis=1)
        dev_X_full = np.concatenate([dev_X, dev_pred_probs], axis=1)
        test_X_full = np.concatenate([test_X, test_pred_probs], axis=1)
        
        # Add prob feature names
        prob_names = [f"PRED_PROB_{i}" for i in range(train_pred_probs.shape[1])]
        feature_names = feature_names + prob_names

        train_preds = clf.predict(train_X, train_pred_probs)
        dev_preds = clf.predict(dev_X, dev_pred_probs)
        test_preds = clf.predict(test_X, test_pred_probs)
        
        train_X = train_X_full
        dev_X = dev_X_full
        test_X = test_X_full

    train_f1 = f1_score(train_Y, train_preds, average="macro")
    dev_f1 = f1_score(dev_Y, dev_preds, average="macro")
    test_f1 = f1_score(test_Y, test_preds, average="macro")

    print("\n[RESULTS]")
    print(f"  Train F1: {train_f1:.4f}")
    print(f"  Dev F1:   {dev_f1:.4f}")
    print(f"  Test F1:  {test_f1:.4f}")
    print(f"  Dev-Test Gap: {dev_f1 - test_f1:+.4f}")

    print("\n[FEATURE IMPORTANCE] Top 15 features:")
    top_features = clf.get_feature_importance(top_k=15)
    for i, (name, importance) in enumerate(top_features, 1):
        print(f"  {i:2d}. {name}: {importance:.4f}")

    # SHAP analysis
    class_names = None
    if subtask == "subtask_2":
        class_names = ["Human", "ChatGPT", "Cohere", "Davinci", "Bloomz", "Dolly"]
    
    run_shap_analysis(
        clf=clf,
        X_train=train_X,
        X_test=test_X,
        feature_names=feature_names,
        subtask=subtask,
        lang=lang,
        variant=model_variant,
        class_names=class_names,
    )

    # Save predictions
    preds_path = RESULTS_DIR / f"{subtask}_{lang}_{model_variant}_seed{SEED}_test_predictions.npz"
    np.savez(
        preds_path,
        train_preds=train_preds,
        dev_preds=dev_preds,
        test_preds=test_preds,
        feature_names=np.array(feature_names),
    )
    print(f"\n[INFO] Saved predictions to {preds_path}")

    tune_source = (
        "lingrf_predout"
        if model_variant in ["lingrf_predout", "lingrf_style_predout"]
        else "lingrf"
    )
    n_classes = 2 if subtask == "subtask_1" else 6
    tune_npz_name = f"{subtask}_{lang}_{model_variant}_seed{SEED}_lingrf_stage2_data.npz"
    tune_npz_path = TUNING_DIR / tune_npz_name
    np.savez(
        tune_npz_path,
        train_X=train_X,
        dev_X=dev_X,
        test_X=test_X,
        train_y=train_Y,
        dev_y=dev_Y,
        test_y=test_Y,
        subtask=subtask,
        lang=lang,
        n_classes=n_classes,
        seed=SEED,
        source=tune_source,
        variant=model_variant,
        multilingual=int(USE_MULTILINGUAL),
        feature_names=np.array(feature_names),
    )
    print(f"[INFO] Saved tuning NPZ: {tune_npz_path}")

    n_total_features = len(feature_names)

    return {
        "subtask": subtask,
        "lang": lang,
        "variant": model_variant,
        "seed": SEED,
        "train_f1": train_f1,
        "dev_f1": dev_f1,
        "test_f1": test_f1,
        "n_ling_features": len(ling_feature_names),
        "n_style_features": len(style_feature_names) if train_style_X is not None else 0,
        "n_total_features": n_total_features,
        "n_train": len(train_texts),
        "n_dev": len(dev_texts),
        "n_test": len(test_texts),
    }


def save_results_tsv(results: list[dict], output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("subtask\tlang\tvariant\tseed\ttrain_f1\tdev_f1\ttest_f1\tn_ling\tn_style\tn_total\n")
        for r in results:
            f.write(f"{r['subtask']}\t{r['lang']}\t{r['variant']}\t{r['seed']}\t"
                    f"{r['train_f1']:.4f}\t{r['dev_f1']:.4f}\t{r['test_f1']:.4f}\t"
                    f"{r['n_ling_features']}\t{r['n_style_features']}\t{r['n_total_features']}\n")
    print(f"\n[INFO] Results saved to {output_path}")


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"lingrf_style_{timestamp}.log"
    tee = Tee(log_file)
    sys.stdout = tee

    print("=" * 80)
    print("LingRF + Style Features + SHAP Analysis")
    print("=" * 80)
    print(f"Subtasks: {SUBTASKS}")
    print(f"Languages: {LANGUAGES}")
    print(f"Variants: {VARIANTS}")
    print(f"SHAP enabled: {DO_SHAP}")
    if DO_SHAP:
        print(f"SHAP samples: {SHAP_SAMPLES}")
    print(f"Device: {DEVICE}")
    print(f"Timestamp: {timestamp}")
    print(f"Seed: {SEED}")
    print("=" * 80)
    log_env_fingerprint()

    results = []

    for subtask in SUBTASKS:
        for lang in LANGUAGES:
            for variant in VARIANTS:
                result = train_single_config(
                    subtask=subtask,
                    lang=lang,
                    model_variant=variant,
                )
                if result:
                    results.append(result)

    print("\n" + "=" * 110)
    print("FINAL SUMMARY - TEST SET RESULTS")
    print("=" * 110)
    
    print(f"\n{'Subtask':<12} {'Lang':<6} {'Variant':<18} {'Seed':<6} {'Train F1':<10} {'Dev F1':<10} {'Test F1':<10} {'Ling':<6} {'Style':<6} {'Total':<6}")
    print("-" * 110)

    for r in results:
        print(f"{r['subtask']:<12} {r['lang']:<6} {r['variant']:<18} {r['seed']:<6} "
              f"{r['train_f1']:<10.4f} {r['dev_f1']:<10.4f} {r['test_f1']:<10.4f} "
              f"{r['n_ling_features']:<6} {r['n_style_features']:<6} {r['n_total_features']:<6}")

    results_path = RESULTS_DIR / f"lingrf_style_results_{timestamp}.tsv"
    save_results_tsv(results, results_path)

    latest_path = RESULTS_DIR / "lingrf_style_results_latest.tsv"
    save_results_tsv(results, latest_path)

    if DO_SHAP:
        print(f"\n[SHAP] Plots saved to: {SHAP_DIR}")

    print("\n" + "=" * 110)
    print("All training complete!")
    print("=" * 110)

    tee.close()


if __name__ == "__main__":
    main()
