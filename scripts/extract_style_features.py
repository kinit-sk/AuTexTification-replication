"""Extracts and saves cached style-feature matrices for train/dev/test splits."""

from __future__ import annotations

import argparse

from _bootstrap import configure_project_root

configure_project_root(__file__, remove_shadowing_utils=False)

import numpy as np

from feature_extraction.style_features import StyleFeatures
from utils.constants import DATA_DIR, DEFAULT_SEED, FEATURES_DIR
from utils.data_utils import load_train_dev_test

SEED = DEFAULT_SEED
FEATURES_DIR.mkdir(parents=True, exist_ok=True)


def extract_and_save(subtask: str, lang: str):
    print(f"\n{'='*60}")
    print(f"Extracting style features: {subtask} / {lang}")
    print(f"{'='*60}")

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
            code_split=False,
            use_fold=0,
        )
    except FileNotFoundError as e:
        print(f"[SKIP] Data not found: {e}")
        return

    print(f"[DATA] Train: {len(train_texts)}, Dev: {len(dev_texts)}, Test: {len(test_texts)}")

    extractor = StyleFeatures(language=lang)

    print("\n[STEP] Extracting train style features...")
    train_features, feature_names = extractor.extract(
        train_texts,
        cache_key=f"train_{subtask}_{lang}",
        use_cache=True,
    )

    print("\n[STEP] Extracting dev style features...")
    dev_features, _ = extractor.extract(
        dev_texts,
        cache_key=f"dev_{subtask}_{lang}",
        use_cache=True,
    )

    print("\n[STEP] Extracting test style features...")
    test_features, _ = extractor.extract(
        test_texts,
        cache_key=f"test_{subtask}_{lang}",
        use_cache=True,
    )

    train_path = FEATURES_DIR / f"train_{subtask}_{lang}_style.npz"
    dev_path = FEATURES_DIR / f"dev_{subtask}_{lang}_style.npz"
    test_path = FEATURES_DIR / f"test_{subtask}_{lang}_style.npz"

    print("\n[STEP] Saving features...")

    np.savez_compressed(
        train_path,
        features=train_features,
        feature_names=np.array(feature_names),
        indices=train_idx,
    )
    print(f"  Saved: {train_path}")
    print(f"  Shape: {train_features.shape}")

    np.savez_compressed(
        dev_path,
        features=dev_features,
        feature_names=np.array(feature_names),
        indices=dev_idx,
    )
    print(f"  Saved: {dev_path}")
    print(f"  Shape: {dev_features.shape}")

    np.savez_compressed(
        test_path,
        features=test_features,
        feature_names=np.array(feature_names),
    )
    print(f"  Saved: {test_path}")
    print(f"  Shape: {test_features.shape}")

    print(f"\n[DONE] {subtask}/{lang} - {len(feature_names)} features extracted")
    print(f"Feature names: {feature_names}")


def load_style_features(subtask: str, lang: str, split: str = "train"):
    """Load previously saved style features for a split."""
    path = FEATURES_DIR / f"{split}_{subtask}_{lang}_style.npz"
    if not path.exists():
        raise FileNotFoundError(f"Style features not found: {path}")

    data = np.load(path, allow_pickle=True)
    result = {"features": data["features"], "feature_names": list(data["feature_names"])}
    if "indices" in data:
        result["indices"] = data["indices"]
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Extract and save style features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_style_features.py --subtask subtask_1 --lang en
  python extract_style_features.py --subtask all --lang all
  python extract_style_features.py --subtask subtask_1 --lang all
        """,
    )

    parser.add_argument(
        "--subtask",
        type=str,
        choices=["subtask_1", "subtask_2", "all"],
        default="subtask_1",
        help="Subtask (default: subtask_1)",
    )
    parser.add_argument(
        "--lang",
        type=str,
        choices=["en", "es", "all"],
        default="en",
        help="Language (default: en)",
    )

    args = parser.parse_args()

    subtasks = ["subtask_1", "subtask_2"] if args.subtask == "all" else [args.subtask]
    languages = ["en", "es"] if args.lang == "all" else [args.lang]

    print("=" * 60)
    print("STYLE FEATURE EXTRACTION")
    print("=" * 60)
    print(f"Subtasks: {subtasks}")
    print(f"Languages: {languages}")
    print(f"Output dir: {FEATURES_DIR}")

    for subtask in subtasks:
        for lang in languages:
            extract_and_save(subtask, lang)

    print("\n" + "=" * 60)
    print("ALL DONE!")
    print("=" * 60)
    print(f"\nFeatures saved to: {FEATURES_DIR}")
    print("\nTo load in your code:")
    print("  from scripts.extract_style_features import load_style_features")
    print("  data = load_style_features('subtask_1', 'en', 'train')")
    print("  features = data['features']  # shape: (n_samples, 22)")


if __name__ == "__main__":
    main()
