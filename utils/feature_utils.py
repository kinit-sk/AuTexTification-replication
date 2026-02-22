"""Feature extraction orchestration."""

from pathlib import Path

import numpy as np
import torch

from feature_extraction.probabilistic_features import (
    ProbabilisticFeatures,
    MultilingualProbFeatures,
    fixed_len,
)
from feature_extraction.grammar_features import GrammarFeatures, WordFrequency


def _compute_prob(
    train_texts: list[str],
    dev_texts: list[str],
    test_texts: list[str],
    device: torch.device,
    local_device: torch.device,
    lang: str,
    multilingual: bool,
    prob_disabled: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute probabilistic features from scratch every time."""
    if multilingual:
        n_models = 4
    elif lang == "en":
        n_models = 4
    else:
        n_models = 2
    n_features = n_models * 3 + 1

    if prob_disabled:
        zeros = lambda texts: np.zeros((len(texts), fixed_len, n_features), dtype=np.float32)
        print("[FEATURES] Prob disabled (flm variant) — using zeros")
        return zeros(train_texts), zeros(dev_texts), zeros(test_texts)

    if multilingual:
        print("[FEATURES] Using MULTILINGUAL probabilistic models")
        prob_extractor = MultilingualProbFeatures(device)
    else:
        prob_extractor = ProbabilisticFeatures(device, local_device, lang, disabled=False)

    train_prob = np.array(prob_extractor.word_features(train_texts))
    dev_prob = np.array(prob_extractor.word_features(dev_texts))
    test_prob = np.array(prob_extractor.word_features(test_texts))

    return train_prob, dev_prob, test_prob


def compute_all_features(
    train_texts,
    dev_texts,
    test_texts,
    train_idx,
    dev_idx,
    subtask: str,
    lang: str,
    device: torch.device,
    local_device: torch.device,
    model_variant: str,
    features_dir: Path,
    multilingual: bool = False,
):
    """Compute features for train/dev/test. Returns train_X, dev_X, test_X, fixed_len."""

    assert model_variant in {"pred", "flm", "pred_flm", "pred_flm_add"}

    use_additional = model_variant == "pred_flm_add"

    if use_additional:
        grammar_train_path = features_dir / f"train_{subtask}_{lang}_grammar.npy"
        grammar_test_path = features_dir / f"test_{subtask}_{lang}_grammar.npy"

        precomp_train_grammar = (
            np.load(grammar_train_path) if grammar_train_path.exists() else None
        )
        precomp_test_grammar = (
            np.load(grammar_test_path) if grammar_test_path.exists() else None
        )

        grammar_holder = {"extractor": None}

        def get_grammar_features(split, texts, idx=None):
            if split in ("train", "dev") and precomp_train_grammar is not None:
                return precomp_train_grammar[idx]

            if split == "test" and precomp_test_grammar is not None:
                return precomp_test_grammar

            if grammar_holder["extractor"] is None:
                grammar_holder["extractor"] = GrammarFeatures(device, local_device, lang)

            return np.array(grammar_holder["extractor"].word_features(texts))

        freq_extractor = WordFrequency(device, local_device, lang)
    else:
        get_grammar_features = None
        freq_extractor = None

    prob_disabled = model_variant == "flm"

    train_prob, dev_prob, test_prob = _compute_prob(
        train_texts, dev_texts, test_texts,
        device, local_device, lang,
        multilingual, prob_disabled,
    )

    if use_additional:
        train_freq = np.array(freq_extractor.word_features(train_texts))
        train_gram = get_grammar_features("train", train_texts, idx=train_idx)
        train_X = np.concatenate([train_prob, train_freq, train_gram], axis=2)
    else:
        train_X = train_prob

    if use_additional:
        dev_freq = np.array(freq_extractor.word_features(dev_texts))
        dev_gram = get_grammar_features("dev", dev_texts, idx=dev_idx)
        dev_X = np.concatenate([dev_prob, dev_freq, dev_gram], axis=2)
    else:
        dev_X = dev_prob

    if use_additional:
        test_freq = np.array(freq_extractor.word_features(test_texts))
        test_gram = get_grammar_features("test", test_texts)
        test_X = np.concatenate([test_prob, test_freq, test_gram], axis=2)
    else:
        test_X = test_prob

    return train_X, dev_X, test_X, fixed_len
