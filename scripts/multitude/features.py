"""Feature extraction and neural prediction helpers for MULTITuDE."""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from feature_extraction import style_features as style_feature_module
from feature_extraction.style_features import (
    FIRST_PERSON_EN,
    FIRST_PERSON_ES,
    FORMAL_WORDS_EN,
    FORMAL_WORDS_ES,
    FUNCTION_WORDS_EN,
    FUNCTION_WORDS_ES,
    HEDGE_WORDS_EN,
    HEDGE_WORDS_ES,
    TRANSITION_WORDS_EN,
    TRANSITION_WORDS_ES,
)
from scripts.multitude.config import (
    FloatArray,
    IntArray,
    MULTITUDE_CHECKPOINT_DIR,
    MultitudeSplits,
    ProbabilitySplits,
    STYLE_FEATURE_NAMES,
    set_seeds,
)
from scripts.multitude.models import MultitudeHybridBiLSTMEncoder, MultitudePredLSTM
from utils.constants import DEVICE, LOCAL_DEVICE
from utils.training_pipeline import train_and_evaluate


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def ngram_repetition(tokens: list[str], n: int) -> float:
    if len(tokens) < n:
        return 0.0
    ngrams: list[tuple[str, ...]] = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    counts: Counter[tuple[str, ...]] = Counter(ngrams)
    repeated_count: int = sum(count - 1 for count in counts.values() if count > 1)
    return safe_divide(float(repeated_count), float(len(ngrams)))


def sentence_parts(text: str) -> list[str]:
    sentences: list[str] = re.split(r"[.!?]+", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def token_list(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def word_sets_for_language(
    language: str,
) -> tuple[set[str], set[str], set[str], set[str], set[str]]:
    if language == "en":
        return (
            set(TRANSITION_WORDS_EN),
            set(HEDGE_WORDS_EN),
            set(FUNCTION_WORDS_EN),
            set(FIRST_PERSON_EN),
            set(FORMAL_WORDS_EN),
        )
    if language == "es":
        return (
            set(TRANSITION_WORDS_ES),
            set(HEDGE_WORDS_ES),
            set(FUNCTION_WORDS_ES),
            set(FIRST_PERSON_ES),
            set(FORMAL_WORDS_ES),
        )
    return set(), set(), set(), set(), set()


def ttr(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def root_ttr(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    return len(set(tokens)) / math.sqrt(len(tokens))


def log_ttr(tokens: list[str]) -> float:
    if len(tokens) < 2:
        return 0.0
    types_count: int = len(set(tokens))
    if types_count < 2:
        return 0.0
    return math.log(types_count) / math.log(len(tokens))


def hapax_ratio(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    counts: Counter[str] = Counter(tokens)
    hapax_count: int = sum(1 for count in counts.values() if count == 1)
    return hapax_count / len(tokens)


def dis_legomena_ratio(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    counts: Counter[str] = Counter(tokens)
    dis_count: int = sum(1 for count in counts.values() if count == 2)
    return dis_count / len(tokens)


def sentence_stats(sentences: list[str]) -> tuple[float, float, float]:
    if not sentences:
        return 0.0, 0.0, 0.0
    lengths: list[int] = [len(token_list(sentence)) for sentence in sentences]
    if not lengths:
        return 0.0, 0.0, 0.0
    avg_length: float = float(np.mean(lengths))
    std_length: float = float(np.std(lengths))
    cv_length: float = safe_divide(std_length, avg_length)
    return avg_length, std_length, cv_length


def count_syllables(word: str) -> int:
    vowels: str = "aeiouyáéíóúàèìòùüñ"
    count: int = sum(1 for char in word.lower() if char in vowels)
    return max(1, count)


def flesch_reading_ease(text: str) -> float:
    if style_feature_module.HAS_TEXTSTAT:
        try:
            return float(style_feature_module.textstat.flesch_reading_ease(text))
        except Exception:
            pass

    sentences: list[str] = sentence_parts(text)
    tokens: list[str] = token_list(text)
    if not sentences or not tokens:
        return 0.0

    total_syllables: int = sum(count_syllables(token) for token in tokens)
    avg_sentence_len: float = len(tokens) / len(sentences)
    avg_syllables: float = total_syllables / len(tokens)
    return 206.835 - 1.015 * avg_sentence_len - 84.6 * avg_syllables


def flesch_kincaid_grade(text: str) -> float:
    if style_feature_module.HAS_TEXTSTAT:
        try:
            return float(style_feature_module.textstat.flesch_kincaid_grade(text))
        except Exception:
            pass

    sentences: list[str] = sentence_parts(text)
    tokens: list[str] = token_list(text)
    if not sentences or not tokens:
        return 0.0

    total_syllables: int = sum(count_syllables(token) for token in tokens)
    avg_sentence_len: float = len(tokens) / len(sentences)
    avg_syllables: float = total_syllables / len(tokens)
    return 0.39 * avg_sentence_len + 11.8 * avg_syllables - 15.59


def rare_word_burstiness(tokens: list[str]) -> float:
    if len(tokens) < 10:
        return 0.0
    counts: Counter[str] = Counter(tokens)
    rare_positions: list[int] = [idx for idx, token in enumerate(tokens) if counts[token] <= 2]
    if len(rare_positions) < 2:
        return 0.0
    gaps: list[int] = [
        rare_positions[idx + 1] - rare_positions[idx]
        for idx in range(len(rare_positions) - 1)
    ]
    if not gaps:
        return 0.0
    mean_gap: float = float(np.mean(gaps))
    std_gap: float = float(np.std(gaps))
    if mean_gap + std_gap == 0.0:
        return 0.0
    return (std_gap - mean_gap) / (std_gap + mean_gap)


def avg_word_length(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    return float(np.mean([len(token) for token in tokens]))


def word_length_std(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    return float(np.std([len(token) for token in tokens]))


def punctuation_ratio(text: str) -> float:
    if not text:
        return 0.0
    punctuation_count: int = sum(1 for char in text if char in '.,;:!?-()[]{}"\'/\\')
    return punctuation_count / len(text)


def ratio_per_sentence(text: str, marker: str) -> float:
    sentences: list[str] = sentence_parts(text)
    if not sentences:
        return 0.0
    return text.count(marker) / len(sentences)


def word_set_ratio(tokens: list[str], words: set[str]) -> float:
    if not tokens or not words:
        return 0.0
    count: int = sum(1 for token in tokens if token in words)
    return count / len(tokens)


def extract_style_values(text: str, language: str) -> list[float]:
    tokens: list[str] = token_list(text)
    sentences: list[str] = sentence_parts(text)
    avg_sentence_length, sentence_length_std, sentence_length_cv = sentence_stats(sentences)
    transition_words, hedge_words, function_words, first_person, formal_words = (
        word_sets_for_language(language=language)
    )

    return [
        ttr(tokens),
        root_ttr(tokens),
        log_ttr(tokens),
        hapax_ratio(tokens),
        dis_legomena_ratio(tokens),
        avg_sentence_length,
        sentence_length_std,
        sentence_length_cv,
        float(len(sentences)),
        ngram_repetition(tokens=tokens, n=2),
        ngram_repetition(tokens=tokens, n=3),
        avg_word_length(tokens),
        word_length_std(tokens),
        float(len(tokens)),
        word_set_ratio(tokens=tokens, words=function_words),
        word_set_ratio(tokens=tokens, words=transition_words),
        word_set_ratio(tokens=tokens, words=hedge_words),
        flesch_reading_ease(text),
        flesch_kincaid_grade(text),
        punctuation_ratio(text),
        ratio_per_sentence(text=text, marker=","),
        rare_word_burstiness(tokens),
        ratio_per_sentence(text=text, marker="!"),
        ratio_per_sentence(text=text, marker="?"),
        word_set_ratio(tokens=tokens, words=first_person),
        word_set_ratio(tokens=tokens, words=formal_words),
    ]


def extract_multilingual_style_features(
    texts: list[str],
    languages: list[str],
) -> tuple[FloatArray, list[str]]:
    if len(texts) != len(languages):
        raise ValueError(
            f"Texts length {len(texts)} does not match languages length {len(languages)}."
        )
    features: FloatArray = np.zeros((len(texts), len(STYLE_FEATURE_NAMES)), dtype=np.float32)
    pairs: list[tuple[str, str]] = list(zip(texts, languages))
    for row_idx, (text, language) in enumerate(tqdm(pairs, desc="Style features", ascii=True)):
        values: list[float] = extract_style_values(text=text, language=language)
        features[row_idx, :] = np.asarray(values, dtype=np.float32)
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0), list(STYLE_FEATURE_NAMES)

def make_sequence_loader(
    features: FloatArray,
    labels: IntArray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    pin_memory: bool = torch.cuda.is_available()
    dataset: TensorDataset = TensorDataset(
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=pin_memory,
    )


def extract_lstm_probabilities(
    model: MultitudePredLSTM,
    features: FloatArray,
    batch_size: int,
    device: torch.device,
) -> FloatArray:
    loader: DataLoader = make_sequence_loader(
        features=features,
        labels=np.zeros(len(features), dtype=np.int64),
        batch_size=batch_size,
        shuffle=False,
    )
    model.eval()
    probabilities: list[FloatArray] = []
    with torch.no_grad():
        for (batch_features, _) in loader:
            log_probs: torch.Tensor = model(batch_features.to(device, non_blocking=True))
            probabilities.append(torch.exp(log_probs).cpu().numpy().astype(np.float32))
    return np.concatenate(probabilities, axis=0)


def train_predout_lstm(
    splits: MultitudeSplits,
    prob_features: ProbabilitySplits,
    seed: int,
    batch_size: int,
    lstm_epochs: int,
    num_classes: int,
) -> tuple[FloatArray, FloatArray, FloatArray, int, float, float]:
    set_seeds(seed=seed)

    train_loader: DataLoader = make_sequence_loader(
        features=prob_features.train,
        labels=splits.train_y,
        batch_size=batch_size,
        shuffle=True,
    )
    dev_loader: DataLoader = make_sequence_loader(
        features=prob_features.dev,
        labels=splits.dev_y,
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader: DataLoader = make_sequence_loader(
        features=prob_features.test,
        labels=splits.test_y,
        batch_size=batch_size,
        shuffle=False,
    )

    model: MultitudePredLSTM = MultitudePredLSTM(
        seq_feature_len=prob_features.train.shape[2],
        num_classes=num_classes,
        local_device=LOCAL_DEVICE,
        hidden_size=64,
        bidirectional=True,
    ).to(DEVICE)

    checkpoint_prefix: str = f"multitude_predout_lstm_seed{seed}"
    result = train_and_evaluate(
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=test_loader,
        model=model,
        model_variant="pred",
        device=DEVICE,
        out_dir=MULTITUDE_CHECKPOINT_DIR,
        checkpoint_prefix=checkpoint_prefix,
        epochs=lstm_epochs,
        freeze_epochs=0,
        cleanup_non_best=True,
    )

    train_probs: FloatArray = extract_lstm_probabilities(
        model=model,
        features=prob_features.train,
        batch_size=batch_size,
        device=DEVICE,
    )
    dev_probs: FloatArray = extract_lstm_probabilities(
        model=model,
        features=prob_features.dev,
        batch_size=batch_size,
        device=DEVICE,
    )
    test_probs: FloatArray = extract_lstm_probabilities(
        model=model,
        features=prob_features.test,
        batch_size=batch_size,
        device=DEVICE,
    )

    return train_probs, dev_probs, test_probs, result.best_epoch, result.dev_f1, result.test_f1


def get_tokenizer(encoder_id: str):
    print(f"\n[TOK] Loading tokenizer: {encoder_id}")
    if "deberta" in encoder_id.lower():
        from transformers import DebertaV2Tokenizer

        return DebertaV2Tokenizer.from_pretrained(encoder_id)

    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(encoder_id, use_fast=True)


def extract_hybrid_probabilities(
    model: MultitudeHybridBiLSTMEncoder,
    seq_features: FloatArray,
    encodings: dict[str, torch.Tensor],
    labels: IntArray,
    batch_size: int,
    device: torch.device,
) -> FloatArray:
    dataset: TensorDataset = TensorDataset(
        torch.tensor(seq_features, dtype=torch.float32),
        encodings["input_ids"],
        encodings["attention_mask"],
        torch.tensor(labels, dtype=torch.long),
    )
    loader: DataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model.eval()
    probabilities: list[FloatArray] = []
    with torch.no_grad():
        for batch in loader:
            moved_batch: list[torch.Tensor] = [item.to(device, non_blocking=True) for item in batch]
            log_probs: torch.Tensor = model(*moved_batch[:-1])
            probabilities.append(torch.exp(log_probs).cpu().numpy().astype(np.float32))
    return np.concatenate(probabilities, axis=0)
