"""Shared ablation constants."""

BASELINE_GROUP_NAME: str = ""
LINGRF_STYLE_VARIANT: str = "lingrf_style"
LINGRF_STYLE_PREDOUT_VARIANT: str = "lingrf_style_predout"
ABLATION_VARIANTS: list[str] = [LINGRF_STYLE_VARIANT, LINGRF_STYLE_PREDOUT_VARIANT]
TASK_LANG_PAIRS: tuple[tuple[str, str], ...] = (
    ("subtask_1", "en"),
    ("subtask_1", "es"),
    ("subtask_2", "en"),
    ("subtask_2", "es"),
)
STYLE_GROUPS: dict[str, list[str]] = {
    "LexicalDiversity": [
        "ttr",
        "root_ttr",
        "log_ttr",
        "hapax_ratio",
        "dis_legomena_ratio",
        "rare_word_burstiness",
    ],
    "SentenceStructure": [
        "avg_sentence_length",
        "sentence_length_std",
        "sentence_length_cv",
        "sentence_count",
    ],
    "RepetitionPatterns": [
        "bigram_repetition",
        "trigram_repetition",
    ],
    "WordLevelStatistics": [
        "avg_word_length",
        "word_length_std",
        "word_count",
    ],
    "FunctionalStylisticMarkers": [
        "function_word_ratio",
        "transition_word_ratio",
        "hedge_word_ratio",
        "first_person_ratio",
        "formal_word_ratio",
    ],
    "ReadabilityMetrics": [
        "flesch_reading_ease",
        "flesch_kincaid_grade",
    ],
    "PunctuationUsage": [
        "punctuation_ratio",
        "comma_ratio",
        "exclamation_ratio",
        "question_ratio",
    ],
}
STYLE_GROUP_NAMES: list[str] = [BASELINE_GROUP_NAME, *STYLE_GROUPS.keys()]
STYLE_FEATURE_SET: set[str] = {
    feature_name
    for group_features in STYLE_GROUPS.values()
    for feature_name in group_features
}
GROUP_ORDER: list[str] = ["Linguistic", *STYLE_GROUPS.keys(), "LSTM_Probs"]
SUBTASK1_CLASS_NAMES: list[str] = ["Human", "AI"]
SUBTASK2_CLASS_NAMES: list[str] = ["A", "B", "C", "D", "E", "F"]


def class_names_for_subtask(subtask: str) -> list[str]:
    if subtask == "subtask_1":
        return SUBTASK1_CLASS_NAMES
    if subtask == "subtask_2":
        return SUBTASK2_CLASS_NAMES
    raise ValueError(f"Unknown subtask: {subtask}")
