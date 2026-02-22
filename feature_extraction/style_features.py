import os
import re
import math
import hashlib
import numpy as np
from pathlib import Path
from collections import Counter
from typing import Optional

try:
    import textstat
    HAS_TEXTSTAT = True
except ImportError:
    HAS_TEXTSTAT = False
    print("[StyleFeatures] textstat not installed, using manual readability formulas")


TRANSITION_WORDS_EN = {
    "however", "therefore", "moreover", "furthermore", "nevertheless",
    "consequently", "additionally", "meanwhile", "nonetheless", "thus",
    "hence", "accordingly", "subsequently", "otherwise", "instead",
    "likewise", "similarly", "conversely", "alternatively", "finally",
    "firstly", "secondly", "thirdly", "lastly", "notably", "specifically",
    "particularly", "indeed", "certainly", "undoubtedly", "clearly",
}

TRANSITION_WORDS_ES = {
    "sin embargo", "por lo tanto", "además", "no obstante", "por consiguiente",
    "asimismo", "mientras tanto", "así", "entonces", "por ende",
    "de hecho", "en consecuencia", "posteriormente", "de lo contrario",
    "igualmente", "similarmente", "por el contrario", "alternativamente",
    "finalmente", "primeramente", "en segundo lugar", "específicamente",
    "particularmente", "ciertamente", "claramente", "obviamente",
}

HEDGE_WORDS_EN = {
    "maybe", "perhaps", "probably", "possibly", "likely", "unlikely",
    "might", "could", "would", "should", "seem", "seems", "appeared",
    "apparently", "presumably", "supposedly", "allegedly", "somewhat",
    "rather", "fairly", "quite", "almost", "nearly", "roughly",
    "approximately", "generally", "usually", "often", "sometimes",
    "occasionally", "rarely", "seldom", "tend", "tends", "suggest",
    "suggests", "indicate", "indicates", "imply", "implies",
}

HEDGE_WORDS_ES = {
    "quizás", "tal vez", "probablemente", "posiblemente", "quizá",
    "podría", "debería", "parece", "aparentemente", "presumiblemente",
    "supuestamente", "algo", "bastante", "casi", "aproximadamente",
    "generalmente", "usualmente", "frecuentemente", "a veces",
    "ocasionalmente", "raramente", "sugiere", "indica", "implica",
}

FUNCTION_WORDS_EN = {
    "the", "a", "an", "and", "or", "but", "if", "then", "because", "as",
    "until", "while", "of", "at", "by", "for", "with", "about", "against",
    "between", "into", "through", "during", "before", "after", "above",
    "below", "to", "from", "up", "down", "in", "out", "on", "off", "over",
    "under", "again", "further", "once", "here", "there", "when", "where",
    "why", "how", "all", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "can", "will", "just", "should", "now", "i", "me", "my", "we",
    "our", "you", "your", "he", "him", "his", "she", "her", "it", "its",
    "they", "them", "their", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "would",
    "could", "ought", "must", "shall", "may", "might", "need",
}

FUNCTION_WORDS_ES = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "y", "o", "pero",
    "si", "porque", "como", "cuando", "donde", "que", "quien", "cual",
    "de", "a", "en", "con", "por", "para", "sin", "sobre", "entre", "hacia",
    "desde", "hasta", "según", "durante", "mediante", "yo", "tú", "él",
    "ella", "nosotros", "vosotros", "ellos", "ellas", "me", "te", "se",
    "nos", "os", "le", "les", "lo", "mi", "tu", "su", "nuestro", "vuestro",
    "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas",
    "aquel", "aquella", "aquellos", "aquellas", "ser", "estar", "haber",
    "tener", "hacer", "poder", "deber", "ir", "venir", "ver", "dar",
    "saber", "querer", "muy", "más", "menos", "tan", "tanto", "mucho",
    "poco", "todo", "nada", "algo", "alguien", "nadie", "ninguno", "otro",
}

FIRST_PERSON_EN = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"}
FIRST_PERSON_ES = {"yo", "me", "mi", "mío", "mía", "nosotros", "nosotras", "nos", "nuestro", "nuestra"}

FORMAL_WORDS_EN = {
    "therefore", "thus", "hence", "consequently", "furthermore", "moreover",
    "nevertheless", "nonetheless", "whereas", "whereby", "wherein", "thereof",
    "herein", "notwithstanding", "aforementioned", "henceforth", "thereby",
    "accordingly", "subsequently", "preceding", "following", "regarding",
    "concerning", "pertaining", "constitute", "demonstrate", "indicate",
    "illustrate", "emphasize", "acknowledge", "facilitate", "implement",
    "utilize", "obtain", "acquire", "establish", "maintain", "ensure",
    "significant", "substantial", "considerable", "comprehensive", "extensive",
    "fundamental", "essential", "crucial", "critical", "paramount",
    "approximately", "predominantly", "primarily", "particularly", "specifically",
    "explicitly", "implicitly", "inherently", "respectively", "ultimately",
}

FORMAL_WORDS_ES = {
    "por lo tanto", "así", "por ende", "consecuentemente", "además", "asimismo",
    "sin embargo", "no obstante", "mientras que", "mediante", "respecto",
    "concerniente", "constituir", "demostrar", "indicar", "ilustrar",
    "enfatizar", "reconocer", "facilitar", "implementar", "utilizar",
    "obtener", "adquirir", "establecer", "mantener", "asegurar",
    "significativo", "sustancial", "considerable", "comprensivo", "extenso",
    "fundamental", "esencial", "crucial", "crítico", "primordial",
    "aproximadamente", "predominantemente", "principalmente", "particularmente",
    "específicamente", "explícitamente", "implícitamente", "inherentemente",
}


class StyleFeatures:
    def __init__(self, language: str = "en", cache_dir: Optional[Path] = None):
        self.language = language

        if language == "en":
            self.transition_words = TRANSITION_WORDS_EN
            self.hedge_words = HEDGE_WORDS_EN
            self.function_words = FUNCTION_WORDS_EN
            self.first_person = FIRST_PERSON_EN
            self.formal_words = FORMAL_WORDS_EN
        else:
            self.transition_words = TRANSITION_WORDS_ES
            self.hedge_words = HEDGE_WORDS_ES
            self.function_words = FUNCTION_WORDS_ES
            self.first_person = FIRST_PERSON_ES
            self.formal_words = FORMAL_WORDS_ES

        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent / "data" / "features" / "style_cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, texts_hash: str, suffix: str = "") -> Path:
        return self.cache_dir / f"style_{self.language}_{texts_hash}{suffix}.npz"

    def _compute_texts_hash(self, texts: list[str]) -> str:
        content = "\n".join(texts[:100])
        content += f"_n{len(texts)}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r'\b\w+\b', text.lower())

    def _get_sentences(self, text: str) -> list[str]:
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _ttr(self, tokens: list[str]) -> float:
        if not tokens:
            return 0.0
        return len(set(tokens)) / len(tokens)

    def _root_ttr(self, tokens: list[str]) -> float:
        if not tokens:
            return 0.0
        return len(set(tokens)) / math.sqrt(len(tokens))

    def _log_ttr(self, tokens: list[str]) -> float:
        if len(tokens) < 2:
            return 0.0
        types = len(set(tokens))
        if types < 2:
            return 0.0
        return math.log(types) / math.log(len(tokens))

    def _hapax_ratio(self, tokens: list[str]) -> float:
        if not tokens:
            return 0.0
        freq = Counter(tokens)
        hapax = sum(1 for w, c in freq.items() if c == 1)
        return hapax / len(tokens)

    def _dis_legomena_ratio(self, tokens: list[str]) -> float:
        if not tokens:
            return 0.0
        freq = Counter(tokens)
        dis = sum(1 for w, c in freq.items() if c == 2)
        return dis / len(tokens)

    def _sentence_stats(self, sentences: list[str]) -> tuple[float, float, float]:
        if not sentences:
            return 0.0, 0.0, 0.0

        lengths = [len(self._tokenize(s)) for s in sentences]
        if not lengths:
            return 0.0, 0.0, 0.0

        avg = np.mean(lengths)
        std = np.std(lengths)
        cv = std / avg if avg > 0 else 0.0

        return float(avg), float(std), float(cv)

    def _ngram_repetition(self, tokens: list[str], n: int) -> float:
        if len(tokens) < n:
            return 0.0

        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        if not ngrams:
            return 0.0

        freq = Counter(ngrams)
        repeated = sum(c - 1 for c in freq.values() if c > 1)
        return repeated / len(ngrams)

    def _function_word_ratio(self, tokens: list[str]) -> float:
        if not tokens:
            return 0.0
        func_count = sum(1 for t in tokens if t in self.function_words)
        return func_count / len(tokens)

    def _transition_word_ratio(self, tokens: list[str]) -> float:
        if not tokens:
            return 0.0
        trans_count = sum(1 for t in tokens if t in self.transition_words)
        return trans_count / len(tokens)

    def _hedge_word_ratio(self, tokens: list[str]) -> float:
        if not tokens:
            return 0.0
        hedge_count = sum(1 for t in tokens if t in self.hedge_words)
        return hedge_count / len(tokens)

    def _flesch_reading_ease(self, text: str) -> float:
        if HAS_TEXTSTAT:
            try:
                return textstat.flesch_reading_ease(text)
            except:
                pass

        sentences = self._get_sentences(text)
        words = self._tokenize(text)
        if not sentences or not words:
            return 0.0

        def count_syllables(word):
            vowels = "aeiouyáéíóúàèìòù"
            count = sum(1 for c in word.lower() if c in vowels)
            return max(1, count)

        total_syllables = sum(count_syllables(w) for w in words)
        avg_sentence_len = len(words) / len(sentences)
        avg_syllables = total_syllables / len(words)

        return 206.835 - 1.015 * avg_sentence_len - 84.6 * avg_syllables

    def _flesch_kincaid_grade(self, text: str) -> float:
        if HAS_TEXTSTAT:
            try:
                return textstat.flesch_kincaid_grade(text)
            except:
                pass

        sentences = self._get_sentences(text)
        words = self._tokenize(text)
        if not sentences or not words:
            return 0.0

        def count_syllables(word):
            vowels = "aeiouyáéíóúàèìòù"
            count = sum(1 for c in word.lower() if c in vowels)
            return max(1, count)

        total_syllables = sum(count_syllables(w) for w in words)
        avg_sentence_len = len(words) / len(sentences)
        avg_syllables = total_syllables / len(words)

        return 0.39 * avg_sentence_len + 11.8 * avg_syllables - 15.59

    def _rare_word_burstiness(self, tokens: list[str]) -> float:
        if len(tokens) < 10:
            return 0.0

        freq = Counter(tokens)
        rare_positions = [i for i, t in enumerate(tokens) if freq[t] <= 2]

        if len(rare_positions) < 2:
            return 0.0

        gaps = [rare_positions[i+1] - rare_positions[i]
                for i in range(len(rare_positions) - 1)]

        if not gaps:
            return 0.0

        mean_gap = np.mean(gaps)
        std_gap = np.std(gaps)

        if mean_gap + std_gap == 0:
            return 0.0

        return (std_gap - mean_gap) / (std_gap + mean_gap)

    def _avg_word_length(self, tokens: list[str]) -> float:
        if not tokens:
            return 0.0
        return np.mean([len(t) for t in tokens])

    def _word_length_std(self, tokens: list[str]) -> float:
        if not tokens:
            return 0.0
        return np.std([len(t) for t in tokens])

    def _punctuation_ratio(self, text: str) -> float:
        if not text:
            return 0.0
        punct_count = sum(1 for c in text if c in '.,;:!?-()[]{}"\'/\\')
        return punct_count / len(text)

    def _comma_ratio(self, text: str) -> float:
        sentences = self._get_sentences(text)
        if not sentences:
            return 0.0
        comma_count = text.count(',')
        return comma_count / len(sentences)

    def _exclamation_ratio(self, text: str) -> float:
        sentences = self._get_sentences(text)
        if not sentences:
            return 0.0
        excl_count = text.count('!')
        return excl_count / len(sentences)

    def _question_ratio(self, text: str) -> float:
        sentences = self._get_sentences(text)
        if not sentences:
            return 0.0
        quest_count = text.count('?')
        return quest_count / len(sentences)

    def _first_person_ratio(self, tokens: list[str]) -> float:
        if not tokens:
            return 0.0
        fp_count = sum(1 for t in tokens if t in self.first_person)
        return fp_count / len(tokens)

    def _formal_word_ratio(self, tokens: list[str]) -> float:
        if not tokens:
            return 0.0
        formal_count = sum(1 for t in tokens if t in self.formal_words)
        return formal_count / len(tokens)

    def extract_single(self, text: str) -> dict[str, float]:
        tokens = self._tokenize(text)
        sentences = self._get_sentences(text)

        avg_sent_len, sent_len_std, sent_len_cv = self._sentence_stats(sentences)

        return {
            "ttr": self._ttr(tokens),
            "root_ttr": self._root_ttr(tokens),
            "log_ttr": self._log_ttr(tokens),
            "hapax_ratio": self._hapax_ratio(tokens),
            "dis_legomena_ratio": self._dis_legomena_ratio(tokens),
            "avg_sentence_length": avg_sent_len,
            "sentence_length_std": sent_len_std,
            "sentence_length_cv": sent_len_cv,
            "sentence_count": float(len(sentences)),
            "bigram_repetition": self._ngram_repetition(tokens, 2),
            "trigram_repetition": self._ngram_repetition(tokens, 3),
            "avg_word_length": self._avg_word_length(tokens),
            "word_length_std": self._word_length_std(tokens),
            "word_count": float(len(tokens)),
            "function_word_ratio": self._function_word_ratio(tokens),
            "transition_word_ratio": self._transition_word_ratio(tokens),
            "hedge_word_ratio": self._hedge_word_ratio(tokens),
            "flesch_reading_ease": self._flesch_reading_ease(text),
            "flesch_kincaid_grade": self._flesch_kincaid_grade(text),
            "punctuation_ratio": self._punctuation_ratio(text),
            "comma_ratio": self._comma_ratio(text),
            "rare_word_burstiness": self._rare_word_burstiness(tokens),
            "exclamation_ratio": self._exclamation_ratio(text),
            "question_ratio": self._question_ratio(text),
            "first_person_ratio": self._first_person_ratio(tokens),
            "formal_word_ratio": self._formal_word_ratio(tokens),
        }

    def extract(
        self,
        texts: list[str],
        cache_key: Optional[str] = None,
        use_cache: bool = True,
    ) -> tuple[np.ndarray, list[str]]:
        if cache_key is None:
            cache_key = self._compute_texts_hash(texts)

        cache_path = self._get_cache_path(cache_key)

        if use_cache and cache_path.exists():
            print(f"[StyleFeatures] Loading from cache: {cache_path}")
            data = np.load(cache_path, allow_pickle=True)
            return data["features"], list(data["feature_names"])

        print(f"[StyleFeatures] Extracting features from {len(texts)} documents...")

        all_features = []
        from tqdm.auto import tqdm
        for text in tqdm(texts, desc="Style features", ascii=True):
            feat_dict = self.extract_single(text)
            all_features.append(feat_dict)

        feature_names = list(all_features[0].keys())

        features = np.zeros((len(texts), len(feature_names)), dtype=np.float32)
        for i, feat_dict in enumerate(all_features):
            for j, name in enumerate(feature_names):
                features[i, j] = feat_dict.get(name, 0.0)

        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        if use_cache:
            print(f"[StyleFeatures] Saving to cache: {cache_path}")
            np.savez_compressed(
                cache_path,
                features=features,
                feature_names=np.array(feature_names),
            )

        print(f"[StyleFeatures] Extracted {len(feature_names)} features")
        return features, feature_names


def get_style_feature_names() -> list[str]:
    return [
        "ttr", "root_ttr", "log_ttr", "hapax_ratio", "dis_legomena_ratio",
        "avg_sentence_length", "sentence_length_std", "sentence_length_cv", "sentence_count",
        "bigram_repetition", "trigram_repetition",
        "avg_word_length", "word_length_std", "word_count",
        "function_word_ratio", "transition_word_ratio", "hedge_word_ratio",
        "flesch_reading_ease", "flesch_kincaid_grade",
        "punctuation_ratio", "comma_ratio",
        "rare_word_burstiness",
        "exclamation_ratio", "question_ratio", "first_person_ratio", "formal_word_ratio",
    ]