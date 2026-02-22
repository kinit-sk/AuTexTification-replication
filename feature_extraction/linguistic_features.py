"""
- Morphological, word-dependency, part-of-speech and named-entity features
- Word frequency indicators (Google Books N-grams)
- Aggregated at document level as counts/percentages
- Used with Random Forest classifier (200 trees, max_depth=60)

spaCy models:
- English: en_core_web_sm
- Spanish: es_core_news_sm
"""

import numpy as np
import pandas as pd
import spacy
from collections import Counter
from pathlib import Path
from tqdm.auto import tqdm


FREQ_THRESHOLDS = {
    "very_high": 1_000_000,
    "medium_low": 1_000,
}


class LinguisticFeatures:
    """
    Extracts document-level linguistic features for Random Forest classification.
    - POS tag counts
    - Dependency label counts
    - NER label counts
    - Morphological feature counts
    - Word frequency statistics (from Google Books N-grams)
    """

    def __init__(self, language: str, resources_dir: Path | str = "resources"):
        self.language = language
        self.resources_dir = Path(resources_dir)
        
        self._load_spacy_model()
        self._load_word_frequencies()

    def _load_spacy_model(self) -> None:
        model_map = {
            "en": "en_core_web_sm",
            "es": "es_core_news_sm",
        }
        
        model_name = model_map.get(self.language)
        if model_name is None:
            raise ValueError(f"Unsupported language: {self.language}")
        
        print(f"[LingFeatures] Loading spaCy model: {model_name}")
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"[LingFeatures] Model not found. Downloading {model_name}...")
            spacy.cli.download(model_name)
            self.nlp = spacy.load(model_name)

    def _load_word_frequencies(self) -> None:
        """Load word frequency data from Google Books N-grams."""
        freq_path = self.resources_dir / self.language / "word_freq_matrix.tsv.gz"
        
        if not freq_path.exists():
            print(f"[LingFeatures] Word frequency file not found: {freq_path}")
            print("[LingFeatures] Word frequency features will use defaults.")
            self.word_freq_dict: dict[str, float] = {}
            return
        
        print(f"[LingFeatures] Loading word frequencies from: {freq_path}")
        df = pd.read_csv(freq_path, sep="\t", compression="gzip")
        df = df[df["word"].notnull()]
        df["word"] = df["word"].astype(str).str.lower()
        self.word_freq_dict = df.set_index("word")["freq"].to_dict()
        print(f"[LingFeatures] Loaded {len(self.word_freq_dict):,} word frequencies")

    def _get_word_freq_category(self, word: str) -> str:
        freq = self.word_freq_dict.get(word.lower(), 0)
        
        if freq == 0:
            return "NO_FREQ"
        elif freq >= FREQ_THRESHOLDS["very_high"]:
            return "VERY_FREQ"
        else:
            return "MEDIUM_LOW_FREQ"

    def _extract_single_doc(self, text: str) -> dict[str, float]:
        doc = self.nlp(text)
        
        counts: dict[str, int] = Counter()
        word_count = 0
        content_word_count = 0
        
        for token in doc:
            if token.is_space:
                continue
                
            word_count += 1
            
            counts[f"DEP_{token.dep_}"] += 1
            
            if self.language == "en":
                counts[f"TAG_{token.tag_}"] += 1
                for morph_feat in token.morph:
                    counts[f"MORPH_{morph_feat}"] += 1
            else:
                counts[f"POS_{token.pos_}"] += 1
                morph_str = str(token.morph)
                if morph_str:
                    counts[f"MORPH_{morph_str}"] += 1
            
            if not token.is_punct:
                content_word_count += 1
        
        for ent in doc.ents:
            counts[f"NER_{ent.label_}"] += 1
        
        tokens_without_ent = sum(1 for t in doc if not t.ent_type_ and not t.is_space)
        counts["NER_NONE"] = tokens_without_ent
        
        freq_categories = Counter()
        for token in doc:
            if token.is_space or token.is_punct:
                continue
            cat = self._get_word_freq_category(token.text)
            freq_categories[cat] += 1
        
        counts["VERY_FREQ_W"] = freq_categories.get("VERY_FREQ", 0)
        counts["MEDIUM_LOW_FREQ_W"] = freq_categories.get("MEDIUM_LOW_FREQ", 0)
        counts["NO_FREQ_W"] = freq_categories.get("NO_FREQ", 0)
        
        features: dict[str, float] = {}
        
        for key, count in counts.items():
            features[key] = float(count)
        
        return features

    def extract_features(
        self,
        texts: list[str],
        feature_names: list[str] | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Extract linguistic features from a list of texts.
        
        Args:
            texts: List of document texts
            feature_names: Optional list of feature names to use (for consistency
                           between train/dev/test). If None, discovers features
                           from the data.
            
        Returns:
            features: numpy array of shape (n_docs, n_features)
            feature_names: list of feature names
        """
        print(f"[LingFeatures] Extracting features from {len(texts)} documents...")
        
        all_features: list[dict[str, int]] = []
        
        for text in tqdm(texts, desc="Extracting linguistic features", ascii=True):
            doc_features = self._extract_single_doc(text)
            all_features.append(doc_features)
        
        if feature_names is None:
            all_keys: set[str] = set()
            for feat_dict in all_features:
                all_keys.update(feat_dict.keys())
            feature_names = sorted(all_keys)
        
        n_docs = len(texts)
        n_features = len(feature_names)
        
        feature_matrix = np.zeros((n_docs, n_features), dtype=np.float32)
        
        name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
        
        for i, feat_dict in enumerate(all_features):
            for name, value in feat_dict.items():
                if name in name_to_idx:
                    feature_matrix[i, name_to_idx[name]] = value
        
        print(f"[LingFeatures] Extracted {n_features} features")
        return feature_matrix, feature_names


class LingRFClassifier:

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 60,
        random_state: int = 10,
    ):
        from sklearn.ensemble import RandomForestClassifier
        
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
        self.feature_names: list[str] = []

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> "LingRFClassifier":
        print(f"[LingRF] Training on {X.shape[0]} samples, {X.shape[1]} features")
        self.model.fit(X, y)
        
        if feature_names:
            self.feature_names = feature_names
        
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def get_feature_importance(self, top_k: int = 20) -> list[tuple[str, float]]:
        if not self.feature_names:
            return []
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_k]
        
        return [
            (self.feature_names[i], importances[i])
            for i in indices
        ]


class LingRFPredOutClassifier:
    """RF on linguistic features + LSTM predictability output probabilities."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 60,
        random_state: int = 10,
    ):
        from sklearn.ensemble import RandomForestClassifier
        
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
        self.feature_names: list[str] = []

    def fit(
        self,
        ling_features: np.ndarray,
        pred_probs: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> "LingRFPredOutClassifier":
        X = np.concatenate([ling_features, pred_probs], axis=1)
        
        if feature_names:
            prob_names = [f"PRED_PROB_{i}" for i in range(pred_probs.shape[1])]
            self.feature_names = feature_names + prob_names
        
        print(f"[LingRF+PredOut] Training on {X.shape[0]} samples, {X.shape[1]} features")
        print(f"  - Linguistic features: {ling_features.shape[1]}")
        print(f"  - Prediction probabilities: {pred_probs.shape[1]}")
        
        self.model.fit(X, y)
        return self

    def predict(
        self,
        ling_features: np.ndarray,
        pred_probs: np.ndarray,
    ) -> np.ndarray:
        X = np.concatenate([ling_features, pred_probs], axis=1)
        return self.model.predict(X)

    def predict_proba(
        self,
        ling_features: np.ndarray,
        pred_probs: np.ndarray,
    ) -> np.ndarray:
        X = np.concatenate([ling_features, pred_probs], axis=1)
        return self.model.predict_proba(X)

    def get_feature_importance(self, top_k: int = 20) -> list[tuple[str, float]]:
        if not self.feature_names:
            return []
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_k]
        
        return [
            (self.feature_names[i], importances[i])
            for i in indices
        ]


class LingRFHybridPlusOutClassifier:
    """Two-stage UltraHybrid: RF on linguistic features + Hybrid+ output probabilities.

    Stage 1: Hybrid+ neural model (BiLSTM on pred+freq+grammar + encoder).
    Stage 2: RF on linguistic features + Stage-1 class probabilities.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 60,
        random_state: int = 10,
    ):
        from sklearn.ensemble import RandomForestClassifier
        
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
        self.feature_names: list[str] = []

    def fit(
        self,
        ling_features: np.ndarray,
        hybrid_probs: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> "LingRFHybridPlusOutClassifier":
        X = np.concatenate([ling_features, hybrid_probs], axis=1)
        
        if feature_names:
            prob_names = [f"HYBRID_PROB_{i}" for i in range(hybrid_probs.shape[1])]
            self.feature_names = feature_names + prob_names
        
        print(f"[LingRF+Hybrid+] Training on {X.shape[0]} samples, {X.shape[1]} features")
        print(f"  - Linguistic features: {ling_features.shape[1]}")
        print(f"  - Hybrid+ probabilities: {hybrid_probs.shape[1]}")
        
        self.model.fit(X, y)
        return self

    def predict(
        self,
        ling_features: np.ndarray,
        hybrid_probs: np.ndarray,
    ) -> np.ndarray:
        X = np.concatenate([ling_features, hybrid_probs], axis=1)
        return self.model.predict(X)

    def predict_proba(
        self,
        ling_features: np.ndarray,
        hybrid_probs: np.ndarray,
    ) -> np.ndarray:
        X = np.concatenate([ling_features, hybrid_probs], axis=1)
        return self.model.predict_proba(X)

    def get_feature_importance(self, top_k: int = 20) -> list[tuple[str, float]]:
        if not self.feature_names:
            return []
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_k]
        
        return [
            (self.feature_names[i], importances[i])
            for i in indices
        ]
