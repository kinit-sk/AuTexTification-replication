import os
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from pathlib import Path
from tqdm.auto import tqdm

import language_tool_python
from transformers import AutoTokenizer, GPT2TokenizerFast
from difflib import SequenceMatcher

FIXED_LEN = 128

class TokenAdapter:
    """Adapter for different tokenizer interfaces."""
    def __init__(self, tokenizer, name: str, ensure_fast: bool = True):
        self.tokenizer = tokenizer
        self.name = name
        self.fast = getattr(tokenizer, "is_fast", False)
        if ensure_fast and not self.fast:
            print(f"[WARN] Tokenizer '{name}' is not fast. Offset mappings may be missing.")

    def encode_ids(self, text: str, add_special_tokens: bool = False):
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def tokens_from_ids(self, ids):
        if hasattr(self.tokenizer, "convert_ids_to_tokens"):
            return self.tokenizer.convert_ids_to_tokens(ids)
        text = self.tokenizer.decode(ids, skip_special_tokens=False)
        return text.split()

    def encode_with_offsets(self, text: str):
        if not self.fast:
            return None
        enc = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        try:
            word_ids = enc.word_ids()
        except Exception:
            word_ids = enc.encodings[0].word_ids if hasattr(enc, "encodings") and enc.encodings else None
        return {
            "word_ids": word_ids,
            "offset_mapping": enc["offset_mapping"]
        }


def load_tokenizer_for_language(lang: str):
    """Loads the correct tokenizer per language."""
    if lang == "en":
        print("[INFO] Loading English tokenizer: distilgpt2")
        tok = GPT2TokenizerFast.from_pretrained("distilgpt2")
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return TokenAdapter(tok, "distilgpt2", ensure_fast=True), "gpt2_en"

    elif lang == "es":
        model_id = "datificate/gpt2-small-spanish"
        print(f"[INFO] Loading Spanish tokenizer: {model_id}")
        try:
            tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load Spanish tokenizer: {e}")

        if tok.pad_token is None:
            if tok.eos_token is not None:
                tok.pad_token = tok.eos_token
            else:
                tok.add_special_tokens({'pad_token': '[PAD]'})

        return TokenAdapter(tok, model_id, ensure_fast=True), "gpt2_es"

    else:
        raise ValueError(f"Unsupported language: {lang}")


def _space_words_and_spans(sent: str) -> tuple[list[str], list[tuple[int, int]]]:
    """Space-based tokenization returning words and their (start, end) char spans."""
    words = sent.split()
    spans: list[tuple[int, int]] = []
    idx = 0
    for w in words:
        while idx < len(sent) and sent[idx].isspace():
            idx += 1
        start = idx
        end = start + len(w)
        spans.append((start, end))
        idx = end
    return words, spans


class GrammarFeatures:
    """Per-token grammar correctness via LanguageTool diff.

    Corrects each sentence, diffs original vs corrected words with
    SequenceMatcher (preserved=1, changed=0), then aligns word-level
    flags to subword tokens via char-span overlap.
    """

    def __init__(self, device, local_device, language: str):
        self.device = device
        self.local_device = local_device
        self.language = language

        self.adapter, src = load_tokenizer_for_language(language)
        self.adapter_source = src

        lt_lang = "en" if language == "en" else "es"
        try:
            self.grammar_checker = language_tool_python.LanguageTool(
                lt_lang, remote_server="http://localhost:8010"
            )
        except Exception:
            print("[WARN] Local LT server unavailable. Falling back to public API.")
            self.grammar_checker = language_tool_python.LanguageToolPublicAPI(lt_lang)
        self._cache = {}

    def _correct_sentence(self, sent: str) -> str:
        if sent in self._cache:
            return self._cache[sent]
        try:
            corrected = self.grammar_checker.correct(sent)
        except Exception:
            corrected = sent
        self._cache[sent] = corrected
        return corrected

    def _compute_word_flags(self, sent: str):
        orig_words = sent.split()
        if not orig_words:
            return []

        corrected = self._correct_sentence(sent)
        corr_words = corrected.split()

        flags = [0] * len(orig_words)

        sm = SequenceMatcher(None, orig_words, corr_words)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                for i in range(i1, i2):
                    flags[i] = 1

        return flags

    def word_features(self, sentences):
        results = np.zeros((len(sentences), FIXED_LEN, 1), dtype=np.float32)

        print(f"Computing grammar features (paper-exact)… [adapter={self.adapter_source}]")
        progress = tqdm(range(len(sentences)), ascii=True)

        for i, sent in enumerate(sentences):

            word_flags = self._compute_word_flags(sent)
            words, word_spans = _space_words_and_spans(sent)

            if len(word_flags) != len(words):
                n = min(len(word_flags), len(words))
                word_flags = word_flags[:n]
                word_spans = word_spans[:n]

            enc = self.adapter.tokenizer(
                sent,
                return_offsets_mapping=True,
                add_special_tokens=False,
            )
            offsets = enc["offset_mapping"]

            token_vals = []

            for (tok_start, tok_end) in offsets:
                if tok_end <= tok_start:
                    token_vals.append(1.0)
                    continue

                assigned = False
                for idx_w, (w_start, w_end) in enumerate(word_spans):
                    # overlap means token belongs to this word
                    if not (tok_end <= w_start or tok_start >= w_end):
                        token_vals.append(float(word_flags[idx_w]))
                        assigned = True
                        break

                if not assigned:
                    token_vals.append(1.0)

            upto = min(FIXED_LEN, len(token_vals))
            results[i, :upto, 0] = np.array(token_vals[:upto], dtype=np.float32)

            progress.update(1)

        return results

        
class WordFrequency:
    """Per-token log word-frequency features via space-based tokenization and char-span overlap."""

    def __init__(self, device, local_device, language: str):
        self.device = device
        self.local_device = local_device
        self.language = language

        print(f"[WF] Initializing word frequency extractor ({language})")

        if language == "en":
            self.tokenizer = GPT2TokenizerFast.from_pretrained("distilgpt2")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        elif language == "es":
            self.tokenizer = AutoTokenizer.from_pretrained(
                "datificate/gpt2-small-spanish", use_fast=True
            )
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        freq_path = Path(f"resources/{language}/word_freq_matrix.tsv.gz")
        if not freq_path.exists():
            print(f"[WF] File missing → default freq=1 for all words")
            self.word_freq_dict = defaultdict(lambda: 1.0)
        else:
            df = pd.read_csv(freq_path, sep="\t", compression="gzip")
            df = df[df["word"].notnull()]
            df["word"] = df["word"].astype(str).str.lower()
            self.word_freq_dict = df.set_index("word")["freq"].to_dict()

    def word_features(self, sentences):
        results = np.zeros((len(sentences), FIXED_LEN, 1), dtype=np.float32)
        print("[WF] Computing word frequency features...")
        bar = tqdm(range(len(sentences)), ascii=True)

        for i, sent in enumerate(sentences):

            words, spans = _space_words_and_spans(sent)

            word_freqs = []
            for w in words:
                f = self.word_freq_dict.get(w.lower(), 1.0)
                if f <= 0:
                    f = 1.0
                word_freqs.append(float(np.log(f)))

            enc = self.tokenizer(
                sent,
                return_offsets_mapping=True,
                add_special_tokens=False
            )
            offsets = enc["offset_mapping"]

            token_vals = []
            for (tok_start, tok_end) in offsets:

                if tok_end <= tok_start:
                    token_vals.append(0.0)
                    continue

                wid = None
                for j, (ws, we) in enumerate(spans):
                    if not (tok_end <= ws or tok_start >= we):
                        wid = j
                        break

                if wid is None:
                    token_vals.append(0.0)           # default freq=1→log1=0
                else:
                    token_vals.append(word_freqs[wid])

            upto = min(FIXED_LEN, len(token_vals))
            results[i, :upto, 0] = np.array(token_vals[:upto])

            bar.update(1)

        return results
