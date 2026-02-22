"""Probabilistic (LM-based) per-token feature extractors."""

from __future__ import annotations

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
)

from feature_extraction.feature_generator import FeatureGenerator

fixed_len: int = 128
BATCH_SIZE: int = 16
eps: float = 1e-40

GPT2_NATIVE_MODELS: frozenset[str] = frozenset(
    {"distilgpt2", "gpt2", "gpt2-medium", "gpt2-large"}
)


class ProbabilisticFeatures(FeatureGenerator):
    """Compute per-token probabilistic features from GPT-2 family models."""

    def __init__(
        self,
        device: torch.device,
        local_device: torch.device,
        language: str,
        disabled: bool,
    ) -> None:
        self.device = device
        self.local_device = local_device
        self.language = language
        self.disabled = disabled

        if language == "en":
            self.models = ["distilgpt2", "gpt2", "gpt2-medium", "gpt2-large"]
        elif language == "es":
            self.models = ["DeepESP/gpt2-spanish", "datificate/gpt2-small-spanish"]
        else:
            raise ValueError(f"[ProbabilisticFeatures] Unsupported language '{language}'")

    def word_features(self, sentences: list[str]) -> np.ndarray:
        FEATURE_NUM = 3
        num_sent = len(sentences)

        results = np.zeros(
            (num_sent, fixed_len, FEATURE_NUM * len(self.models) + 1), dtype=np.float32
        )

        if self.disabled:
            print("[ProbabilisticFeatures] Disabled flag active — returning zero matrix.")
            return results

        self.aggregated_results = np.zeros(
            (num_sent, FEATURE_NUM * len(self.models)), dtype=np.float32
        )

        for i_m, model_id in enumerate(self.models):
            print(f"[ProbabilisticFeatures] Computing probabilities using: {model_id}")

            if not model_id.startswith("DeepESP/gpt2-spanish"):
                model = GPT2LMHeadModel.from_pretrained(model_id)
                tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
                tokenizer.pad_token = tokenizer.eos_token
            else:
                model = AutoModelForCausalLM.from_pretrained(model_id, is_decoder=True)
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                if tokenizer.pad_token is None:
                    if tokenizer.eos_token is not None:
                        tokenizer.pad_token = tokenizer.eos_token

            model.to(self.device)
            model.eval()

            batches = [sentences[i : i + BATCH_SIZE] for i in range(0, num_sent, BATCH_SIZE)]
            pbar = tqdm(range(len(batches)), ascii=True)

            with torch.no_grad():
                for i_b, batch in enumerate(batches):
                    encodings = tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=fixed_len,
                        return_offsets_mapping=True,
                        return_tensors="pt",
                    )

                    encodings = {k: v.to(self.device) for k, v in encodings.items()}
                    target_ids = encodings["input_ids"].clone()

                    outputs = model(
                        encodings["input_ids"],
                        attention_mask=encodings["attention_mask"],
                    )
                    logits = outputs["logits"]

                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = target_ids[..., 1:].contiguous()

                    probs = torch.nn.functional.softmax(shift_logits, dim=-1)
                    probs_seen = torch.gather(probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)
                    greedy = torch.argmax(probs, dim=-1)
                    probs_greedy = torch.gather(probs, 2, greedy.unsqueeze(-1)).squeeze(-1)

                    log_probs_seen = torch.log(probs_seen + eps).to(self.local_device).numpy()
                    log_probs_seen = np.concatenate(
                        (np.zeros((len(batch), 1)), log_probs_seen), axis=1
                    )

                    log_probs_greedy = torch.log(probs_greedy + eps).to(self.local_device).numpy()
                    log_probs_greedy = np.concatenate(
                        (np.zeros((len(batch), 1)), log_probs_greedy), axis=1
                    )

                    entropy = torch.sum(torch.log(probs + eps) * (-probs), dim=-1)
                    entropy = entropy.to(self.local_device).numpy()
                    entropy = np.concatenate((np.zeros((len(batch), 1)), entropy), axis=1)

                    mask = encodings["attention_mask"].to(self.local_device).numpy()

                    start_idx = i_b * BATCH_SIZE
                    seq_len = mask.shape[1]

                    if i_m == 0:
                        results[start_idx : start_idx + len(batch), :seq_len, 0] = mask

                    results[start_idx : start_idx + len(batch), :seq_len, i_m * FEATURE_NUM + 1] = (
                        log_probs_seen * mask
                    )
                    results[start_idx : start_idx + len(batch), :seq_len, i_m * FEATURE_NUM + 2] = (
                        log_probs_greedy * mask
                    )
                    results[start_idx : start_idx + len(batch), :seq_len, i_m * FEATURE_NUM + 3] = (
                        entropy * mask
                    )

                    for i in range(len(batch)):
                        mask_i = mask[i] == 1
                        self.aggregated_results[start_idx + i, i_m * FEATURE_NUM] = np.mean(
                            log_probs_seen[i, mask_i]
                        )
                        self.aggregated_results[start_idx + i, i_m * FEATURE_NUM + 1] = np.mean(
                            log_probs_greedy[i, mask_i]
                        )
                        self.aggregated_results[start_idx + i, i_m * FEATURE_NUM + 2] = np.mean(
                            entropy[i, mask_i]
                        )

                    pbar.update(1)

        return results


MULTILINGUAL_LARGE_MODELS: list[str] = [
    "Qwen/Qwen2.5-3B",
    "meta-llama/Llama-3.2-3B",
    "facebook/xglm-2.9B",
    "bigscience/bloom-1b7",
]


class MultilingualProbFeatures(FeatureGenerator):
    """Compute probabilistic features from multilingual causal LMs."""

    def __init__(self, device: torch.device, models: list[str] | None = None) -> None:
        self.device = device
        self.models = models or MULTILINGUAL_LARGE_MODELS

    def _load_model(self, model_id: str):
        print(f"  Loading: {model_id}")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.to(self.device)
        model.eval()
        return model, tokenizer

    def word_features(self, sentences: list[str], batch_size: int = 8) -> np.ndarray:
        FEATURE_NUM = 3
        num_sent = len(sentences)
        num_models = len(self.models)

        results = np.zeros(
            (num_sent, fixed_len, FEATURE_NUM * num_models + 1), dtype=np.float32
        )

        for i_m, model_id in enumerate(self.models):
            print(f"\n[MultilingualProb] Model {i_m + 1}/{num_models}: {model_id}")
            model, tokenizer = self._load_model(model_id)

            batches = [sentences[i : i + batch_size] for i in range(0, num_sent, batch_size)]
            pbar = tqdm(batches, desc=f"  {model_id.split('/')[-1]}", ascii=True)

            with torch.no_grad():
                for i_b, batch in enumerate(pbar):
                    try:
                        encodings = tokenizer(
                            batch,
                            padding=True,
                            truncation=True,
                            max_length=fixed_len,
                            return_tensors="pt",
                        )
                    except Exception:
                        encodings = tokenizer(
                            batch,
                            padding=True,
                            truncation=True,
                            max_length=fixed_len,
                            return_tensors="pt",
                            return_offsets_mapping=False,
                        )

                    input_ids = encodings["input_ids"].to(self.device)
                    attention_mask = encodings["attention_mask"].to(self.device)

                    outputs = model(input_ids, attention_mask=attention_mask)
                    logits = outputs.logits

                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = input_ids[..., 1:].contiguous()

                    probs = torch.nn.functional.softmax(shift_logits, dim=-1)
                    probs_seen = torch.gather(probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)
                    greedy = torch.argmax(probs, dim=-1)
                    probs_greedy = torch.gather(probs, 2, greedy.unsqueeze(-1)).squeeze(-1)

                    log_probs_seen = torch.log(probs_seen + eps).cpu().numpy()
                    log_probs_seen = np.concatenate(
                        (np.zeros((len(batch), 1)), log_probs_seen), axis=1
                    )

                    log_probs_greedy = torch.log(probs_greedy + eps).cpu().numpy()
                    log_probs_greedy = np.concatenate(
                        (np.zeros((len(batch), 1)), log_probs_greedy), axis=1
                    )

                    entropy = (
                        torch.sum(torch.log(probs + eps) * (-probs), dim=-1).cpu().numpy()
                    )
                    entropy = np.concatenate((np.zeros((len(batch), 1)), entropy), axis=1)

                    mask = attention_mask.cpu().numpy()
                    start_idx = i_b * batch_size
                    seq_len = mask.shape[1]

                    if i_m == 0:
                        results[start_idx : start_idx + len(batch), :seq_len, 0] = mask

                    results[
                        start_idx : start_idx + len(batch), :seq_len, i_m * FEATURE_NUM + 1
                    ] = (log_probs_seen[:, :seq_len] * mask)
                    results[
                        start_idx : start_idx + len(batch), :seq_len, i_m * FEATURE_NUM + 2
                    ] = (log_probs_greedy[:, :seq_len] * mask)
                    results[
                        start_idx : start_idx + len(batch), :seq_len, i_m * FEATURE_NUM + 3
                    ] = (entropy[:, :seq_len] * mask)

            del model, tokenizer
            torch.cuda.empty_cache()

        results = np.nan_to_num(results, nan=0.0, posinf=0.0, neginf=0.0)
        return results


def load_prob_model(
    model_id: str, device: torch.device
) -> tuple:
    """Load a causal LM and its tokenizer, handling quirks per model family."""
    print(f"  Loading model: {model_id}")

    if model_id in GPT2_NATIVE_MODELS:
        model = GPT2LMHeadModel.from_pretrained(model_id)
        tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token

    elif model_id.startswith("DeepESP/gpt2-spanish"):
        model = AutoModelForCausalLM.from_pretrained(model_id, is_decoder=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

    elif model_id.startswith("datificate/"):
        model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

    elif "llama" in model_id.lower():
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    elif "qwen" in model_id.lower():
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    elif "xglm" in model_id.lower():
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    elif "bloom" in model_id.lower():
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    elif "mgpt" in model_id.lower():
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                model.resize_token_embeddings(len(tokenizer))

    model.to(device)
    model.eval()
    return model, tokenizer


class ConfigurableProbFeatures(FeatureGenerator):
    """Probabilistic feature extractor accepting arbitrary model ids."""

    def __init__(
        self,
        device: torch.device,
        local_device: torch.device,
        model_ids: list[str],
        disabled: bool = False,
        batch_size: int = 8,
    ) -> None:
        self.device = device
        self.local_device = local_device
        self.model_ids = model_ids
        self.disabled = disabled
        self.batch_size = batch_size

    def word_features(self, sentences: list[str]) -> np.ndarray:
        FEATURE_NUM = 3
        num_sent = len(sentences)
        num_models = len(self.model_ids)

        results = np.zeros(
            (num_sent, fixed_len, FEATURE_NUM * num_models + 1), dtype=np.float32
        )

        if self.disabled:
            print("[ProbabilisticFeatures] Disabled — returning zero matrix.")
            return results

        self.aggregated_results = np.zeros(
            (num_sent, FEATURE_NUM * num_models), dtype=np.float32
        )

        for i_m, model_id in enumerate(self.model_ids):
            print(f"\n[ProbabilisticFeatures] Model {i_m + 1}/{num_models}: {model_id}")

            model, tokenizer = load_prob_model(model_id, self.device)

            batches = [
                sentences[i : i + self.batch_size]
                for i in range(0, num_sent, self.batch_size)
            ]
            pbar = tqdm(
                range(len(batches)), ascii=True, desc=f"  {model_id.split('/')[-1]}"
            )

            with torch.no_grad():
                for i_b, batch in enumerate(batches):
                    try:
                        encodings = tokenizer(
                            batch,
                            padding=True,
                            truncation=True,
                            max_length=fixed_len,
                            return_tensors="pt",
                        )
                    except Exception:
                        encodings = tokenizer(
                            batch,
                            padding=True,
                            truncation=True,
                            max_length=fixed_len,
                            return_tensors="pt",
                            return_offsets_mapping=False,
                        )

                    input_ids = encodings["input_ids"].to(self.device)
                    attention_mask = encodings["attention_mask"].to(self.device)
                    target_ids = input_ids.clone()

                    outputs = model(input_ids, attention_mask=attention_mask)
                    logits = outputs.logits

                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = target_ids[..., 1:].contiguous()

                    probs = torch.nn.functional.softmax(shift_logits, dim=-1)
                    probs_seen = torch.gather(
                        probs, 2, shift_labels.unsqueeze(-1)
                    ).squeeze(-1)
                    greedy = torch.argmax(probs, dim=-1)
                    probs_greedy = torch.gather(
                        probs, 2, greedy.unsqueeze(-1)
                    ).squeeze(-1)

                    log_probs_seen = torch.log(probs_seen + eps).cpu().numpy()
                    log_probs_seen = np.concatenate(
                        (np.zeros((len(batch), 1)), log_probs_seen), axis=1
                    )

                    log_probs_greedy = torch.log(probs_greedy + eps).cpu().numpy()
                    log_probs_greedy = np.concatenate(
                        (np.zeros((len(batch), 1)), log_probs_greedy), axis=1
                    )

                    entropy = (
                        torch.sum(torch.log(probs + eps) * (-probs), dim=-1).cpu().numpy()
                    )
                    entropy = np.concatenate(
                        (np.zeros((len(batch), 1)), entropy), axis=1
                    )

                    mask = attention_mask.cpu().numpy()

                    start_idx = i_b * self.batch_size
                    seq_len = mask.shape[1]

                    if i_m == 0:
                        results[start_idx : start_idx + len(batch), :seq_len, 0] = mask

                    results[
                        start_idx : start_idx + len(batch),
                        :seq_len,
                        i_m * FEATURE_NUM + 1,
                    ] = (log_probs_seen * mask)
                    results[
                        start_idx : start_idx + len(batch),
                        :seq_len,
                        i_m * FEATURE_NUM + 2,
                    ] = (log_probs_greedy * mask)
                    results[
                        start_idx : start_idx + len(batch),
                        :seq_len,
                        i_m * FEATURE_NUM + 3,
                    ] = (entropy * mask)

                    for i in range(len(batch)):
                        mask_i = mask[i] == 1
                        self.aggregated_results[start_idx + i, i_m * FEATURE_NUM] = np.mean(
                            log_probs_seen[i, mask_i]
                        )
                        self.aggregated_results[
                            start_idx + i, i_m * FEATURE_NUM + 1
                        ] = np.mean(log_probs_greedy[i, mask_i])
                        self.aggregated_results[
                            start_idx + i, i_m * FEATURE_NUM + 2
                        ] = np.mean(entropy[i, mask_i])

                    pbar.update(1)

            pbar.close()

            del model
            del tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return np.nan_to_num(results, nan=0.0, posinf=0.0, neginf=0.0)
