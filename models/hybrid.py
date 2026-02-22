from __future__ import annotations

from torch.nn import Module, LSTM, Linear, LogSoftmax, NLLLoss
import torch
from transformers import AutoModel


class BaseClassifier(Module):
    """Common utilities for all classifiers."""

    def __init__(self, task: str, local_device: torch.device):
        super().__init__()
        self.num_classes = 2 if task == "subtask_1" else 6
        self.softmax_layer = LogSoftmax(dim=1)
        self.loss_fn = NLLLoss()
        self.local_device = local_device

    def compute_loss(self, pred, true):
        return self.loss_fn(pred, true)

    def postprocessing(self, Y, argmax: bool = True):
        if argmax:
            decisions = Y.argmax(dim=1).to(self.local_device).cpu().numpy()
        else:
            decisions = Y.to(self.local_device).cpu().numpy()
        return decisions


class PredLSTM(BaseClassifier):
    """BiLSTM over sequence predictability features only."""

    def __init__(
        self,
        seq_feature_len: int,
        task: str,
        local_device: torch.device,
        hidden_size: int = 64,
        bidirectional: bool = True,
    ):
        super().__init__(task, local_device)

        self.lstm_layer = LSTM(
            input_size=seq_feature_len,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=bidirectional,
        )

        lstm_out_size = hidden_size * (2 if bidirectional else 1)
        self.linear_layer = Linear(lstm_out_size, self.num_classes)

    def forward(self, x_sequence_features, x_input_ids=None, x_attention_mask=None):
        _, (hidden_state, _) = self.lstm_layer(x_sequence_features)
        transposed = torch.transpose(hidden_state, 0, 1)
        reshaped = torch.reshape(transposed, (transposed.shape[0], -1))
        scores = self.linear_layer(reshaped)
        log_probabilities = self.softmax_layer(scores)
        return log_probabilities

    def freeze_llm(self):
        pass

    def unfreeze_llm(self):
        pass


def _pool_encoder_output(out) -> torch.Tensor:
    """Return pooler_output when available, else CLS embedding."""
    pooler = getattr(out, "pooler_output", None)
    if pooler is not None:
        return pooler
    return out.last_hidden_state[:, 0, :]


class FLMRoBERTa(BaseClassifier):
    """Encoder-only transformer baseline using AutoModel."""

    def __init__(
        self,
        task: str,
        local_device: torch.device,
        roberta_variant: str,
        baseline_compat_no_freeze: bool = True,
    ):
        super().__init__(task, local_device)

        self.llm = AutoModel.from_pretrained(roberta_variant)
        linear_size = int(self.llm.config.hidden_size)
        self.linear_layer = Linear(linear_size, self.num_classes)
        self._baseline_compat_no_freeze = baseline_compat_no_freeze

    def forward(self, x_sequence_features, x_input_ids, x_attention_mask):
        out = self.llm(
            input_ids=x_input_ids,
            attention_mask=x_attention_mask,
        )
        pooled = _pool_encoder_output(out)

        scores = self.linear_layer(pooled)
        log_probabilities = self.softmax_layer(scores)
        return log_probabilities

    def freeze_llm(self):
        if self._baseline_compat_no_freeze:
            return
        for p in self.llm.parameters():
            p.requires_grad = False

    def unfreeze_llm(self):
        if self._baseline_compat_no_freeze:
            return
        for p in self.llm.parameters():
            p.requires_grad = True


class HybridBiLSTMRoBERTa(BaseClassifier):
    """BiLSTM over sequence features + transformer encoder backbone."""

    def __init__(
        self,
        seq_feature_len: int,
        task: str,
        local_device: torch.device,
        roberta_variant: str,
        disable_sequence: bool = False,
        lstm_hidden_size: int = 64,
        lstm_bidirectional: bool = True,
    ):
        super().__init__(task, local_device)

        self.llm = AutoModel.from_pretrained(roberta_variant)
        linear_size = int(self.llm.config.hidden_size)

        self.disable_sequence = disable_sequence
        if not disable_sequence:
            self.lstm_layer = LSTM(
                input_size=seq_feature_len,
                hidden_size=lstm_hidden_size,
                batch_first=True,
                bidirectional=lstm_bidirectional,
            )
            lstm_out_size = lstm_hidden_size * (2 if lstm_bidirectional else 1)
            linear_size += lstm_out_size

        self.linear_layer = Linear(linear_size, self.num_classes)

    def forward(self, x_sequence_features, x_input_ids, x_attention_mask):
        out = self.llm(
            input_ids=x_input_ids,
            attention_mask=x_attention_mask,
        )
        pooled = _pool_encoder_output(out)

        if not self.disable_sequence:
            _, (hidden_state, _) = self.lstm_layer(x_sequence_features)
            transposed = torch.transpose(hidden_state, 0, 1)
            reshaped = torch.reshape(transposed, (transposed.shape[0], -1))
            concatenated = torch.cat((reshaped, pooled), dim=-1)
        else:
            concatenated = pooled

        scores = self.linear_layer(concatenated)
        log_probabilities = self.softmax_layer(scores)
        return log_probabilities

    def freeze_llm(self):
        for param in self.llm.parameters():
            param.requires_grad = False

    def unfreeze_llm(self):
        for param in self.llm.parameters():
            param.requires_grad = True