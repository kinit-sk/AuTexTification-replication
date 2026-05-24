"""MULTITuDE neural model definitions."""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.nn import LSTM, Linear, LogSoftmax, NLLLoss
from transformers import AutoModel


class MultitudeBaseClassifier(nn.Module):
    """Shared loss and prediction helpers for 8-class local models."""

    def __init__(self, num_classes: int, local_device: torch.device) -> None:
        super().__init__()
        self.num_classes: int = num_classes
        self.softmax_layer: LogSoftmax = LogSoftmax(dim=1)
        self.loss_fn: NLLLoss = NLLLoss()
        self.local_device: torch.device = local_device

    def compute_loss(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(pred, true)

    def postprocessing(self, y: torch.Tensor, argmax: bool) -> NDArray[np.int_ | np.float32]:
        if argmax:
            decisions: NDArray[np.int_] = y.argmax(dim=1).to(self.local_device).cpu().numpy()
            return decisions
        values: NDArray[np.float32] = y.to(self.local_device).cpu().numpy()
        return values


class MultitudePredLSTM(MultitudeBaseClassifier):
    """BiLSTM over multilingual probabilistic token features."""

    def __init__(
        self,
        seq_feature_len: int,
        num_classes: int,
        local_device: torch.device,
        hidden_size: int,
        bidirectional: bool,
    ) -> None:
        super().__init__(num_classes=num_classes, local_device=local_device)

        self.lstm_layer: LSTM = LSTM(
            input_size=seq_feature_len,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=bidirectional,
        )

        lstm_out_size: int = hidden_size * (2 if bidirectional else 1)
        self.linear_layer: Linear = Linear(lstm_out_size, self.num_classes)

    def forward(self, x_sequence_features: torch.Tensor) -> torch.Tensor:
        _, (hidden_state, _) = self.lstm_layer(x_sequence_features)
        transposed: torch.Tensor = torch.transpose(hidden_state, 0, 1)
        reshaped: torch.Tensor = torch.reshape(transposed, (transposed.shape[0], -1))
        scores: torch.Tensor = self.linear_layer(reshaped)
        return self.softmax_layer(scores)

    def freeze_llm(self) -> None:
        return

    def unfreeze_llm(self) -> None:
        return


class MultitudeHybridBiLSTMEncoder(MultitudeBaseClassifier):
    """BiLSTM over probabilistic features fused with a multilingual encoder."""

    def __init__(
        self,
        seq_feature_len: int,
        num_classes: int,
        local_device: torch.device,
        encoder_id: str,
        lstm_hidden_size: int,
        lstm_bidirectional: bool,
    ) -> None:
        super().__init__(num_classes=num_classes, local_device=local_device)

        self.llm: nn.Module = AutoModel.from_pretrained(encoder_id)
        linear_size: int = int(self.llm.config.hidden_size)

        self.lstm_layer: LSTM = LSTM(
            input_size=seq_feature_len,
            hidden_size=lstm_hidden_size,
            batch_first=True,
            bidirectional=lstm_bidirectional,
        )
        lstm_out_size: int = lstm_hidden_size * (2 if lstm_bidirectional else 1)
        self.linear_layer: Linear = Linear(linear_size + lstm_out_size, self.num_classes)

    def forward(
        self,
        x_sequence_features: torch.Tensor,
        x_input_ids: torch.Tensor,
        x_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        out = self.llm(input_ids=x_input_ids, attention_mask=x_attention_mask)
        pooler: torch.Tensor | None = getattr(out, "pooler_output", None)
        pooled: torch.Tensor = pooler if pooler is not None else out.last_hidden_state[:, 0, :]

        _, (hidden_state, _) = self.lstm_layer(x_sequence_features)
        transposed: torch.Tensor = torch.transpose(hidden_state, 0, 1)
        reshaped: torch.Tensor = torch.reshape(transposed, (transposed.shape[0], -1))
        concatenated: torch.Tensor = torch.cat((reshaped, pooled), dim=-1)
        scores: torch.Tensor = self.linear_layer(concatenated)
        return self.softmax_layer(scores)

    def freeze_llm(self) -> None:
        for param in self.llm.parameters():
            param.requires_grad = False

    def unfreeze_llm(self) -> None:
        for param in self.llm.parameters():
            param.requires_grad = True
