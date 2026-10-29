"""Classifiers of Phase 4: a causal dilated 1D convolutional network and an LSTM.

Both take a window of the 24 feature channels (3 per winch, cdpr_ft.learning.features) over the
last W estimator cycles, shape (batch, 24, W), and return the logits of the 17 modes. The
standardisation of the channels (mean and standard deviation of the training windows) is part
of the model, so that a saved model is applied to raw features. Before the standardisation the
channels are compressed by the signed logarithm sign(x) ln(1 + |x|) (protocol, amendment 1): the
length residual of a broken cable reaches thousands of millimetres, the residuals that reveal an
actuator fault a few.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

N_CHANNELS = 24
N_MODES = 17


def compress(x):
    """Signed logarithm, applied to the scaled channels (works on tensors and arrays)."""
    if isinstance(x, torch.Tensor):
        return torch.sign(x) * torch.log1p(torch.abs(x))
    import numpy as np
    return np.sign(x) * np.log1p(np.abs(x))


class Standardise(nn.Module):
    """Compression, then the standardisation with the statistics of the compressed training windows."""

    def __init__(self, mean=None, std=None):
        super().__init__()
        self.register_buffer("mean", torch.zeros(N_CHANNELS) if mean is None else torch.as_tensor(mean).float())
        self.register_buffer("std", torch.ones(N_CHANNELS) if std is None else torch.as_tensor(std).float())

    def forward(self, x):                       # (B, C, W)
        return (compress(x) - self.mean[None, :, None]) / self.std[None, :, None]


class CausalCNN(nn.Module):
    """Stack of causal convolutions with dilation 2^l (padding on the past side only). The
    features of the last time step and the mean over the window feed a linear layer."""

    def __init__(self, layers=3, channels=64, kernel=5, dropout=0.1, mean=None, std=None):
        super().__init__()
        self.norm = Standardise(mean, std)
        self.convs = nn.ModuleList()
        self.pads = []
        c_in = N_CHANNELS
        for layer in range(layers):
            d = 2 ** layer
            self.convs.append(nn.Conv1d(c_in, channels, kernel, dilation=d))
            self.pads.append((kernel - 1) * d)
            c_in = channels
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(2 * channels, N_MODES)

    def forward(self, x):
        h = self.norm(x)
        for conv, pad in zip(self.convs, self.pads):
            h = self.drop(F.relu(conv(F.pad(h, (pad, 0)))))
        return self.head(torch.cat([h[:, :, -1], h.mean(dim=2)], dim=1))


class LSTMNet(nn.Module):
    def __init__(self, hidden=64, layers=1, dropout=0.1, mean=None, std=None):
        super().__init__()
        self.norm = Standardise(mean, std)
        self.lstm = nn.LSTM(N_CHANNELS, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, N_MODES)

    def forward(self, x):
        h, _ = self.lstm(self.norm(x).transpose(1, 2))
        return self.head(self.drop(h[:, -1]))


def build(params: dict, mean=None, std=None) -> nn.Module:
    if params["arch"] == "cnn":
        return CausalCNN(params["cnn_layers"], params["cnn_channels"], params["cnn_kernel"], params["dropout"], mean, std)
    if params["arch"] == "lstm":
        return LSTMNet(params["lstm_hidden"], params["lstm_layers"], params["dropout"], mean, std)
    raise ValueError(params["arch"])


def n_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())
