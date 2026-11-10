"""Temporal network components for next-hour electricity forecasting.

Both models use an already prepared, past-only consumption sequence and a
separate vector of forecast-time covariates. They return one prediction per
sample. Dataset splitting, fitted preprocessing and metrics belong to the
training pipeline and are intentionally outside these modules.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def _validate_inputs(history, context, context_size):
    if history.ndim != 3 or history.shape[2] != 1 or history.shape[1] < 1:
        raise ValueError('history must have shape [batch, time, 1] with at least one time step')
    if context.ndim != 2 or context.shape != (history.shape[0], context_size):
        raise ValueError(f'context must have shape [batch, {context_size}]')


class CausalResidualBlock(nn.Module):
    """Two dilated convolutions with left-only padding and a residual path."""

    def __init__(self, channels, kernel_size, dilation, dropout):
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence):
        hidden = self.conv1(F.pad(sequence, (self.left_padding, 0)))
        hidden = self.dropout(F.relu(hidden))
        hidden = self.conv2(F.pad(hidden, (self.left_padding, 0)))
        hidden = self.dropout(F.relu(hidden))
        return F.relu(sequence + hidden)


class TCNRegressor(nn.Module):
    """Causal dilated TCN with forecast-time context concatenated at the head.

    History is oldest-to-newest with shape ``[batch, time, 1]``; context has
    shape ``[batch, context_size]``. The scalar prediction has shape ``[batch]``.
    The receptive field is exposed in ``receptive_field``; increasing input
    length alone cannot make the last representation see beyond that field.
    """

    def __init__(self, context_size, channels=32, layers=6, kernel_size=3, dropout=0.1):
        super().__init__()
        if context_size < 0 or channels < 1 or layers < 1 or kernel_size < 1:
            raise ValueError('invalid context size or TCN dimensions')
        if not 0 <= dropout < 1:
            raise ValueError('dropout must be in [0, 1)')
        self.context_size = context_size
        self.receptive_field = 1 + 2 * (kernel_size - 1) * (2 ** layers - 1)
        self.project = nn.Conv1d(1, channels, 1)
        self.blocks = nn.Sequential(*[
            CausalResidualBlock(channels, kernel_size, 2 ** index, dropout)
            for index in range(layers)
        ])
        self.head = nn.Linear(channels + context_size, 1)

    def forward(self, history, context):
        _validate_inputs(history, context, self.context_size)
        sequence = self.blocks(self.project(history.transpose(1, 2)))
        last = sequence[:, :, -1]
        return self.head(torch.cat((last, context), dim=1)).squeeze(-1)


class TransformerRegressor(nn.Module):
    """Transformer encoder over a completed historical window.

    Sinusoidal positional encodings preserve ordering. All encoder positions
    are historical at the forecast origin, so attention can connect any pair
    within that window without using future target observations. Forecast-time
    covariates enter the regression head separately.
    """

    def __init__(self, context_size, d_model=32, heads=4, layers=2, feedforward=128, dropout=0.1):
        super().__init__()
        if context_size < 0 or d_model < 1 or heads < 1 or layers < 1 or feedforward < 1:
            raise ValueError('invalid context size or Transformer dimensions')
        if d_model % heads:
            raise ValueError('d_model must be divisible by heads')
        if not 0 <= dropout < 1:
            raise ValueError('dropout must be in [0, 1)')
        self.context_size = context_size
        self.d_model = d_model
        self.project = nn.Linear(1, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Linear(d_model + context_size, 1)

    def forward(self, history, context):
        _validate_inputs(history, context, self.context_size)
        sequence = self.project(history)
        steps = torch.arange(sequence.shape[1], device=sequence.device, dtype=sequence.dtype).unsqueeze(1)
        dimensions = torch.arange(0, self.d_model, 2, device=sequence.device, dtype=sequence.dtype)
        angles = steps * torch.exp(dimensions * (-math.log(10000.0) / self.d_model))
        positional = torch.zeros(sequence.shape[1], self.d_model, device=sequence.device, dtype=sequence.dtype)
        positional[:, 0::2] = angles.sin()
        positional[:, 1::2] = angles[:, :positional[:, 1::2].shape[1]].cos()
        encoded = self.encoder(sequence + positional.unsqueeze(0))
        return self.head(torch.cat((encoded[:, -1, :], context), dim=1)).squeeze(-1)
