"""LSTM and GRU starting points for one-hour consumption regression.

History is a chronological sequence ending at the forecast origin. Context
contains features known at that origin, including the target-hour calendar.
Input scaling and feature availability are the training pipeline's job.
"""

import torch
from torch import nn


class _RecurrentForecaster(nn.Module):
    """Encode a load history, combine its last state with context, and regress."""

    def __init__(self, cell, context_size, hidden_size=64, num_layers=1, dropout=0.0):
        super().__init__()
        if context_size < 0:
            raise ValueError('context_size must be nonnegative')
        if hidden_size < 1 or num_layers < 1:
            raise ValueError('hidden_size and num_layers must be positive')
        if not 0 <= dropout < 1:
            raise ValueError('dropout must be in [0, 1)')
        self.context_size = context_size
        self.recurrent = cell(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size + context_size, 1)

    def forward(self, history, context):
        """Return one prediction per row for history [B, T, 1] and context [B, C]."""
        if history.ndim != 3 or history.shape[1] == 0 or history.shape[2] != 1:
            raise ValueError('history must have shape [batch, nonempty time, 1]')
        if context.ndim != 2 or context.shape != (history.shape[0], self.context_size):
            raise ValueError('context must have shape [batch, context_size]')
        sequence, _ = self.recurrent(history)
        return self.head(torch.cat((sequence[:, -1, :], context), dim=1)).squeeze(-1)


class LSTMForecaster(_RecurrentForecaster):
    """Use LSTM memory and gates to encode the completed consumption history."""

    def __init__(self, context_size, hidden_size=64, num_layers=1, dropout=0.0):
        super().__init__(nn.LSTM, context_size, hidden_size, num_layers, dropout)


class GRUForecaster(_RecurrentForecaster):
    """Use GRU update and reset gates to encode the completed consumption history."""

    def __init__(self, context_size, hidden_size=64, num_layers=1, dropout=0.0):
        super().__init__(nn.GRU, context_size, hidden_size, num_layers, dropout)
