"""Regression metrics in source units and percentage terms."""

import numpy as np


def score(actual, predicted):
    """Report MAE, RMSE and MAPE, excluding zero actuals only from MAPE."""
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if actual.shape != predicted.shape or not actual.size:
        raise ValueError('Actual and predicted arrays must have matching nonempty shapes')
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError('Metrics require finite values')
    errors = predicted - actual
    nonzero = actual != 0
    mape = float(np.mean(np.abs(errors[nonzero] / actual[nonzero])) * 100) if nonzero.any() else None
    return {'count': int(actual.size), 'mae': float(np.abs(errors).mean()),
            'rmse': float(np.sqrt(np.mean(errors ** 2))), 'mape_percent': mape,
            'mape_zero_actuals_excluded': int((~nonzero).sum())}
