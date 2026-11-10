"""Train a direct, horizon-conditioned 24-hour model at one daily origin."""

import argparse
import copy
import hashlib
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch

from train import MODELS, FeatureConfig, Preprocessor, code_hashes, digest, load_split, loader


def predict_profiles(model, history, context):
    """Return [B, 24] predictions from shared past history and future context.

    All 24 history copies are identical within an origin. The context contains
    only forecast-time inputs such as target calendar and horizon. Actual future
    consumption is never fed back to the model.
    """
    if history.ndim != 3 or history.shape[2] != 1:
        raise ValueError('history must have shape [batch, time, 1]')
    if context.ndim != 3 or context.shape[:2] != (history.shape[0], 24):
        raise ValueError('context must have shape [batch, 24, features]')
    batch, steps, _ = history.shape
    repeated = history[:, None, :, :].expand(batch, 24, steps, 1).reshape(batch * 24, steps, 1)
    outputs = model(repeated, context.reshape(batch * 24, context.shape[2]))
    return outputs.reshape(batch, 24)


def profile_metrics(predicted, actual):
    """Compute hourly and 24-hour summed errors in original source units."""
    predicted, actual = np.asarray(predicted), np.asarray(actual)
    if predicted.shape != actual.shape or predicted.ndim != 2 or predicted.shape[1] != 24:
        raise ValueError('predictions and targets must have matching [profiles, 24] shapes')
    if not np.isfinite(predicted).all() or not np.isfinite(actual).all():
        raise ValueError('predictions and targets must be finite')
    errors = predicted - actual
    totals = predicted.sum(axis=1) - actual.sum(axis=1)
    return {'hourly_mae': float(np.mean(np.abs(errors))),
            'hourly_rmse': float(np.sqrt(np.mean(errors ** 2))),
            'total_mae': float(np.mean(np.abs(totals))),
            'total_rmse': float(np.sqrt(np.mean(totals ** 2)))}


def baseline_profiles(history):
    """Return previous-day and, when available, previous-week profiles.

    Input history is unscaled and ordered from oldest to newest. For a target
    at origin + h, the previous-day value is at origin + h - 24 hours.
    """
    history = np.asarray(history)
    if history.ndim != 3 or history.shape[1] < 24 or history.shape[2] != 1:
        raise ValueError('baseline requires at least 24 unscaled hourly readings')
    baselines = {'previous_day': history[:, -24:, 0]}
    if history.shape[1] >= 168:
        baselines['previous_week'] = history[:, -168:-144, 0]
    return baselines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--history', type=int, choices=(24, 72, 168), default=24)
    parser.add_argument('--calendar', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--rolling', action='store_true')
    parser.add_argument('--seasonal-lags', action='store_true')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=32, help='Number of forecast origins per batch')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error('epochs, patience, batch size and learning rate must be positive')
    if args.output.exists():
        parser.error('output directory must not already exist')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)

    config = FeatureConfig(history_hours=args.history, calendar=args.calendar,
                           rolling=args.rolling, seasonal_lags=args.seasonal_lags)
    # Final test and evaluation splits are deliberately not loaded here.
    fit = load_split(args.dataset, 'daily', 'fit', config)
    validation = load_split(args.dataset, 'daily', 'validation', config)
    starting_code = {**code_hashes(), Path(__file__).name: digest(__file__)}
    manifest_path = args.dataset / 'manifest.json'
    starting_manifest = digest(manifest_path)
    split_files = ('daily/fit/features.csv', 'daily/fit/labels.csv',
                   'daily/validation/features.csv', 'daily/validation/labels.csv')
    starting_splits = {name: digest(args.dataset / name) for name in split_files}
    manifest = json.loads(manifest_path.read_text())
    if any(starting_splits[name] != manifest['output_sha256'][name] for name in split_files):
        raise ValueError('Training or validation data changed after loading')

    scaler = Preprocessor.fit(fit)
    args.output.mkdir(parents=True)
    scaler.save(args.output / 'preprocessor.json')
    generator = torch.Generator().manual_seed(args.seed)
    training = loader(fit, scaler, args.batch_size, True, generator)
    validating = loader(validation, scaler, args.batch_size)
    model = MODELS[args.model](context_size=len(fit.context_columns))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = torch.nn.MSELoss()
    best_loss, best_state, best_epoch, stale = float('inf'), None, 0, 0
    started = time.monotonic()
    epoch_results = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, profile_count = 0.0, 0
        for sequence, context, target in training:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(predict_profiles(model, sequence, context), target)
            if not torch.isfinite(loss):
                raise ValueError('Training loss is not finite')
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(target)
            profile_count += len(target)
        model.eval()
        outputs = []
        with torch.no_grad():
            for sequence, context, _ in validating:
                outputs.append(predict_profiles(model, sequence, context).numpy())
        predicted = scaler.inverse_labels(np.concatenate(outputs, axis=0))
        metrics = profile_metrics(predicted, validation.labels)
        epoch_results.append({'epoch': epoch, 'fit_mse_scaled': total_loss / profile_count, **metrics})
        print(f"{args.model} epoch {epoch}: hourly MAE {metrics['hourly_mae']:.4f}, "
              f"24-hour total MAE {metrics['total_mae']:.4f}", flush=True)
        if metrics['hourly_mae'] < best_loss:
            best_loss, best_state, best_epoch, stale = metrics['hourly_mae'], copy.deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    if (code_hashes() != {k: v for k, v in starting_code.items() if k != Path(__file__).name}
            or digest(__file__) != starting_code[Path(__file__).name]
            or digest(manifest_path) != starting_manifest
            or any(digest(args.dataset / name) != expected for name, expected in starting_splits.items())):
        raise RuntimeError('Code or dataset changed during training')
    torch.save(best_state, args.output / 'model.pt')
    baselines = {name: profile_metrics(values, validation.labels)
                 for name, values in baseline_profiles(validation.history).items()}
    target_keys = '\n'.join(f'{origin}|{target}' for origin, targets in zip(validation.origins, validation.targets)
                            for target in targets)
    result = {
        'model': args.model, 'task': '24 elapsed hours from one 00:00 UTC origin',
        'seed': args.seed, 'history_hours': args.history,
        'feature_config': {'calendar': args.calendar, 'rolling': args.rolling,
                           'seasonal_lags': args.seasonal_lags},
        'features': {'history': fit.history_columns, 'context': fit.context_columns},
        'dataset_input_sha256': manifest['input_sha256'],
        'fit_profiles': len(fit.labels), 'validation_profiles': len(validation.labels),
        'batch_size_origins': args.batch_size, 'learning_rate': args.learning_rate,
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'best_epoch': best_epoch, 'best_validation': epoch_results[best_epoch - 1],
        'baselines': baselines, 'seconds': time.monotonic() - started, 'epochs': epoch_results,
        'training_provenance': {
            'dataset_manifest_sha256': starting_manifest, 'split_sha256': starting_splits,
            'validation_forecast_keys_sha256': hashlib.sha256(target_keys.encode()).hexdigest(),
            'code_sha256': starting_code,
            'python': platform.python_version(), 'numpy': np.__version__, 'torch': torch.__version__,
            'optimizer': 'Adam', 'loss': 'MSE on scaled target',
            'model_parameters': 'default constructors', 'selection': 'validation hourly MAE',
        },
    }
    (args.output / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
