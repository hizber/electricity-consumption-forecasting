"""Train one shared starting-point model on the hourly fitting split."""

import argparse
import copy
import hashlib
import json
import platform
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'model_data'))
from preprocessing import FeatureConfig, Preprocessor, load_split
from recurrent import GRUForecaster, LSTMForecaster
from temporal import TCNRegressor, TransformerRegressor


MODELS = {'lstm': LSTMForecaster, 'gru': GRUForecaster,
          'tcn': TCNRegressor, 'transformer': TransformerRegressor}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def code_hashes():
    files = (Path(__file__), Path(__file__).with_name('recurrent.py'),
             Path(__file__).with_name('temporal.py'), Path(__file__).resolve().parents[1] / 'model_data/preprocessing.py')
    return {path.name: digest(path) for path in files}


def loader(dataset, scaler, batch_size, shuffle=False, generator=None):
    history, context, labels = scaler.transform(dataset)
    arrays = [torch.from_numpy(np.asarray(x, dtype=np.float32)) for x in (history, context, labels)]
    return DataLoader(TensorDataset(*arrays), batch_size=batch_size, shuffle=shuffle,
                      generator=generator, num_workers=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--history', type=int, choices=(24, 72, 168), default=24)
    parser.add_argument('--calendar', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--rolling', action='store_true', help='Add rolling summaries within the selected history')
    parser.add_argument('--weather-temperature', action='store_true')
    parser.add_argument('--weather-temperature-rolling', action='store_true')
    parser.add_argument('--weather-humidity', action='store_true')
    parser.add_argument('--weather-wind', action='store_true')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=128)
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
    config = FeatureConfig(history_hours=args.history, calendar=args.calendar, rolling=args.rolling,
                           weather_temperature=args.weather_temperature,
                           weather_temperature_rolling=args.weather_temperature_rolling,
                           weather_humidity=args.weather_humidity, weather_wind=args.weather_wind)
    fit = load_split(args.dataset, 'hourly', 'fit', config)
    validation = load_split(args.dataset, 'hourly', 'validation', config)
    starting_code = code_hashes()
    starting_manifest = digest(args.dataset / 'manifest.json')
    split_files = ('hourly/fit/features.csv', 'hourly/fit/labels.csv',
                   'hourly/validation/features.csv', 'hourly/validation/labels.csv')
    starting_splits = {name: digest(args.dataset / name) for name in split_files}
    manifest = json.loads((args.dataset / 'manifest.json').read_text())
    if any(starting_splits[name] != manifest['output_sha256'][name] for name in split_files):
        raise ValueError('Training or validation data changed after loading')
    scaler = Preprocessor.fit(fit)
    args.output.mkdir(parents=True)
    scaler.save(args.output / 'preprocessor.json')
    random_generator = torch.Generator().manual_seed(args.seed)
    training = loader(fit, scaler, args.batch_size, True, random_generator)
    validating = loader(validation, scaler, args.batch_size)
    model = MODELS[args.model](context_size=len(fit.context_columns))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = torch.nn.MSELoss()
    best_loss, best_state, best_epoch, stale = float('inf'), None, 0, 0
    started = time.monotonic()
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, count = 0.0, 0
        for sequence, context, target in training:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(sequence, context), target)
            if not torch.isfinite(loss):
                raise ValueError('Training loss is not finite')
            loss.backward()
            optimizer.step()
            total += loss.item() * len(target)
            count += len(target)
        model.eval()
        predictions = []
        with torch.no_grad():
            for sequence, context, _ in validating:
                predictions.append(model(sequence, context).numpy())
        predicted = scaler.inverse_labels(np.concatenate(predictions))
        errors = predicted - validation.labels
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        if not np.isfinite(mae) or not np.isfinite(rmse):
            raise ValueError('Validation metrics are not finite')
        history.append({'epoch': epoch, 'fit_mse_scaled': total / count, 'validation_mae': mae, 'validation_rmse': rmse})
        print(f'{args.model} epoch {epoch}: MAE {mae:.4f}, RMSE {rmse:.4f}', flush=True)
        if mae < best_loss:
            best_loss, best_state, best_epoch, stale = mae, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if (code_hashes() != starting_code or digest(args.dataset / 'manifest.json') != starting_manifest
            or any(digest(args.dataset / name) != expected for name, expected in starting_splits.items())):
        raise RuntimeError('Code or dataset changed during training')
    torch.save(best_state, args.output / 'model.pt')
    target_before = validation.history[:, -24, 0]
    baseline_errors = target_before - validation.labels
    result = {
        'model': args.model, 'seed': args.seed, 'history_hours': args.history,
        'feature_config': asdict(config),
        'weather_availability': manifest.get('weather_availability') if
        any((args.weather_temperature, args.weather_humidity, args.weather_wind)) else None,
        'features': {'history': fit.history_columns, 'context': fit.context_columns},
        'dataset_input_sha256': json.loads((args.dataset / 'manifest.json').read_text())['input_sha256'],
        'fit_rows': len(fit.labels), 'validation_rows': len(validation.labels),
        'batch_size': args.batch_size, 'learning_rate': args.learning_rate,
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'best_epoch': best_epoch, 'validation_mae': best_loss,
        'validation_rmse_at_best_epoch': history[best_epoch - 1]['validation_rmse'],
        'previous_day_mae': float(np.mean(np.abs(baseline_errors))),
        'previous_day_rmse': float(np.sqrt(np.mean(baseline_errors ** 2))),
        'seconds': time.monotonic() - started, 'epochs': history,
        'training_provenance': {
            'dataset_manifest_sha256': starting_manifest,
            'split_sha256': starting_splits,
            'validation_targets_sha256': hashlib.sha256('\n'.join(validation.targets).encode()).hexdigest(),
            'code_sha256': starting_code,
            'python': platform.python_version(), 'numpy': np.__version__, 'torch': torch.__version__,
            'optimizer': 'Adam', 'loss': 'MSE on scaled target', 'model_parameters': 'default constructors',
        },
    }
    (args.output / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
