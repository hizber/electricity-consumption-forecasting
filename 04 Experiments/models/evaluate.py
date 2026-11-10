"""Freeze a trained run, or explicitly evaluate it on held-out periods.

Run --help for the two modes. Creating a freeze reads fitting and validation
artifacts only. Evaluating requires a previously created freeze and an
explicit --evaluate flag. This tool never tunes a model or its preprocessing.
"""

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch

from train import MODELS, loader
from train_daily import baseline_profiles, predict_profiles, profile_metrics

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'model_data'))
from preprocessing import FeatureConfig, Preprocessor, load_split


HERE = Path(__file__).resolve().parent
RUNS = HERE / 'runs'
HELDOUT = ('test', 'evaluation_1', 'evaluation_2', 'evaluation_3')
CODE = (HERE / 'evaluate.py', HERE / 'train.py', HERE / 'train_daily.py', HERE / 'recurrent.py', HERE / 'temporal.py',
        HERE.parent / 'model_data' / 'preprocessing.py')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _within_runs(path):
    path = Path(path).resolve()
    if not path.is_relative_to(RUNS.resolve()) or path == RUNS.resolve():
        raise ValueError('Run, freeze and output paths must be inside models/runs')
    return path


def _config(metrics, scaler):
    if metrics['model'] not in MODELS:
        raise ValueError('Unknown model in training metrics')
    selection = metrics.get('feature_config', {'calendar': True, 'rolling': False})
    config = FeatureConfig(**{'history_hours': metrics['history_hours'], **selection})
    expected_task = 'daily' if 'batch_size_origins' in metrics else 'hourly'
    if (scaler.horizon != expected_task or scaler.history_columns != config.history_columns()
            or scaler.context_columns != config.context_columns(expected_task)):
        raise ValueError('Saved preprocessing schema differs from the training configuration')
    if (metrics['features']['history'] != list(scaler.history_columns)
            or metrics['features']['context'] != list(scaler.context_columns)):
        raise ValueError('Training metrics and preprocessing feature schemas differ')
    return config


def _verify_development_data(dataset, metrics, task):
    manifest_path = dataset / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest['input_sha256'] != metrics['dataset_input_sha256']:
        raise ValueError('Dataset source hash differs from the training run')
    split_hashes = {}
    for split in ('fit', 'validation'):
        for filename in ('features.csv', 'labels.csv'):
            relative = f'{task}/{split}/{filename}'
            split_hashes[relative] = digest(dataset / relative)
            if split_hashes[relative] != manifest['output_sha256'][relative]:
                raise ValueError(f'Dataset hash mismatch: {relative}')
    provenance = metrics.get('training_provenance')
    if provenance is not None and (
        provenance['dataset_manifest_sha256'] != digest(manifest_path)
        or provenance['split_sha256'] != split_hashes
    ):
        raise ValueError('Dataset differs from the data recorded during training')
    return manifest_path


def create_freeze(dataset, run, freeze_file):
    """Record current artifacts and code without reading any held-out rows."""
    dataset = Path(dataset).resolve()
    run, freeze_file = _within_runs(run), _within_runs(freeze_file)
    if freeze_file.parent != run:
        raise ValueError('Freeze file must be directly inside the training run directory')
    if freeze_file.exists():
        raise FileExistsError('Freeze file already exists')
    metrics_path = run / 'metrics.json'
    scaler_path = run / 'preprocessor.json'
    model_path = run / 'model.pt'
    metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
    scaler = Preprocessor.load(scaler_path)
    _config(metrics, scaler)
    manifest_path = _verify_development_data(dataset, metrics, scaler.horizon)
    provenance = metrics.get('training_provenance')
    if provenance is not None:
        training_hashes = {p.name: digest(p) for p in CODE if p.name != 'evaluate.py'
                           and (p.name != 'train_daily.py' or scaler.horizon == 'daily')}
        if (provenance['code_sha256'] != training_hashes
                or (provenance['python'], provenance['numpy'], provenance['torch']) != (
                    platform.python_version(), np.__version__, torch.__version__)):
            raise ValueError('Code or environment differs from the record captured during training')
    model = MODELS[metrics['model']](context_size=len(scaler.context_columns))
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True), strict=True)
    if sum(p.numel() for p in model.parameters()) != metrics['parameter_count']:
        raise ValueError('Checkpoint parameter count differs from recorded metrics')
    record = {
        'version': 1,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'run_path': str(run.relative_to(RUNS.resolve())),
        'model': metrics['model'],
        'task': scaler.horizon,
        'history_hours': metrics['history_hours'],
        'dataset_manifest_sha256': digest(manifest_path),
        'dataset_source_sha256': metrics['dataset_input_sha256'],
        'artifacts_sha256': {p.name: digest(p) for p in (metrics_path, scaler_path, model_path)},
        'current_code_sha256': {str(p.relative_to(HERE.parent)): digest(p) for p in CODE},
        'current_python': platform.python_version(),
        'current_numpy': np.__version__,
        'current_torch': torch.__version__,
        'provenance_limit': (None if provenance is not None else
                             'This run predates training-time provenance. The freeze captures '
                             'current code and artifacts without verifying the training environment.'),
    }
    with freeze_file.open('x', encoding='utf-8') as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
        handle.write('\n')
    return record


def _verify_freeze(dataset, freeze_file):
    dataset = Path(dataset).resolve()
    freeze_file = _within_runs(freeze_file)
    record = json.loads(freeze_file.read_text(encoding='utf-8'))
    if record['version'] != 1:
        raise ValueError('Unsupported freeze version')
    run = _within_runs(RUNS / record['run_path'])
    if freeze_file.parent != run:
        raise ValueError('Freeze file is not in its recorded run directory')
    if digest(dataset / 'manifest.json') != record['dataset_manifest_sha256']:
        raise ValueError('Dataset manifest differs from frozen manifest')
    for name, expected in record['artifacts_sha256'].items():
        if name not in ('metrics.json', 'preprocessor.json', 'model.pt') or digest(run / name) != expected:
            raise ValueError(f'Training artifact differs from freeze: {name}')
    if set(record['artifacts_sha256']) != {'metrics.json', 'preprocessor.json', 'model.pt'}:
        raise ValueError('Freeze does not identify all required training artifacts')
    actual_code = {str(p.relative_to(HERE.parent)): digest(p) for p in CODE}
    if actual_code != record['current_code_sha256']:
        raise ValueError('Model or preprocessing code differs from freeze')
    if (platform.python_version(), np.__version__, torch.__version__) != (
            record['current_python'], record['current_numpy'], record['current_torch']):
        raise ValueError('Python, NumPy or PyTorch version differs from freeze')
    metrics = json.loads((run / 'metrics.json').read_text(encoding='utf-8'))
    scaler = Preprocessor.load(run / 'preprocessor.json')
    config = _config(metrics, scaler)
    _verify_development_data(dataset, metrics, scaler.horizon)
    if (record['model'], record['task'], record['history_hours'], record['dataset_source_sha256']) != (
            metrics['model'], scaler.horizon, metrics['history_hours'], metrics['dataset_input_sha256']):
        raise ValueError('Freeze does not match training metrics')
    model = MODELS[metrics['model']](context_size=len(scaler.context_columns))
    model.load_state_dict(torch.load(run / 'model.pt', map_location='cpu', weights_only=True), strict=True)
    model.eval()
    return record, metrics, scaler, config, model


def score(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if actual.shape != predicted.shape or not actual.size:
        raise ValueError('Actual and predicted arrays must have matching nonempty shapes')
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError('Metrics require finite values')
    errors = predicted - actual
    return {'count': int(actual.size), 'mae': float(np.abs(errors).mean()),
            'rmse': float(np.sqrt(np.mean(errors ** 2)))}


def slices(targets, actual, predicted, baseline):
    """Summarize errors by target-end Helsinki hour and meteorological season."""
    local = [datetime.fromisoformat(t).astimezone(ZoneInfo('Europe/Helsinki')) for t in targets]
    seasons = {12: 'winter', 1: 'winter', 2: 'winter', 3: 'spring', 4: 'spring', 5: 'spring',
               6: 'summer', 7: 'summer', 8: 'summer', 9: 'autumn', 10: 'autumn', 11: 'autumn'}
    result = {}
    for name, keys in (('hour', [str(t.hour) for t in local]),
                       ('season', [seasons[t.month] for t in local])):
        result[name] = {}
        keys = np.asarray(keys)
        for key in sorted(set(keys), key=lambda item: int(item) if name == 'hour' else item):
            mask = keys == key
            result[name][key] = {'model': score(actual[mask], predicted[mask]),
                                 'previous_day': score(actual[mask], baseline[mask])}
    return result


def largest_error_days(targets, actual, predicted, baseline, limit=5):
    """Rank complete UTC target days by model MAE, with baseline errors for context."""
    days = np.array([(datetime.fromisoformat(t).astimezone(timezone.utc).date().isoformat())
                     for t in targets])
    scores = []
    for day, count in zip(*np.unique(days, return_counts=True)):
        if count != 24:
            continue
        mask = days == day
        scores.append({'target_date_utc': str(day), 'model': score(actual[mask], predicted[mask]),
                       'previous_day': score(actual[mask], baseline[mask]),
                       'mean_signed_error': float((predicted[mask] - actual[mask]).mean())})
    return sorted(scores, key=lambda day: day['model']['mae'], reverse=True)[:limit]


def predict(data, scaler, model, batch_size):
    model.eval()
    predictions = []
    forward = predict_profiles if scaler.horizon == 'daily' else lambda m, h, c: m(h, c)
    with torch.no_grad():
        for sequence, context, _ in loader(data, scaler, batch_size):
            predictions.append(forward(model, sequence, context).numpy())
    return scaler.inverse_labels(np.concatenate(predictions))


def period_metrics(data, predicted):
    if data.horizon == 'daily':
        baselines = baseline_profiles(data.history)
        return {'profiles': len(data.labels), 'model': profile_metrics(predicted, data.labels),
                **{name: profile_metrics(values, data.labels) for name, values in baselines.items()},
                'hourly_slices': slices(data.targets.ravel(), data.labels.ravel(),
                                        predicted.ravel(), baselines['previous_day'].ravel())}
    baseline = data.history[:, -24, 0]
    return {'model': score(data.labels, predicted), 'previous_day': score(data.labels, baseline),
            'slices': slices(data.targets, data.labels, predicted, baseline),
            'largest_error_days': largest_error_days(data.targets, data.labels, predicted, baseline)}


def evaluate(dataset, freeze_file, output, splits=HELDOUT):
    """Evaluate the frozen model without retraining or saving actual labels."""
    dataset, output = Path(dataset).resolve(), _within_runs(output)
    if output.exists():
        raise FileExistsError('Output directory already exists')
    if not splits or len(set(splits)) != len(splits) or any(s not in HELDOUT for s in splits):
        raise ValueError('Only unique held-out split names may be evaluated')
    freeze_file = _within_runs(freeze_file)
    record, metrics, scaler, config, model = _verify_freeze(dataset, freeze_file)
    results = {}
    batch_size = metrics['batch_size_origins' if scaler.horizon == 'daily' else 'batch_size']
    for split in splits:
        data = load_split(dataset, scaler.horizon, split, config, verify_hashes=True)
        predicted = predict(data, scaler, model, batch_size)
        targets = data.targets.ravel()
        results[split] = {**period_metrics(data, predicted),
                          'first_target_utc': str(targets[0]), 'last_target_utc': str(targets[-1]),
                          'target_times_sha256': hashlib.sha256('\n'.join(targets).encode()).hexdigest()}
    output.mkdir(parents=True)
    report = {'freeze_sha256': digest(freeze_file), 'model': record['model'], 'task': scaler.horizon,
              'dataset_manifest_sha256': record['dataset_manifest_sha256'],
              'provenance_limit': record['provenance_limit'], 'periods': results}
    (output / 'evaluation.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--create-freeze', action='store_true', help='Freeze a training run without reading held-out data')
    mode.add_argument('--evaluate', action='store_true', help='Evaluate held-out periods using an existing freeze')
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--run', type=Path, help='Training run directory; required to create a freeze')
    parser.add_argument('--freeze-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, help='New evaluation output directory under models/runs')
    parser.add_argument('--splits', nargs='+', choices=HELDOUT, default=list(HELDOUT))
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.create_freeze:
        if args.run is None or args.output is not None:
            parser.error('--create-freeze requires --run and does not use --output')
        create_freeze(args.dataset, args.run, args.freeze_file)
        print(f'Created freeze: {args.freeze_file}')
    else:
        if args.output is None or args.run is not None:
            parser.error('--evaluate requires --output; the freeze identifies the run')
        result = evaluate(args.dataset, args.freeze_file, args.output, args.splits)
        print(json.dumps({'model': result['model'], 'periods': list(result['periods'])}, indent=2))


if __name__ == '__main__':
    main()
