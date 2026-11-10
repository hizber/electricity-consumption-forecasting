"""Recompute matched hourly validation comparisons from saved model runs."""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import torch

import evaluate
from preprocessing import Preprocessor, load_split
from train import MODELS, code_hashes


def summarize(dataset, run):
    """Load fitting and validation rows and reproduce one saved run's scores."""
    dataset = Path(dataset).resolve()
    run = evaluate._within_runs(run)
    metrics = json.loads((run / 'metrics.json').read_text(encoding='utf-8'))
    scaler = Preprocessor.load(run / 'preprocessor.json')
    if scaler.horizon != 'hourly':
        raise ValueError('Validation feature comparisons require hourly runs')
    config = evaluate._config(metrics, scaler)
    evaluate._verify_development_data(dataset, metrics, 'hourly')
    provenance = metrics.get('training_provenance')
    if (not provenance or provenance['code_sha256'] != code_hashes()
            or (provenance['python'], provenance['numpy'], provenance['torch']) !=
               (platform.python_version(), np.__version__, torch.__version__)):
        raise ValueError('Current training code and environment must match the recorded run provenance')
    fit = load_split(dataset, 'hourly', 'fit', config)
    validation = load_split(dataset, 'hourly', 'validation', config)
    model = MODELS[metrics['model']](context_size=len(scaler.context_columns))
    model.load_state_dict(torch.load(run / 'model.pt', map_location='cpu', weights_only=True), strict=True)
    predicted = evaluate.predict(validation, scaler, model, metrics['batch_size'])
    baseline = validation.history[:, -24, 0]
    result = evaluate.score(validation.labels, predicted)
    if (not np.isclose(result['mae'], metrics['validation_mae'], rtol=0, atol=1e-6)
            or not np.isclose(result['rmse'], metrics['validation_rmse_at_best_epoch'],
                              rtol=0, atol=1e-6)):
        raise ValueError('Recomputed validation scores differ from the recorded run')
    target_hash = hashlib.sha256('\n'.join(validation.targets).encode()).hexdigest()
    if target_hash != provenance['validation_targets_sha256']:
        raise ValueError('Validation target keys differ from the training record')
    label_hash = hashlib.sha256(validation.labels.astype('<f8').tobytes()).hexdigest()
    missing = {}
    for data in (fit, validation):
        missing[data.split] = {
            name: int(np.isnan(data.context[:, index]).sum())
            for index, name in enumerate(data.context_columns)
            if name.startswith(('temperature_', 'humidity_', 'wind_speed_'))
            or name == 'heating_degrees_15c'
        }
    manifest = json.loads((dataset / 'manifest.json').read_text(encoding='utf-8'))
    return {
        'model': metrics['model'], 'seed': metrics['seed'],
        'history_hours': metrics['history_hours'],
        'features': metrics['features'], 'feature_config': metrics.get('feature_config'),
        'best_epoch': metrics['best_epoch'], 'training_seconds': metrics['seconds'],
        'batch_size': metrics['batch_size'], 'learning_rate': metrics['learning_rate'],
        'optimizer': provenance['optimizer'], 'loss': provenance['loss'],
        'parameter_count': metrics['parameter_count'],
        'weather_availability': manifest.get('weather_availability') if (
            config.weather_temperature or config.weather_humidity or config.weather_wind) else None,
        'dataset_manifest_sha256': evaluate.digest(dataset / 'manifest.json'),
        'validation_target_sha256': target_hash, 'validation_labels_sha256': label_hash,
        'model_scores': result, 'previous_day_scores': evaluate.score(validation.labels, baseline),
        'slices': evaluate.slices(validation.targets, validation.labels, predicted, baseline),
        'missing_before_imputation': missing,
    }


def summarize_experiments(experiments):
    results = {}
    for name, dataset, run in experiments:
        if name in results:
            raise ValueError(f'duplicate experiment name: {name}')
        results[name] = summarize(dataset, run)
    targets = {(item['validation_target_sha256'], item['validation_labels_sha256'])
               for item in results.values()}
    if len(targets) != 1:
        raise ValueError('Experiments do not use identical validation targets and labels')
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', action='append', nargs=3, metavar=('NAME', 'DATASET', 'RUN'),
                        required=True, help='Repeat for every model to compare')
    parser.add_argument('--output', type=Path, required=True, help='New JSON file within models/runs')
    args = parser.parse_args()
    torch.set_num_threads(2)
    output = evaluate._within_runs(args.output)
    if output.exists():
        parser.error('output file already exists')
    results = summarize_experiments(args.experiment)
    with output.open('x', encoding='utf-8') as handle:
        json.dump(results, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(f'Compared {len(results)} experiments on matching validation targets')


if __name__ == '__main__':
    main()
