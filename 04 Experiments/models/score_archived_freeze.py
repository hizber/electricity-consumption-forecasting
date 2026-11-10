"""Recalculate metrics for a frozen run using its original code snapshot."""

import argparse
import hashlib
import importlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True,
                        help='Archived repository containing the original model code and linked runs')
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--freeze-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New JSON file inside models/runs')
    parser.add_argument('--splits', nargs='+',
                        choices=('validation', 'test', 'evaluation_1', 'evaluation_2', 'evaluation_3'),
                        default=('validation', 'test', 'evaluation_1', 'evaluation_2', 'evaluation_3'))
    args = parser.parse_args()
    if len(set(args.splits)) != len(args.splits):
        parser.error('split names must be unique')
    torch.set_num_threads(2)
    spec = importlib.util.spec_from_file_location('current_scoring_metrics', Path(__file__).with_name('metrics.py'))
    current_metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(current_metrics)
    snapshot_models = (args.snapshot / '04 Experiments/models').resolve()
    sys.path.insert(0, str(snapshot_models))
    archived = importlib.import_module('evaluate')
    if Path(archived.__file__).resolve() != snapshot_models / 'evaluate.py':
        raise ValueError('Could not load the requested archived evaluator')
    output = archived._within_runs(args.output)
    if output.exists():
        parser.error('output file already exists')
    record, metrics, scaler, config, model = archived._verify_freeze(args.dataset, args.freeze_file)
    batch_size = metrics['batch_size_origins' if scaler.horizon == 'daily' else 'batch_size']
    results = {}
    for split in args.splits:
        data = archived.load_split(args.dataset, scaler.horizon, split, config, verify_hashes=True)
        predicted = archived.predict(data, scaler, model, batch_size)
        if scaler.horizon == 'daily':
            baseline = archived.baseline_profiles(data.history)['previous_day']
            period = {
                'profiles': len(data.labels),
                'model_hourly': current_metrics.score(data.labels, predicted),
                'previous_day_hourly': current_metrics.score(data.labels, baseline),
                'model_total': current_metrics.score(data.labels.sum(axis=1), predicted.sum(axis=1)),
                'previous_day_total': current_metrics.score(data.labels.sum(axis=1), baseline.sum(axis=1)),
            }
        else:
            baseline = data.history[:, -24, 0]
            period = {'model': current_metrics.score(data.labels, predicted),
                      'previous_day': current_metrics.score(data.labels, baseline)}
        period['target_times_sha256'] = hashlib.sha256('\n'.join(data.targets.ravel()).encode()).hexdigest()
        results[split] = period
    report = {'model': record['model'], 'task': scaler.horizon,
              'freeze_sha256': archived.digest(args.freeze_file),
              'dataset_manifest_sha256': record['dataset_manifest_sha256'],
              'scoring_code_sha256': archived.digest(__file__),
              'metrics_code_sha256': archived.digest(Path(__file__).with_name('metrics.py')),
              'periods': results}
    with output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(f"Recalculated {record['model']} {scaler.horizon} metrics for {len(results)} periods")


if __name__ == '__main__':
    main()
