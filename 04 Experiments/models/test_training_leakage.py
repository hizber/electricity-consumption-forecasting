"""Regression checks for training and preprocessing isolation from validation."""

import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import torch

import evaluate
from summarize_validation import summarize_experiments

HERE = Path(__file__).resolve().parent


def write_dataset(root, task, validation_offset):
    outputs, counts, boundaries = {}, {}, {}
    for split, year in (('fit', 2019), ('validation', 2020)):
        start = datetime(year, 1, 1, tzinfo=timezone.utc)
        boundaries[split] = {'start_inclusive': start.isoformat(),
                             'end_exclusive': (start + timedelta(days=5)).isoformat()}
        folder = root / task / split
        folder.mkdir(parents=True)
        history = [f'load_lag_{i:03d}' for i in range(1, 25)]
        names = ['origin_utc', 'target_end_utc', *history]
        if task == 'daily':
            names.append('horizon_hours')
        features, labels = [], []
        for profile in range(2):
            origin = start + timedelta(days=profile)
            horizons = range(1, 25) if task == 'daily' else (1, 2)
            for horizon in horizons:
                row_origin = origin if task == 'daily' else origin + timedelta(hours=horizon - 1)
                target = origin + timedelta(hours=horizon)
                keys = {'origin_utc': row_origin.isoformat(), 'target_end_utc': target.isoformat()}
                row = {**keys, **{name: (index + profile + 1) / 10
                                  for index, name in enumerate(history)}}
                if task == 'daily':
                    row['horizon_hours'] = horizon
                features.append(row)
                labels.append({**keys, 'consumption': (profile + horizon) / 10 +
                               (validation_offset if split == 'validation' else 0)})
        for filename, fields, rows in (
                ('features.csv', names, features),
                ('labels.csv', ['origin_utc', 'target_end_utc', 'consumption'], labels)):
            path = folder / filename
            with path.open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            outputs[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        counts[split] = len(features)
    manifest = {'input_sha256': 'synthetic', 'output_sha256': outputs,
                'split_boundaries_utc': boundaries, task: {'rows': counts}}
    (root / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')


class TrainingLeakageTests(unittest.TestCase):
    def test_training_rejects_overlapping_fit_and_validation_periods(self):
        for task, script in (('hourly', 'train.py'), ('daily', 'train_daily.py')):
            with self.subTest(task=task), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / 'dataset'
                write_dataset(root, task, 0)
                manifest_path = root / 'manifest.json'
                manifest = json.loads(manifest_path.read_text())
                for filename in ('features.csv', 'labels.csv'):
                    fit = root / task / 'fit' / filename
                    shutil.copyfile(root / task / 'validation' / filename, fit)
                    manifest['output_sha256'][fit.relative_to(root).as_posix()] = hashlib.sha256(
                        fit.read_bytes()).hexdigest()
                manifest['split_boundaries_utc']['fit']['end_exclusive'] = '2020-01-05T00:00:00+00:00'
                manifest_path.write_text(json.dumps(manifest))
                output = Path(directory) / 'run'
                result = subprocess.run([
                    sys.executable, str(HERE / script), '--dataset', str(root),
                    '--output', str(output), '--model', 'gru', '--history', '24',
                    '--no-calendar', '--epochs', '1',
                ], capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('Split boundaries overlap or are out of order', result.stderr)
                self.assertFalse(output.exists())

    def test_validation_labels_do_not_change_one_epoch_fitting(self):
        for task, script in (('hourly', 'train.py'), ('daily', 'train_daily.py')):
            with self.subTest(task=task), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                runs = []
                for name, offset in (('original', 0), ('changed', 1000)):
                    dataset, output = root / f'data_{name}', root / f'run_{name}'
                    write_dataset(dataset, task, offset)
                    subprocess.run([
                        sys.executable, str(HERE / script), '--dataset', str(dataset),
                        '--output', str(output), '--model', 'gru', '--history', '24',
                        '--no-calendar', '--seed', '42', '--epochs', '1', '--batch-size', '2',
                    ], check=True, capture_output=True, text=True)
                    self.assertFalse((dataset / task / 'test').exists())
                    runs.append(output)
                first, second = runs
                self.assertEqual((first / 'preprocessor.json').read_bytes(),
                                 (second / 'preprocessor.json').read_bytes())
                first_model = torch.load(first / 'model.pt', map_location='cpu', weights_only=True)
                second_model = torch.load(second / 'model.pt', map_location='cpu', weights_only=True)
                for key in first_model:
                    torch.testing.assert_close(first_model[key], second_model[key], rtol=0, atol=0)
                first_metrics = json.loads((first / 'metrics.json').read_text())
                second_metrics = json.loads((second / 'metrics.json').read_text())
                self.assertEqual(first_metrics['epochs'][0]['fit_mse_scaled'],
                                 second_metrics['epochs'][0]['fit_mse_scaled'])
                mae = 'validation_mae' if task == 'hourly' else 'hourly_mae'
                self.assertNotEqual(first_metrics['epochs'][0][mae], second_metrics['epochs'][0][mae])
                if task == 'hourly':
                    experiments = [('original', root / 'data_original', first),
                                   ('changed', root / 'data_changed', second)]
                    with patch.object(evaluate, 'RUNS', root):
                        with self.assertRaisesRegex(ValueError, 'identical validation targets and labels'):
                            summarize_experiments(experiments)


if __name__ == '__main__':
    unittest.main()
