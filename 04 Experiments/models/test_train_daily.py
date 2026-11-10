"""Checks for fixed-origin daily training and baseline alignment."""

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

from train_daily import baseline_profiles, predict_profiles, profile_metrics
from train import MODELS
from preprocessing import CALENDAR


class DailyTrainerTests(unittest.TestCase):
    def test_all_models_use_same_history_at_every_horizon(self):
        torch.manual_seed(7)
        sequence = torch.arange(48, dtype=torch.float32).reshape(2, 24, 1)
        context = torch.randn(2, 24, 2)
        for name, cls in MODELS.items():
            with self.subTest(model=name):
                model = cls(context_size=2).eval()
                with torch.no_grad():
                    all_steps = predict_profiles(model, sequence, context)
                    for horizon in (0, 12, 23):
                        standalone = model(sequence, context[:, horizon, :])
                        torch.testing.assert_close(all_steps[:, horizon], standalone, rtol=1e-5, atol=1e-6)
                self.assertEqual(tuple(all_steps.shape), (2, 24))

    def test_baselines_and_total_metrics(self):
        history = np.arange(168, dtype=float).reshape(1, 168, 1)
        baseline = baseline_profiles(history)
        np.testing.assert_array_equal(baseline['previous_day'], np.arange(144, 168).reshape(1, 24))
        np.testing.assert_array_equal(baseline['previous_week'], np.arange(24).reshape(1, 24))
        metrics = profile_metrics(np.ones((2, 24)), np.zeros((2, 24)))
        self.assertEqual(metrics, {'hourly_mae': 1, 'hourly_rmse': 1,
                                   'total_mae': 24, 'total_rmse': 24})

    def test_cli_reads_only_fit_and_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'data'
            split_spec = {'fit': ('2019-01-01', '2019-01-03'),
                          'validation': ('2020-01-01', '2020-01-03')}
            outputs, boundaries, counts = {}, {}, {}
            for split, (start_text, end_text) in split_spec.items():
                start = datetime.fromisoformat(start_text).replace(tzinfo=timezone.utc)
                end = datetime.fromisoformat(end_text).replace(tzinfo=timezone.utc)
                boundaries[split] = {'start_inclusive': start.isoformat(), 'end_exclusive': end.isoformat()}
                folder = root / 'daily' / split
                folder.mkdir(parents=True)
                keys = [f'load_lag_{n:03d}' for n in range(1, 25)]
                feature_path, label_path = folder / 'features.csv', folder / 'labels.csv'
                with feature_path.open('w', newline='') as xf, label_path.open('w', newline='') as yf:
                    writer = csv.DictWriter(xf, fieldnames=['origin_utc', 'target_end_utc', 'horizon_hours', *keys, *CALENDAR])
                    actual = csv.DictWriter(yf, fieldnames=['origin_utc', 'target_end_utc', 'consumption'])
                    writer.writeheader()
                    actual.writeheader()
                    for step in range(1, 25):
                        target = (start + timedelta(hours=step)).isoformat()
                        row = {'origin_utc': start.isoformat(), 'target_end_utc': target,
                               'horizon_hours': step, **{key: float(i + 1) for i, key in enumerate(keys)},
                               **{key: 0 for key in CALENDAR}}
                        writer.writerow(row)
                        actual.writerow({'origin_utc': start.isoformat(), 'target_end_utc': target,
                                         'consumption': float(step + 1)})
                counts[split] = 24
                for path in (feature_path, label_path):
                    outputs[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest = {'input_sha256': 'synthetic', 'split_boundaries_utc': boundaries,
                        'output_sha256': outputs, 'daily': {'rows': counts}}
            (root / 'manifest.json').write_text(json.dumps(manifest))
            output = Path(tmp) / 'run'
            subprocess.run([sys.executable, str(Path(__file__).with_name('train_daily.py')),
                            '--dataset', str(root), '--model', 'gru', '--output', str(output),
                            '--history', '24', '--epochs', '1', '--batch-size', '1'],
                           check=True, capture_output=True, text=True)
            result = json.loads((output / 'metrics.json').read_text())
            self.assertEqual((result['fit_profiles'], result['validation_profiles']), (1, 1))
            self.assertIn('previous_day', result['baselines'])
            self.assertNotIn('previous_week', result['baselines'])
            self.assertEqual(set(result['training_provenance']['split_sha256']), set(outputs))
            self.assertFalse((root / 'daily' / 'test').exists())


if __name__ == '__main__':
    unittest.main()
