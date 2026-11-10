import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import evaluate
from preprocessing import Dataset, FeatureConfig, Preprocessor


class FakeModel(torch.nn.Module):
    def __init__(self, context_size):
        super().__init__()
        self.head = torch.nn.Linear(context_size + 1, 1)

    def forward(self, history, context):
        return self.head(torch.cat((history[:, -1, :], context), dim=1)).squeeze(-1)


class EvaluationTests(unittest.TestCase):
    def test_fixed_origin_period_reports_hourly_and_total_errors(self):
        origin = datetime(2020, 1, 1, tzinfo=timezone.utc)
        targets = np.array([[(origin + timedelta(hours=h)).isoformat() for h in range(1, 25)]])
        sample = Dataset('test', 'daily', (), (), np.array([origin.isoformat()]), targets,
                         np.zeros((1, 24, 1)), np.zeros((1, 24, 1)), np.ones((1, 24)))
        report = evaluate.period_metrics(sample, np.full((1, 24), 3.))
        self.assertEqual(report['profiles'], 1)
        self.assertEqual(report['model']['hourly_mae'], 2.)
        self.assertEqual(report['model']['total_mae'], 48.)
        self.assertEqual(report['previous_day']['total_mae'], 24.)
        self.assertEqual(report['hourly_slices']['season']['winter']['model']['count'], 24)

    def test_metrics_and_local_time_slices(self):
        actual = np.array([1., 2., 3.])
        predicted = np.array([2., 0., 3.])
        baseline = np.array([1., 1., 1.])
        self.assertEqual(evaluate.score(actual, predicted)['count'], 3)
        self.assertAlmostEqual(evaluate.score(actual, predicted)['mae'], 1.)
        parts = evaluate.slices(['2020-01-01T00:00:00+00:00',
                                 '2020-06-01T00:00:00+00:00',
                                 '2020-06-01T01:00:00+00:00'], actual, predicted, baseline)
        self.assertEqual(parts['season']['winter']['model']['count'], 1)
        self.assertEqual(parts['season']['summer']['model']['count'], 2)
        self.assertEqual(parts['hour']['2']['model']['count'], 1)
        self.assertEqual(parts['hour']['3']['model']['count'], 1)
        self.assertEqual(parts['hour']['4']['model']['count'], 1)

    def test_freeze_detects_artifact_and_dataset_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / 'runs'
            run = runs / 'model'
            run.mkdir(parents=True)
            dataset = root / 'dataset'
            dataset.mkdir()
            config = FeatureConfig(history_hours=24)
            history = config.history_columns()
            context = config.context_columns('hourly')
            scaler = Preprocessor('hourly', history, context, np.zeros(len(history) + len(context)),
                                  np.ones(len(history) + len(context)), 0., 1.)
            scaler.save(run / 'preprocessor.json')
            model = FakeModel(len(context))
            torch.save(model.state_dict(), run / 'model.pt')
            metrics = {'model': 'lstm', 'history_hours': 24,
                       'features': {'history': list(history), 'context': list(context)},
                       'dataset_input_sha256': 'source-hash',
                       'batch_size': 1,
                       'parameter_count': sum(p.numel() for p in model.parameters())}
            (run / 'metrics.json').write_text(json.dumps(metrics))
            hashes = {}
            for split in ('fit', 'validation'):
                folder = dataset / 'hourly' / split
                folder.mkdir(parents=True)
                for file in ('features.csv', 'labels.csv'):
                    path = folder / file
                    path.write_text('synthetic only\n')
                    hashes[f'hourly/{split}/{file}'] = evaluate.digest(path)
            (dataset / 'manifest.json').write_text(json.dumps({'input_sha256': 'source-hash', 'output_sha256': hashes}))
            freeze = run / 'freeze.json'
            with patch.object(evaluate, 'RUNS', runs), patch.dict(evaluate.MODELS, {'lstm': FakeModel}):
                evaluate.create_freeze(dataset, run, freeze)
                evaluate._verify_freeze(dataset, freeze)
                sample = Dataset('test', 'hourly', history, context,
                                 np.array(['2020-10-01T00:00:00+00:00']),
                                 np.array(['2020-10-01T01:00:00+00:00']),
                                 np.ones((1, 24, 1)), np.zeros((1, len(context))), np.array([2.]))
                with patch.object(evaluate, 'load_split', return_value=sample) as load:
                    report = evaluate.evaluate(dataset, freeze, runs / 'test-output', ('test',))
                    self.assertEqual(report['periods']['test']['model']['count'], 1)
                    self.assertEqual(report['periods']['test']['previous_day']['mae'], 1.)
                    self.assertNotIn('actual', json.dumps(report))
                    self.assertEqual(load.call_args.args[2], 'test')
                (dataset / 'hourly/fit/features.csv').write_text('changed\n')
                with self.assertRaisesRegex(ValueError, 'Dataset hash mismatch'):
                    evaluate._verify_freeze(dataset, freeze)
                (dataset / 'hourly/fit/features.csv').write_text('synthetic only\n')
                (run / 'metrics.json').write_text('{}')
                with self.assertRaisesRegex(ValueError, 'artifact differs'):
                    evaluate._verify_freeze(dataset, freeze)


if __name__ == '__main__':
    unittest.main()
