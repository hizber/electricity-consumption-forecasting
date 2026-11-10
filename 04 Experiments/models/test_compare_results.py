import unittest
import json
import tempfile
from copy import deepcopy
from pathlib import Path

from compare_results import PERIODS, compare, improvement, pooled


class ComparisonTests(unittest.TestCase):
    def test_pooled_metrics_weight_counts_and_excluded_zeros(self):
        scores = [
            {'count': 2, 'mae': 2., 'rmse': 2., 'mape_percent': 50.,
             'mape_zero_actuals_excluded': 1},
            {'count': 4, 'mae': 1., 'rmse': 1., 'mape_percent': 25.,
             'mape_zero_actuals_excluded': 0},
        ]
        combined = pooled(scores)
        self.assertEqual(combined['count'], 6)
        self.assertAlmostEqual(combined['mae'], 4 / 3)
        self.assertAlmostEqual(combined['rmse'], (2.) ** .5)
        self.assertAlmostEqual(combined['mape_percent'], 30.)
        self.assertEqual(combined['mape_zero_actuals_excluded'], 1)
        scores[0]['mape_zero_actuals_excluded'] = -1
        with self.assertRaisesRegex(ValueError, 'Invalid counts'):
            pooled(scores)

    def test_error_reduction_and_zero_baseline(self):
        baseline = {'mae': 2., 'rmse': 4., 'mape_percent': 0.}
        model = {'mae': 1., 'rmse': 5., 'mape_percent': 20.}
        self.assertEqual(improvement(model, baseline),
                         {'mae': 50., 'rmse': -25., 'mape_percent': None})

    def test_comparison_rejects_different_data_targets_and_baselines(self):
        metric = {'count': 1, 'mae': 2., 'rmse': 2., 'mape_percent': 50.,
                  'mape_zero_actuals_excluded': 0}
        first = {'task': 'hourly', 'model': 'lstm', 'dataset_manifest_sha256': 'same',
                 'periods': {period: {'target_times_sha256': period,
                                      'model': metric, 'previous_day': metric} for period in PERIODS}}
        second = deepcopy(first)
        second['model'] = 'gru'
        with tempfile.TemporaryDirectory() as directory:
            first_path, second_path = Path(directory) / 'first.json', Path(directory) / 'second.json'
            first_path.write_text(json.dumps(first))
            experiments = [('lstm', first_path), ('gru', second_path)]
            second['dataset_manifest_sha256'] = 'different'
            second_path.write_text(json.dumps(second))
            with self.assertRaisesRegex(ValueError, 'same frozen dataset'):
                compare(experiments, 'hourly')
            second['dataset_manifest_sha256'] = 'same'
            second['periods']['test']['target_times_sha256'] = 'different'
            second_path.write_text(json.dumps(second))
            with self.assertRaisesRegex(ValueError, 'Target keys differ'):
                compare(experiments, 'hourly')
            second['periods']['test']['target_times_sha256'] = 'test'
            second['periods']['test']['previous_day']['mae'] = 3.
            second_path.write_text(json.dumps(second))
            with self.assertRaisesRegex(ValueError, 'Baseline metrics differ'):
                compare(experiments, 'hourly')


if __name__ == '__main__':
    unittest.main()
