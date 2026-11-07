"""Offline tests for optional, timestamp-aligned OpenAI proxy features."""
import csv
from pathlib import Path
import tempfile
import unittest
import feature_selection as fs


def base_row(target='2019-10-01T00:00:00Z', lag=2.0):
    return {'target_end_utc': target, 'lag_1': lag, 'hour': 3, 'Consumption': 123.0,
            'origin_utc': '2019-09-30T23:00:00Z'}


def openai_row(target='2019-10-01T00:00:00Z', origin='2019-09-30T23:00:00Z'):
    row = {'target_end_utc': target, 'origin_utc': origin, 'request_sha256': 'test-request'}
    row.update({name: '0.5' for name in fs.FEATURES})
    return row


class FeatureSwitchTests(unittest.TestCase):
    def test_disabled_ignores_openai_even_if_unusable(self):
        class Poison:
            def __iter__(self):
                raise AssertionError('OpenAI data must not be read when disabled')
        rows = [base_row()]
        result = fs.build_training_features(rows, ['lag_1', 'hour'], openai_rows=Poison())
        self.assertEqual(result, [{'lag_1': 2.0, 'hour': 3}])
        self.assertEqual(rows[0]['Consumption'], 123.0)

    def test_enabled_joins_exact_utc_in_base_order_without_metadata(self):
        base = [base_row(), base_row('2019-10-01T01:00:00Z', lag=4.0)]
        scores = [openai_row('2019-10-01T01:00:00Z', '2019-10-01T00:00:00Z'),
                  openai_row('2019-10-01T03:00:00+03:00', '2019-10-01T02:00:00+03:00')]
        scores[0]['increase'] = '0.8'; scores[1]['increase'] = '0.2'
        features = fs.build_training_features(base, ['lag_1', 'hour'], include_openai=True, openai_rows=scores)
        self.assertEqual([row['lag_1'] for row in features], [2.0, 4.0])
        self.assertEqual([row['openai__increase'] for row in features], [0.2, 0.8])
        self.assertEqual(set(features[0]), {'lag_1', 'hour'} | {f'openai__{name}' for name in fs.FEATURES})
        for name in ['Consumption', 'target_end_utc', 'origin_utc', 'request_sha256']:
            self.assertNotIn(name, features[0])
        self.assertNotIn('openai__increase', base[0])

    def test_switch_can_be_disabled_after_base_contains_prefixed_columns(self):
        base = base_row(); base['openai__increase'] = 0.7
        self.assertEqual(fs.build_training_features([base], ['lag_1']), [{'lag_1': 2.0}])
        with self.assertRaisesRegex(ValueError, 'already contains'):
            fs.build_training_features([base], ['lag_1'], include_openai=True, openai_rows=[openai_row()])

    def test_labels_metadata_and_generated_scores_cannot_be_base_predictors(self):
        for forbidden in ['Consumption', 'target_end_utc', 'origin_utc', 'request_sha256',
                          'target', 'label', 'y', 'occupancy_0', 'openai__increase']:
            with self.subTest(forbidden=forbidden), self.assertRaises(ValueError):
                fs.build_training_features([base_row()], [forbidden])
        with self.assertRaises(ValueError):
            fs.build_training_features([base_row()], ['lag_1', 'lag_1'])

    def test_enabled_requires_match_and_checks_duplicate_keys(self):
        with self.assertRaisesRegex(ValueError, 'required'):
            fs.build_training_features([base_row()], ['lag_1'], include_openai=True)
        with self.assertRaisesRegex(ValueError, 'No OpenAI scores'):
            fs.build_training_features([base_row()], ['lag_1'], include_openai=True, openai_rows=[])
        with self.assertRaisesRegex(ValueError, 'Duplicate OpenAI'):
            fs.build_training_features([base_row()], ['lag_1'], include_openai=True, openai_rows=[openai_row(), openai_row()])
        with self.assertRaisesRegex(ValueError, 'Duplicate base'):
            fs.build_training_features([base_row(), base_row()], ['lag_1'], include_openai=True, openai_rows=[openai_row()])

    def test_wrong_or_naive_timestamps_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'one hour after'):
            fs.build_training_features([base_row()], ['lag_1'], include_openai=True,
                                       openai_rows=[openai_row(origin='2019-09-30T22:00:00Z')])
        with self.assertRaisesRegex(ValueError, 'timezone'):
            fs.build_training_features([base_row()], ['lag_1'], include_openai=True,
                                       openai_rows=[openai_row(origin='2019-09-30T23:00:00')])
        with self.assertRaisesRegex(ValueError, 'timezone'):
            fs.build_training_features([base_row('2019-10-01T00:00:00')], ['lag_1'],
                                       include_openai=True, openai_rows=[openai_row()])

    def test_missing_extra_or_nonfinite_generated_scores_are_rejected(self):
        cases = []
        row = openai_row(); row.pop('increase'); cases.append(row)
        row = openai_row(); row['Consumption'] = '4.0'; cases.append(row)
        row = openai_row(); row['increase'] = 'nan'; cases.append(row)
        row = openai_row(); row['increase'] = '1.1'; cases.append(row)
        for row in cases:
            with self.subTest(row=row), self.assertRaises(ValueError):
                fs.build_training_features([base_row()], ['lag_1'], include_openai=True, openai_rows=[row])

    def test_csv_reader_rejects_duplicate_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test.csv'
            path.write_text('lag_1,lag_1\n1,2\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'unique'):
                fs.read_csv_rows(path)
            path.write_text('lag_1,hour\n1,2,3\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'more fields'):
                fs.read_csv_rows(path)

    def test_csv_round_trip_reads_without_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test.csv'
            with path.open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=['lag_1', 'hour'])
                writer.writeheader(); writer.writerow({'lag_1': 1, 'hour': 2})
            before = path.read_bytes()
            self.assertEqual(fs.read_csv_rows(path), [{'lag_1': '1', 'hour': '2'}])
            self.assertEqual(path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
