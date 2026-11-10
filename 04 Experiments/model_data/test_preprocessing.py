import csv
import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from preprocessing import Dataset, FeatureConfig, Preprocessor, load_split


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write_split(self, horizon, split, records, *, weather=False):
        directory = self.root / horizon / split
        directory.mkdir(parents=True)
        history = FeatureConfig(24, calendar=False).history_columns()
        features = ['origin_utc', 'target_end_utc', *history]
        if horizon == 'daily':
            features.append('horizon_hours')
        if weather:
            features.extend(('temperature_latest', 'temperature_missing', 'heating_degrees_15c'))
        rows = []
        for origin, target, step, level in records:
            row = {'origin_utc': origin.isoformat(), 'target_end_utc': target.isoformat(),
                   **{name: level + i for i, name in enumerate(history)}}
            if horizon == 'daily':
                row['horizon_hours'] = step
            if weather:
                row.update({'temperature_latest': 2.0, 'temperature_missing': 0,
                            'heating_degrees_15c': 13.0})
            rows.append(row)
        for name, headers, items in [('features.csv', features, rows),
                                     ('labels.csv', ['origin_utc', 'target_end_utc', 'consumption'],
                                      [{'origin_utc': r['origin_utc'], 'target_end_utc': r['target_end_utc'],
                                        'consumption': i + 3} for i, r in enumerate(rows)])]:
            with (directory / name).open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(items)
        manifest_path = self.root / 'manifest.json'
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
            'split_boundaries_utc': {}, 'output_sha256': {}, 'hourly': {'rows': {}}, 'daily': {'rows': {}}}
        if split == 'validation':
            start, end = '2020-02-01T00:00:00+00:00', '2020-03-01T00:00:00+00:00'
        else:
            start, end = '2020-01-01T00:00:00+00:00', '2020-02-01T00:00:00+00:00'
        manifest['split_boundaries_utc'][split] = {'start_inclusive': start, 'end_exclusive': end}
        manifest[horizon]['rows'][split] = len(rows)
        if weather:
            manifest['weather_availability'] = {'kind': 'observed_weather',
                                                'assumed_publication_delay_hours': 1,
                                                'historical_publication_and_revision_times_verified': False}
        for filename in ('features.csv', 'labels.csv'):
            path = directory / filename
            manifest['output_sha256'][path.relative_to(self.root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))

    def test_selection_respects_history_and_order(self):
        config = FeatureConfig(24, calendar=False, rolling=True, seasonal_lags=True)
        self.assertEqual(config.history_columns()[0], 'load_lag_024')
        self.assertEqual(config.history_columns()[-1], 'load_lag_001')
        self.assertNotIn('load_mean_168h', config.context_columns('hourly'))
        self.assertNotIn('load_previous_week_target', config.context_columns('daily'))
        self.assertIn('load_previous_day_target', config.context_columns('daily'))

    def test_rejects_out_of_order_and_empty_split_boundaries(self):
        start = datetime(2020, 1, 2, tzinfo=timezone.utc)
        records = [(start, start + timedelta(hours=1), 1, 1.0)]
        self.write_split('hourly', 'fit', records)
        manifest_path = self.root / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['split_boundaries_utc']['validation'] = {
            'start_inclusive': '2019-01-01T00:00:00+00:00',
            'end_exclusive': '2019-02-01T00:00:00+00:00',
        }
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'overlap or are out of order'):
            load_split(self.root, 'hourly', 'fit', FeatureConfig(24, calendar=False))
        manifest['split_boundaries_utc'].pop('validation')
        manifest['split_boundaries_utc']['fit']['end_exclusive'] = '2020-01-01T00:00:00+00:00'
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'Invalid split boundaries'):
            load_split(self.root, 'hourly', 'fit', FeatureConfig(24, calendar=False))

    def test_hourly_fit_only_transform_and_serialization(self):
        start = datetime(2020, 1, 2, tzinfo=timezone.utc)
        records = [(start + timedelta(hours=i), start + timedelta(hours=i+1), 1, float(i)) for i in range(3)]
        self.write_split('hourly', 'fit', records)
        validation_records = [(origin + timedelta(days=31), target + timedelta(days=31), step, level)
                              for origin, target, step, level in records]
        self.write_split('hourly', 'validation', validation_records)
        config = FeatureConfig(24, calendar=False)
        fit = load_split(self.root, 'hourly', 'fit', config)
        validation = load_split(self.root, 'hourly', 'validation', config)
        prep = Preprocessor.fit(fit)
        before = prep.means.copy()
        validation.history[:] = 999999
        prep.transform(validation)
        np.testing.assert_equal(prep.means, before)
        with self.assertRaises(ValueError):
            Preprocessor.fit(validation)
        path = self.root / 'preprocessor.json'
        prep.save(path)
        restored = Preprocessor.load(path)
        x, context, y = restored.transform(fit)
        self.assertEqual(x.shape, (3, 24, 1))
        self.assertEqual(context.shape, (3, 0))
        np.testing.assert_allclose(restored.inverse_labels(y), fit.labels)

    def test_daily_common_origin_and_tamper_detection(self):
        origin = datetime(2020, 1, 2, tzinfo=timezone.utc)
        records = [(origin, origin + timedelta(hours=i), i, 1.0) for i in range(1, 25)]
        self.write_split('daily', 'fit', records)
        data = load_split(self.root, 'daily', 'fit', FeatureConfig(24, calendar=False))
        self.assertEqual(data.history.shape, (1, 24, 1))
        self.assertEqual(data.context.shape, (1, 24, 1))
        self.assertEqual(data.labels.shape, (1, 24))
        prep = Preprocessor.fit(data)
        self.assertEqual(prep.transform(data)[1].shape, (1, 24, 1))
        label_file = self.root / 'daily/fit/labels.csv'
        label_file.write_text(label_file.read_text() + 'tampered\n')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            load_split(self.root, 'daily', 'fit', FeatureConfig(24, calendar=False))

    def test_daily_rejects_misaligned_labels_and_changing_history(self):
        origin = datetime(2020, 1, 2, tzinfo=timezone.utc)
        records = [(origin, origin + timedelta(hours=i), i, 1.0) for i in range(1, 25)]
        self.write_split('daily', 'fit', records)
        directory = self.root / 'daily/fit'
        feature_file, label_file = directory / 'features.csv', directory / 'labels.csv'
        with label_file.open(newline='') as f:
            reader = csv.DictReader(f)
            labels = list(reader)
            label_fields = reader.fieldnames
        labels[1]['target_end_utc'] = labels[0]['target_end_utc']
        with label_file.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=label_fields)
            writer.writeheader()
            writer.writerows(labels)
        with self.assertRaisesRegex(ValueError, 'keys do not match'):
            load_split(self.root, 'daily', 'fit', FeatureConfig(24, calendar=False), verify_hashes=False)

        # Restore the paired label and modify one history value inside the same profile.
        labels[1]['target_end_utc'] = (origin + timedelta(hours=2)).isoformat()
        with label_file.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=label_fields)
            writer.writeheader()
            writer.writerows(labels)
        with feature_file.open(newline='') as f:
            reader = csv.DictReader(f)
            features = list(reader)
            feature_fields = reader.fieldnames
        features[-1]['load_lag_001'] = '12345'
        with feature_file.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=feature_fields)
            writer.writeheader()
            writer.writerows(features)
        with self.assertRaisesRegex(ValueError, 'share one origin and history'):
            load_split(self.root, 'daily', 'fit', FeatureConfig(24, calendar=False), verify_hashes=False)

    def test_weather_selection_requires_explicit_manifest_and_limits_history(self):
        config = FeatureConfig(24, calendar=False, weather_temperature=True,
                               weather_temperature_rolling=True, weather_humidity=True)
        self.assertIn('temperature_latest', config.context_columns('hourly'))
        self.assertIn('temperature_mean_24h', config.context_columns('hourly'))
        self.assertNotIn('temperature_mean_168h', config.context_columns('hourly'))
        self.assertIn('humidity_missing', config.context_columns('hourly'))
        with self.assertRaisesRegex(ValueError, 'require weather_temperature'):
            FeatureConfig(weather_temperature_rolling=True)
        origin = datetime(2020, 1, 2, tzinfo=timezone.utc)
        self.write_split('hourly', 'fit', [(origin, origin + timedelta(hours=1), 1, 1.0)])
        with self.assertRaisesRegex(ValueError, 'explicit nonnegative availability-delay'):
            load_split(self.root, 'hourly', 'fit', config)

    def test_weather_imputation_uses_only_fit(self):
        config = FeatureConfig(24, calendar=False, weather_temperature=True)
        context = np.array([[1.0, 0.0, 14.0], [np.nan, 1.0, np.nan], [3.0, 0.0, 12.0]])
        data = Dataset('fit', 'hourly', config.history_columns(), config.context_columns('hourly'),
                       np.array(['a', 'b', 'c']), np.array(['d', 'e', 'f']),
                       np.ones((3, 24, 1)), context.copy(), np.array([1.0, 2.0, 3.0]))
        scaler = Preprocessor.fit(data)
        np.testing.assert_allclose(scaler.means[-3:], [2.0, 1 / 3, 13.0])
        validation = Dataset('validation', 'hourly', data.history_columns, data.context_columns,
                             data.origins, data.targets, data.history, context.copy(), data.labels)
        validation.context[0, 0] = 9999
        _, transformed, _ = scaler.transform(validation)
        self.assertEqual(transformed[1, 0], 0.0)
        np.testing.assert_allclose(scaler.means[-3:], [2.0, 1 / 3, 13.0])

    def test_daily_rejects_weather_that_changes_within_profile(self):
        origin = datetime(2020, 1, 2, tzinfo=timezone.utc)
        records = [(origin, origin + timedelta(hours=i), i, 1.0) for i in range(1, 25)]
        self.write_split('daily', 'fit', records, weather=True)
        config = FeatureConfig(24, calendar=False, weather_temperature=True)
        dataset = load_split(self.root, 'daily', 'fit', config)
        self.assertEqual(dataset.context.shape, (1, 24, 4))
        path = self.root / 'daily/fit/features.csv'
        with path.open(newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fields = reader.fieldnames
        rows[-1]['temperature_latest'] = '99'
        with path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(ValueError, 'weather must be fixed'):
            load_split(self.root, 'daily', 'fit', config, verify_hashes=False)


if __name__ == '__main__':
    unittest.main()
