import unittest
from datetime import datetime, timedelta, timezone

from generate_features import feature_rows, finnish_public_holidays


class FeatureTests(unittest.TestCase):
    def records(self, start, count):
        return [(start + timedelta(hours=i), float(i), {
            'temperature': float(i), 'humidity': 50.0, 'wind_speed': 2.0,
        }) for i in range(count)]

    def test_lags_and_future_perturbation(self):
        rows = self.records(datetime(2020, 1, 1, tzinfo=timezone.utc), 171)
        origin, target, features, label = next(feature_rows(rows, 1))
        self.assertEqual(target - origin, timedelta(hours=1))
        self.assertEqual((features['load_lag_001'], features['load_lag_024'],
                          features['load_lag_168'], label), (167, 144, 0, 168))
        changed = list(rows)
        for i in range(168, len(changed)):
            stamp, _, weather = changed[i]
            changed[i] = (stamp, 999999.0, {**weather, 'temperature': 999999.0})
        self.assertEqual(features, next(feature_rows(changed, 1))[2])

    def test_weather_delay_and_missing(self):
        rows = self.records(datetime(2020, 1, 1, tzinfo=timezone.utc), 170)
        self.assertEqual(next(feature_rows(rows, 0))[2]['temperature_latest'], 167.0)
        self.assertEqual(next(feature_rows(rows, 2))[2]['temperature_latest'], 165.0)
        stamp, load, weather = rows[166]
        rows[166] = (stamp, load, {**weather, 'temperature': None})
        features = next(feature_rows(rows, 1))[2]
        self.assertIsNone(features['temperature_latest'])
        self.assertEqual(features['temperature_missing'], 1)
        self.assertIsNone(features['temperature_mean_3h'])
        self.assertEqual(features['humidity_latest'], 50.0)

    def test_rejects_weather_delays_that_could_select_future_observations(self):
        rows = self.records(datetime(2020, 1, 1, tzinfo=timezone.utc), 170)
        for latency in (-1, 0.5, True):
            with self.subTest(latency=latency), self.assertRaisesRegex(ValueError, 'nonnegative integer'):
                next(feature_rows(rows, latency))

    def test_finnish_holidays_and_dst(self):
        from datetime import date
        holidays = finnish_public_holidays(2020)
        self.assertIn(date(2020, 4, 10), holidays)
        self.assertIn(date(2020, 6, 20), holidays)
        self.assertIn(date(2020, 10, 31), holidays)
        start = datetime(2020, 10, 18, 0, tzinfo=timezone.utc)
        rows = self.records(start, 171)
        results = list(feature_rows(rows))
        self.assertEqual(results[1][1] - results[0][1], timedelta(hours=1))
        # The autumn DST change repeats local 03:00 for successive UTC hours.
        self.assertAlmostEqual(results[0][2]['hour_sin'], results[1][2]['hour_sin'])
        self.assertAlmostEqual(results[0][2]['hour_cos'], results[1][2]['hour_cos'])


if __name__ == '__main__':
    unittest.main()
