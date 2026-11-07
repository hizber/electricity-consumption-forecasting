import unittest
from datetime import datetime, timedelta, timezone

from generate_daily_features import daily_rows


class DailyFeaturesTest(unittest.TestCase):
    def make_records(self):
        start = datetime(2019, 12, 25, 1, tzinfo=timezone.utc)
        return [(start + timedelta(hours=i), float(i), {
            'temperature': float(i), 'humidity': 50.0, 'wind_speed': 2.0,
        }) for i in range(192)]

    def test_fixed_origin_and_no_future_leakage(self):
        records = self.make_records()
        baseline = list(daily_rows(records, 1))
        self.assertEqual(len(baseline), 24)
        self.assertEqual([r[2] for r in baseline], list(range(1, 25)))
        origin = baseline[0][0]
        for _, target, horizon, features, _ in baseline:
            self.assertEqual(target, origin + timedelta(hours=horizon))
            self.assertEqual(features['load_lag_001'], 167.0)
            self.assertEqual(features['load_lag_168'], 0.0)
            self.assertEqual(features['temperature_latest'], 166.0)
            self.assertEqual(features['load_previous_day_target'], 143.0 + horizon)
            self.assertEqual(features['load_previous_week_target'], horizon - 1.0)
        changed = list(records)
        for i in range(168, 192):
            stamp, _, _ = changed[i]
            changed[i] = (stamp, 9999.0, {
                'temperature': 9999.0, 'humidity': 9999.0, 'wind_speed': 9999.0,
            })
        modified = list(daily_rows(changed, 1))
        self.assertEqual([r[3] for r in baseline], [r[3] for r in modified])
        self.assertNotEqual(baseline[0][4], modified[0][4])

    def test_weather_is_optional_and_missing_is_preserved(self):
        records = self.make_records()
        stamp, load, weather = records[166]
        records[166] = (stamp, load, {**weather, 'temperature': None})
        without = next(daily_rows(records))[3]
        with_weather = next(daily_rows(records, 1))[3]
        self.assertNotIn('temperature_latest', without)
        self.assertIsNone(with_weather['temperature_latest'])
        self.assertEqual(with_weather['temperature_missing'], 1)
        self.assertIsNone(with_weather['temperature_mean_3h'])

    def test_incomplete_profile_is_omitted(self):
        records = self.make_records()
        stamp, _, weather = records[190]
        records[190] = (stamp, None, weather)
        self.assertEqual(list(daily_rows(records)), [])


if __name__ == '__main__':
    unittest.main()
