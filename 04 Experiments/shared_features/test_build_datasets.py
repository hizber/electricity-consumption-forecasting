import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from build_datasets import SPLITS, close_outputs, daily_rows, hourly_rows, save_rows, verify_saved_rows


class BuildDatasetTests(unittest.TestCase):
    def test_chronological_split_writes_and_empty_splits(self):
        first = SPLITS['fit'][0] + timedelta(hours=1)
        last = SPLITS['evaluation_1'][0] + timedelta(hours=1)
        rows = ((t - timedelta(hours=1), t, t, {'lag': 1}, {'consumption': 2})
                for t in (first, last))
        with tempfile.TemporaryDirectory() as directory:
            result = save_rows(Path(directory), 'hourly', rows)
            verify_saved_rows(Path(directory), 'hourly', result['rows'])
            self.assertEqual(result['rows']['fit'], 1)
            self.assertEqual(result['rows']['evaluation_1'], 1)
            self.assertEqual(result['rows']['test'], 0)

    def test_replaced_output_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = (root / 'features.csv').open('w')
            second = (root / 'labels.csv').open('w')
            first.write('some data')
            (root / 'features.csv').unlink()
            (root / 'features.csv').write_text('replacement')
            with self.assertRaisesRegex(OSError, 'replaced'):
                close_outputs(first, second)
            self.assertTrue(first.closed and second.closed)

    def test_weather_delay_and_daily_common_origin(self):
        start = datetime(2019, 12, 25, 1, tzinfo=timezone.utc)
        records = [(start + timedelta(hours=i), float(i), {
            'temperature': float(i), 'humidity': 60.0, 'wind_speed': 3.0,
        }) for i in range(192)]
        hourly = next(hourly_rows(records, 1))
        # Hourly target index 168 has origin index 167 and uses index 166.
        self.assertEqual(hourly[3]['temperature_latest'], 166.0)
        profile = list(daily_rows(records, 1))
        self.assertEqual(len(profile), 24)
        self.assertTrue(all(item[3]['temperature_latest'] == 166.0 for item in profile))
        changed = list(records)
        for i in range(168, len(changed)):
            time, consumption, weather = changed[i]
            changed[i] = (time, consumption + 1000, {**weather, 'temperature': 10000.0})
        altered = list(daily_rows(changed, 1))
        self.assertEqual([item[3] for item in profile], [item[3] for item in altered])
        self.assertNotIn('temperature_latest', next(hourly_rows(records))[3])


if __name__ == '__main__':
    unittest.main()
