"""Offline tests for UTC-window threshold calculation."""
import datetime as dt
import tempfile
import unittest
from pathlib import Path

import choose_thresholds as ct

UTC = dt.timezone.utc
HOUR = dt.timedelta(hours=1)


class ThresholdTests(unittest.TestCase):
    def setUp(self):
        self.start = dt.datetime(2019, 1, 1, tzinfo=UTC)
        self.end = self.start + 5 * HOUR
        self.series = {self.start + i * HOUR: float(i) for i in range(5)}

    def test_linear_quantile(self):
        self.assertAlmostEqual(ct.linear_quantile([0, 1, 2, 3, 4]), 3.6)
        self.assertEqual(ct.linear_quantile([7]), 7)

    def test_window_excludes_outside_points_and_cross_boundary_pairs(self):
        series = {self.start - HOUR: 99999, **self.series, self.end: -99999}
        result = ct.calculate_thresholds(series, self.start, self.end)
        self.assertEqual(result['reading_count'], 5)
        self.assertEqual(result['adjacent_hourly_pair_count'], 4)
        self.assertAlmostEqual(result['high_consumption_threshold'], 3.6)
        self.assertAlmostEqual(result['absolute_change_threshold'], 1.0)
        self.assertEqual(result['high_event_count_in_fit'], 1)
        self.assertEqual(result['change_event_count_in_fit'], 4)

    def test_future_changes_do_not_change_statistics(self):
        before = {**self.series, self.end: 5.0, self.end + HOUR: 6.0}
        after = {**self.series, self.end: 9999.0, self.end + HOUR: -8888.0}
        self.assertEqual(ct.calculate_thresholds(before, self.start, self.end),
                         ct.calculate_thresholds(after, self.start, self.end))

    def test_missing_hour_excludes_difference_over_gap(self):
        series = dict(self.series)
        del series[self.start + 2 * HOUR]
        result = ct.calculate_thresholds(series, self.start, self.end)
        self.assertEqual(result['reading_count'], 4)
        self.assertEqual(result['adjacent_hourly_pair_count'], 2)
        self.assertEqual(result['absolute_change_threshold'], 1.0)

    def test_no_pairs_or_nonpositive_threshold_are_rejected(self):
        with self.assertRaises(ValueError):
            ct.calculate_thresholds({self.start: 1., self.start + 2*HOUR: 2.}, self.start, self.end)
        with self.assertRaises(ValueError):
            ct.calculate_thresholds({self.start: 1., self.start + HOUR: 1.}, self.start, self.end)

    def test_output_cannot_overwrite_source(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'source.csv'
            data = 'MTime,Consumption\n2019-01-01 00:00:00,1\n2019-01-01 01:00:00,2\n'
            path.write_text(data)
            result = ct.main(['--input', str(path), '--start', '2019-01-01T00:00:00Z',
                              '--end', '2019-01-01T02:00:00Z', '--output', str(path)])
            self.assertEqual(result, 1)
            self.assertEqual(path.read_text(), data)


if __name__ == '__main__':
    unittest.main()
