"""Offline contract tests for the Decisions feature generator (no API calls)."""
import copy
import contextlib
import csv
import io
import datetime as dt
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import decision_features as df

UTC = dt.timezone.utc


def sample_response():
    # Illustrative synthetic values, not real occupancy probabilities.
    values = [0.10, 0.20, 0.25, 0.15, 0.20, 0.10]
    labels = ['0', '1', '2', '3', '4', '5_plus']
    return {'model': 'test-snapshot', 'answers': [
        {'type': 'choice', 'name': 'occupants', 'choice': '2', 'confidence': .25,
         'probabilities': [{'value': k, 'probability': v} for k, v in reversed(list(zip(labels, values)))]},
        {'type': 'predicate', 'name': 'high_load', 'probability': .2},
        {'type': 'predicate', 'name': 'increase', 'probability': .7},
        {'type': 'predicate', 'name': 'large_change', 'probability': .3},
        {'type': 'predicate', 'name': 'atypical', 'probability': .4},
        {'type': 'choice', 'name': 'activity', 'choice': 'stable', 'confidence': .4,
         'probabilities': [{'value': 'stable', 'probability': .4},
                           {'value': 'fluctuating', 'probability': .3},
                           {'value': 'falling', 'probability': .2},
                           {'value': 'rising', 'probability': .1}]},
    ]}


class InputContractTests(unittest.TestCase):
    def setUp(self):
        self.start = dt.datetime(2020, 3, 28, 0, tzinfo=UTC)
        self.series = {self.start + dt.timedelta(hours=i): 1.0 + i / 10 for i in range(48)}
        self.origin = self.start + dt.timedelta(hours=30)

    def request(self, series=None, origin=None):
        return df.make_request(series if series is not None else self.series,
                               origin if origin is not None else self.origin,
                               'test-snapshot', high_threshold=3.5,
                               change_threshold=1.0, history_hours=24)

    def test_future_values_do_not_change_request_or_cache_key(self):
        expected = self.request()
        changed = dict(self.series)
        for stamp in changed:
            if stamp > self.origin:
                changed[stamp] = 999999.0
        self.assertEqual(expected, self.request(changed))
        self.assertEqual(df.request_key(expected), df.request_key(self.request(changed)))

    def test_origin_value_is_included(self):
        changed = dict(self.series)
        changed[self.origin] = 98765.4321
        self.assertNotEqual(self.request(), self.request(changed))

    def test_missing_hour_in_context_is_rejected(self):
        changed = dict(self.series)
        del changed[self.origin - dt.timedelta(hours=10)]
        with self.assertRaises(ValueError):
            self.request(changed)

    def test_earlier_than_context_does_not_change_request(self):
        changed = dict(self.series)
        changed[self.origin - dt.timedelta(hours=24)] = 54321.0
        self.assertEqual(self.request(), self.request(changed))

    def test_source_decimal_comma_utc_and_duplicate_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.csv'
            path.write_text('MTime,Consumption\n2020-01-01 00:00:00,"1,25"\n2020-01-01 01:00:00,"2,5"\n', encoding='utf-8')
            result = df.load_series(path)
            self.assertEqual(result[dt.datetime(2020, 1, 1, tzinfo=UTC)], 1.25)
            self.assertEqual(result[dt.datetime(2020, 1, 1, 1, tzinfo=UTC)], 2.5)
            with path.open('a', encoding='utf-8') as handle:
                handle.write('2020-01-01 01:00:00,"3,0"\n')
            with self.assertRaises(ValueError):
                df.load_series(path)

    def test_request_key_is_canonical_and_changes_with_request(self):
        self.assertEqual(df.request_key({'a': 1, 'b': 2}), df.request_key({'b': 2, 'a': 1}))
        self.assertNotEqual(df.request_key(self.request()), df.request_key(self.request(origin=self.origin - dt.timedelta(hours=1))))
        self.assertNotEqual(df.request_key({'model': 'a'}), df.request_key({'model': 'b'}))


class ResponseContractTests(unittest.TestCase):
    def test_probabilities_are_mapped_by_value_and_cumulative(self):
        features = df.parse_response(sample_response())
        self.assertAlmostEqual(features['occupancy_0'], .10)
        self.assertAlmostEqual(features['occupancy_5_plus'], .10)
        for k, expected in enumerate([.90, .70, .45, .30, .10], start=1):
            self.assertAlmostEqual(features[f'occupancy_ge_{k}'], expected)
        self.assertAlmostEqual(features['high_load'], .2)
        self.assertAlmostEqual(features['increase'], .7)
        self.assertAlmostEqual(features['large_change'], .3)
        self.assertAlmostEqual(features['atypical'], .4)
        self.assertAlmostEqual(features['activity_stable'], .4)
        self.assertAlmostEqual(features['activity_rising'], .1)

    def test_invalid_response_is_rejected(self):
        cases = []
        missing = sample_response(); missing['answers'] = missing['answers'][:-1]; cases.append(missing)
        refused = sample_response(); refused['answers'][0] = {'type': 'refusal', 'name': 'occupants'}; cases.append(refused)
        duplicate = sample_response(); duplicate['answers'][0]['probabilities'][0]['value'] = '0'; cases.append(duplicate)
        nan = sample_response(); nan['answers'][1]['probability'] = math.nan; cases.append(nan)
        outside = sample_response(); outside['answers'][1]['probability'] = 1.1; cases.append(outside)
        not_normalized = sample_response(); not_normalized['answers'][0]['probabilities'][0]['probability'] = .8; cases.append(not_normalized)
        for response in cases:
            with self.subTest(response=response), self.assertRaises((ValueError, TypeError, KeyError)):
                df.parse_response(response)


class CacheContractTests(unittest.TestCase):
    def test_identical_request_is_cached_and_changed_request_is_not(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = []
            def sender(request):
                calls.append(copy.deepcopy(request))
                return sample_response()
            request = {'model': 'test-snapshot', 'input': 'first'}
            first, hit = df.get_or_create(request, Path(directory), sender)
            self.assertFalse(hit)
            second, hit = df.get_or_create(dict(reversed(list(request.items()))), Path(directory), sender)
            self.assertTrue(hit)
            self.assertEqual(first, second)
            self.assertEqual(len(calls), 1)
            _, hit = df.get_or_create({'model': 'test-snapshot', 'input': 'second'}, Path(directory), sender)
            self.assertFalse(hit)
            self.assertEqual(len(calls), 2)

    def test_failed_sender_is_not_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = []
            def sender(request):
                calls.append(request)
                if len(calls) == 1:
                    raise RuntimeError('synthetic network error')
                return sample_response()
            request = {'model': 'test-snapshot', 'input': 'same'}
            with self.assertRaises(RuntimeError):
                df.get_or_create(request, Path(directory), sender)
            _, hit = df.get_or_create(request, Path(directory), sender)
            self.assertFalse(hit)
            self.assertEqual(len(calls), 2)


class CliIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.input = root / 'source.csv'
        self.output = root / 'scores.csv'
        self.cache = root / 'cache'
        with self.input.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['MTime', 'Consumption'])
            start = dt.datetime(2020, 1, 1, tzinfo=UTC)
            for index in range(30):
                timestamp = start + dt.timedelta(hours=index)
                writer.writerow([timestamp.strftime('%Y-%m-%d %H:%M:%S'), f'{index},5'])
        self.args = ['--input', str(self.input), '--start', '2020-01-02T00:00:00Z',
                     '--end', '2020-01-02T03:00:00Z', '--high-threshold', '20',
                     '--change-threshold', '3', '--limit', '3', '--output', str(self.output),
                     '--cache-dir', str(self.cache)]
        self.calls = []

    def sender(self, payload):
        self.calls.append(copy.deepcopy(payload))
        return sample_response()

    def run_main(self, *extra):
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.object(df, 'send_request', side_effect=self.sender), \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = df.main(self.args + list(extra))
        return result, stdout.getvalue(), stderr.getvalue()

    def rows(self):
        with self.output.open(encoding='utf-8', newline='') as stream:
            return list(csv.DictReader(stream))

    def test_dry_run_has_no_network_or_output_side_effects(self):
        result, out, _ = self.run_main()
        self.assertEqual(result, 0)
        self.assertIn('DRY RUN', out)
        self.assertIn('2020-01-02T00:00:00Z', out)
        self.assertFalse(self.calls)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.cache.exists())

    def test_pause_resume_caches_each_request_and_rejects_manifest_mismatch(self):
        result, _, _ = self.run_main('--execute', '--max-new-requests', '1')
        self.assertEqual(result, 2)
        self.assertEqual(len(self.rows()), 1)
        self.assertEqual(len(self.calls), 1)
        result, _, _ = self.run_main('--execute', '--max-new-requests', '1')
        self.assertEqual(result, 2)
        self.assertEqual(len(self.rows()), 2)
        self.assertEqual(len(self.calls), 2)
        result, _, _ = self.run_main('--execute', '--max-new-requests', '1')
        self.assertEqual(result, 0)
        rows = self.rows()
        self.assertEqual(len(rows), 3)
        self.assertEqual(len(self.calls), 3)
        self.assertEqual([r['target_end_utc'] for r in rows], [
            '2020-01-02T00:00:00Z', '2020-01-02T01:00:00Z', '2020-01-02T02:00:00Z'])
        self.assertEqual(len({r['request_sha256'] for r in rows}), 3)
        original = self.output.read_bytes()
        result, _, err = self.run_main('--execute', '--model', 'different-model')
        self.assertEqual(result, 1)
        self.assertIn('manifest differs', err)
        self.assertEqual(self.output.read_bytes(), original)
        self.assertEqual(len(self.calls), 3)

    def test_existing_output_without_manifest_is_not_overwritten(self):
        self.output.write_text('precious data', encoding='utf-8')
        result, _, err = self.run_main('--execute')
        self.assertEqual(result, 1)
        self.assertIn('without a manifest', err)
        self.assertEqual(self.output.read_text(encoding='utf-8'), 'precious data')
        self.assertEqual(len(self.calls), 0)

    def test_output_cannot_overwrite_input(self):
        original = self.input.read_bytes()
        result, _, err = self.run_main('--execute', '--output', str(self.input))
        self.assertEqual(result, 1)
        self.assertIn('must not overwrite the source', err)
        self.assertEqual(self.input.read_bytes(), original)
        self.assertEqual(len(self.calls), 0)

    def test_inconsistent_returned_model_rejected_without_poisoning_cache(self):
        def model_drift(payload):
            self.calls.append(copy.deepcopy(payload))
            response = sample_response()
            if len(self.calls) == 2:
                response['model'] = 'different-snapshot'
            return response
        self.sender = model_drift
        result, _, err = self.run_main('--execute', '--max-new-requests', '3')
        self.assertEqual(result, 1)
        self.assertIn('different model identifier', err)
        self.assertEqual(len(self.rows()), 1)
        self.assertEqual(len(list(self.cache.glob('*.json'))), 1)

    def test_deleted_later_cache_entry_does_not_truncate_existing_output(self):
        result, _, _ = self.run_main('--execute', '--max-new-requests', '3')
        self.assertEqual(result, 0)
        previous = self.output.read_bytes()
        rows = self.rows()
        self.assertEqual(len(rows), 3)
        (self.cache / (rows[1]['request_sha256'] + '.json')).unlink()
        result, _, _ = self.run_main('--execute', '--max-new-requests', '0')
        self.assertNotEqual(result, 0)
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(self.output.read_bytes(), previous)

    def test_interrupted_run_keeps_completed_rows_and_reuses_cache(self):
        def fail_second(payload):
            self.calls.append(copy.deepcopy(payload))
            if len(self.calls) == 2:
                raise RuntimeError('synthetic interruption')
            return sample_response()
        self.sender = fail_second
        result, _, err = self.run_main('--execute', '--max-new-requests', '3')
        self.assertEqual(result, 1)
        self.assertIn('synthetic interruption', err)
        self.assertEqual(len(self.rows()), 1)
        self.assertEqual(len(list(self.cache.glob('*.json'))), 1)
        self.sender = lambda payload: (self.calls.append(copy.deepcopy(payload)) or sample_response())
        result, _, _ = self.run_main('--execute', '--max-new-requests', '3')
        self.assertEqual(result, 0)
        self.assertEqual(len(self.rows()), 3)
        self.assertEqual(len(self.calls), 4)


if __name__ == '__main__':
    unittest.main()
