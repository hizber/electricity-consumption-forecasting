#!/usr/bin/env python3
"""Generate causal OpenAI Decisions features for the next consumption hour.

Dry-run is the default. --execute sends data and may incur API charges. See README.md.
Only the Python standard library is needed (Python 3.9+ and timezone database).
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from urllib import error, request as urllib_request
from zoneinfo import ZoneInfo

UTC = timezone.utc
HOUR = timedelta(hours=1)
MODEL = 'gpt-6-luna'
OCCUPANTS = ('0', '1', '2', '3', '4', '5_plus')
ACTIVITY = ('rising', 'falling', 'fluctuating', 'stable')
PREDICATES = ('high_load', 'increase', 'large_change', 'atypical')
FEATURES = ([f'occupancy_{n}' for n in OCCUPANTS]
            + [f'occupancy_ge_{n}' for n in range(1, 6)]
            + [f'activity_{name}' for name in ACTIVITY] + list(PREDICATES))
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / '02 Data' / 'Electric-consumption' / 'dataset_mansion.csv'


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def stamp(value):
    return value.astimezone(UTC).isoformat().replace('+00:00', 'Z')


def parse_time(value, *, naive_is_utc=False):
    parsed = datetime.fromisoformat(value.strip().replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        if not naive_is_utc:
            raise ValueError('Specify a timezone, for example 2019-10-01T00:00:00Z')
        parsed = parsed.replace(tzinfo=UTC)
    parsed = parsed.astimezone(UTC)
    if parsed.minute or parsed.second or parsed.microsecond:
        raise ValueError(f'Timestamp must be on an exact hour: {value}')
    return parsed


def number(value):
    try:
        result = float(str(value).strip().replace(',', '.'))
    except (ValueError, TypeError) as exc:
        raise ValueError(f'Invalid numeric value: {value!r}') from exc
    if not math.isfinite(result):
        raise ValueError('Non-finite numeric value')
    return result


def load_series(path):
    """Load the supplied CSV: its naive MTime values are explicitly documented as UTC."""
    series = {}
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if not {'MTime', 'Consumption'} <= set(reader.fieldnames or []):
            raise ValueError('CSV must contain MTime and Consumption columns')
        for line, row in enumerate(reader, 2):
            try:
                timestamp = parse_time(row['MTime'], naive_is_utc=True)
                value = number(row['Consumption'])
            except (ValueError, KeyError) as exc:
                raise ValueError(f'Invalid data at CSV row {line}: {exc}') from exc
            if timestamp in series:
                raise ValueError(f'Duplicate timestamp: {stamp(timestamp)}')
            series[timestamp] = value
    if not series:
        raise ValueError('Input CSV has no rows')
    return series


def make_request(series, origin, model, high_threshold, change_threshold, history_hours=24):
    """Create an API request using observations no later than origin, never target values."""
    if origin.tzinfo is None or origin.utcoffset() is None:
        raise ValueError('Origin must be timezone-aware')
    origin = origin.astimezone(UTC)
    if origin.minute or origin.second or origin.microsecond:
        raise ValueError('Origin must be on an exact hour')
    if history_hours < 1:
        raise ValueError('history_hours must be positive')
    high_threshold, change_threshold = number(high_threshold), number(change_threshold)
    if change_threshold <= 0:
        raise ValueError('change threshold must be positive')
    times = [origin - HOUR * index for index in reversed(range(history_hours))]
    missing = [stamp(time) for time in times if time not in series]
    if missing:
        raise ValueError(f'Incomplete hourly history at {stamp(origin)}; missing {missing[0]}')
    local_target = (origin + HOUR).astimezone(ZoneInfo('Europe/Helsinki'))
    evidence = {
        'forecast_origin_utc': stamp(origin),
        'target_interval_end_utc': stamp(origin + HOUR),
        'target_local_time': local_target.isoformat(),
        'target_local_weekday': local_target.strftime('%A'),
        'consumption_unit': 'unspecified source unit',
        'history_oldest_first': [{'end_utc': stamp(time), 'consumption': number(series[time])} for time in times],
        'latest_consumption': number(series[origin]),
        'history_mean': sum(number(series[time]) for time in times) / len(times),
        'high_consumption_threshold': high_threshold,
        'large_absolute_change_threshold': change_threshold,
        'context': 'A house with electric floor heating and five residents; visitors and household changes are possible. No occupancy labels are available.',
    }
    choices = [{'value': value, 'description': ('Five or more people' if value == '5_plus' else f'{value} people')}
               for value in OCCUPANTS]
    return {
        'model': model,
        'input': canonical(evidence),
        'questions': [
            {'type': 'choice', 'name': 'occupants',
             'instructions': 'Estimate the number of people present at the beginning of the target hour. Use only the supplied history and context; occupancy is not directly observed.',
             'choices': choices},
            {'type': 'choice', 'name': 'activity',
             'instructions': 'Classify the observed history in the supplied order. Apply the following precedence: rising if latest minus oldest is at least large_absolute_change_threshold; otherwise falling if oldest minus latest is at least that threshold; otherwise fluctuating if maximum minus minimum is at least that threshold; otherwise stable.',
             'choices': [
                 {'value': 'rising', 'description': 'History rises by at least the change threshold from oldest to latest.'},
                 {'value': 'falling', 'description': 'History falls by at least the change threshold from oldest to latest.'},
                 {'value': 'fluctuating', 'description': 'Neither rising nor falling; range is at least the change threshold.'},
                 {'value': 'stable', 'description': 'None of the other conditions applies.'},
             ]},
            {'type': 'predicate', 'name': 'high_load',
             'instructions': 'Will consumption in the target hour be strictly greater than high_consumption_threshold? Use only the supplied evidence.'},
            {'type': 'predicate', 'name': 'increase',
             'instructions': 'Will consumption in the target hour be strictly greater than latest_consumption? Use only the supplied evidence.'},
            {'type': 'predicate', 'name': 'large_change',
             'instructions': 'Will the absolute difference between target-hour consumption and latest_consumption be at least large_absolute_change_threshold? Use only the supplied evidence.'},
            {'type': 'predicate', 'name': 'atypical',
             'instructions': 'Will the absolute difference between target-hour consumption and history_mean be at least large_absolute_change_threshold? Use only the supplied evidence.'},
        ],
    }


def request_key(payload):
    return hashlib.sha256(canonical(payload).encode('utf-8')).hexdigest()


def probability(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError('Probability must be numeric')
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError('Probability must be finite and between 0 and 1')
    return result


def parse_response(response):
    """Validate every answer; do not convert refusals or missing values into features."""
    if not isinstance(response, dict) or not isinstance(response.get('answers'), list):
        raise ValueError('Missing answers in Decisions response')
    named = {}
    for answer in response['answers']:
        if not isinstance(answer, dict) or not isinstance(answer.get('name'), str):
            raise ValueError('Answer is missing a name')
        name = answer['name']
        if name in named:
            raise ValueError(f'Duplicate answer: {name}')
        named[name] = answer
    expected = {'occupants', 'activity', *PREDICATES}
    if set(named) != expected:
        raise ValueError('Response has missing or unexpected answers')
    def choice_distribution(name, values):
        answer = named[name]
        if answer.get('type') != 'choice' or not isinstance(answer.get('probabilities'), list):
            raise ValueError(f'{name} was refused or is not a choice distribution')
        scores = {}
        for item in answer['probabilities']:
            if not isinstance(item, dict) or item.get('value') not in values:
                raise ValueError(f'Invalid {name} category')
            value = item['value']
            if value in scores:
                raise ValueError(f'Duplicate {name} category')
            scores[value] = probability(item.get('probability'))
        total = sum(scores.values())
        if set(scores) != set(values) or not math.isclose(total, 1.0, abs_tol=1e-4):
            raise ValueError(f'{name} probabilities must include all categories and sum to 1')
        # Account for rounding in a valid near-one distribution without changing raw response.
        return {value: score / total for value, score in scores.items()}
    probs = choice_distribution('occupants', OCCUPANTS)
    result = {f'occupancy_{value}': probs[value] for value in OCCUPANTS}
    for count in range(1, 6):
        result[f'occupancy_ge_{count}'] = sum(probs[value] for value in OCCUPANTS[count:])
    activities = choice_distribution('activity', ACTIVITY)
    result.update({f'activity_{value}': activities[value] for value in ACTIVITY})
    for name in PREDICATES:
        answer = named[name]
        if answer.get('type') != 'predicate':
            raise ValueError(f'{name} answer was refused or is not a predicate')
        result[name] = probability(answer.get('probability'))
    return result


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(data, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def get_or_create(payload, cache_dir, sender):
    """Cache an accepted full response by exact request; errors are never cached."""
    key = request_key(payload)
    path = Path(cache_dir) / f'{key}.json'
    if path.exists():
        saved = json.loads(path.read_text(encoding='utf-8'))
        if saved.get('request') != payload or saved.get('request_sha256') != key:
            raise ValueError(f'Cache request mismatch: {path}')
        parse_response(saved.get('response'))
        return saved['response'], True
    response = sender(payload)
    parse_response(response)
    atomic_json(path, {'request_sha256': key, 'request': payload, 'response': response,
                       'cached_at_utc': stamp(datetime.now(UTC))})
    return response, False


def send_request(payload):
    key = os.environ.get('OPENAI_API_KEY')
    if not key:
        raise RuntimeError('Set OPENAI_API_KEY in your environment for --execute')
    req = urllib_request.Request('https://api.openai.com/v1/decisions',
        data=canonical(payload).encode('utf-8'), method='POST',
        headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'})
    try:
        with urllib_request.urlopen(req, timeout=60) as response:
            return json.loads(response.read().decode('utf-8'))
    except error.HTTPError as exc:
        raise RuntimeError(f'OpenAI API returned HTTP {exc.code}; no automatic retry. Check access, quota or rate limits, then resume.') from exc
    except error.URLError as exc:
        raise RuntimeError('OpenAI API request failed; no automatic retry. Check the connection, then resume.') from exc


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_csv(path, rows):
    """Write accumulated successful rows atomically, so interruption is resumable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', newline='', dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        writer = csv.DictWriter(stream, fieldnames=['origin_utc', 'target_end_utc', 'request_sha256', *FEATURES])
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT, help='Input CSV with MTime and Consumption')
    parser.add_argument('--start', required=True, help='First target END timestamp, inclusive, with timezone')
    parser.add_argument('--end', required=True, help='Last target END timestamp, exclusive, with timezone')
    parser.add_argument('--high-threshold', required=True, type=float, help='Frozen high-load cutoff in source units')
    parser.add_argument('--change-threshold', required=True, type=float, help='Frozen absolute-change cutoff (>0), in source units')
    parser.add_argument('--model', default=MODEL, help='Documented Decisions model; server access required')
    parser.add_argument('--history-hours', type=int, default=24)
    parser.add_argument('--limit', type=int, default=5, help='Maximum eligible target hours to process (default: 5)')
    parser.add_argument('--execute', action='store_true', help='Actually send requests (may cost money)')
    parser.add_argument('--max-new-requests', type=int, default=5, help='Per-run cap on uncached API calls')
    parser.add_argument('--output', type=Path, default=Path('decision_features.csv'))
    parser.add_argument('--cache-dir', type=Path, default=Path('decision_cache'))
    args = parser.parse_args(argv)
    try:
        start, end = parse_time(args.start), parse_time(args.end)
        if start >= end or args.limit < 1 or args.max_new_requests < 0:
            raise ValueError('Require start < end, limit >= 1 and max-new-requests >= 0')
        if args.history_hours < 1 or not math.isfinite(args.high_threshold) or not math.isfinite(args.change_threshold) or args.change_threshold <= 0:
            raise ValueError('Require finite thresholds, positive change threshold and history-hours >= 1')
        series = load_series(args.input)
        # Selection uses target timestamp only; no target value enters the request.
        origins = [t for t in sorted(series) if start <= t + HOUR < end and t + HOUR in series]
        origins = [t for t in origins if all(t - HOUR * i in series for i in range(args.history_hours))][:args.limit]
        if not origins:
            raise ValueError('No eligible target hours in range with a complete history')
        first = make_request(series, origins[0], args.model, args.high_threshold, args.change_threshold, args.history_hours)
        if not args.execute:
            print('DRY RUN: no API calls or output files. Example request:')
            print(json.dumps(first, ensure_ascii=False, indent=2))
            print(f'Eligible targets selected: {len(origins)}; range bounds apply to target timestamps.')
            return 0
        output_path = args.output.resolve()
        manifest_path = Path(str(args.output) + '.manifest.json')
        protected = {args.input.resolve(), Path(__file__).resolve()}
        if output_path in protected or manifest_path.resolve() in protected or output_path == args.cache_dir.resolve() or args.cache_dir.resolve() in output_path.parents:
            raise ValueError('Output and manifest must not overwrite the source, script, or a cache file.')
        config = {'input_sha256': file_sha256(args.input), 'script_sha256': file_sha256(__file__),
                  'start_utc': stamp(start), 'end_utc_exclusive': stamp(end), 'limit': args.limit,
                  'history_hours': args.history_hours, 'model': args.model,
                  'high_threshold': args.high_threshold, 'change_threshold': args.change_threshold,
                  'target_definition': 'next hour ending at origin plus one hour'}
        if manifest_path.exists():
            old = json.loads(manifest_path.read_text(encoding='utf-8'))
            if old.get('config') != config:
                raise ValueError('Existing output manifest differs. Use a different --output for this configuration.')
        elif args.output.exists():
            raise ValueError('Output exists without a manifest. Choose a different --output.')
        else:
            atomic_json(manifest_path, {'config': config, 'cache_dir': str(args.cache_dir.resolve()),
                        'created_utc': stamp(datetime.now(UTC)), 'resolved_model': None,
                        'api_docs': 'https://developers.openai.com/api/docs/guides/decisions'})
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        # Previously exported rows must be backed by the exact cached responses.
        # Without this check, a deleted cache could silently regenerate different scores
        # and overwrite a reproducible result on resume.
        if args.output.exists():
            with args.output.open(newline='', encoding='utf-8') as stream:
                previous = list(csv.DictReader(stream))
            if len(previous) > len(origins):
                raise ValueError('Existing output contains more rows than the current selection')
            for row, origin in zip(previous, origins):
                payload = make_request(series, origin, args.model, args.high_threshold,
                                       args.change_threshold, args.history_hours)
                key = request_key(payload)
                path = args.cache_dir / f'{key}.json'
                if row.get('origin_utc') != stamp(origin) or row.get('target_end_utc') != stamp(origin + HOUR) or row.get('request_sha256') != key or not path.exists():
                    raise ValueError('Existing output does not match its request cache; restore the cache before resuming.')
                saved = json.loads(path.read_text(encoding='utf-8'))
                if saved.get('request_sha256') != key or saved.get('request') != payload:
                    raise ValueError('Existing cached request does not match output.')
                saved_response = saved.get('response')
                features = parse_response(saved_response)
                if not manifest.get('resolved_model') or saved_response.get('model') != manifest['resolved_model']:
                    raise ValueError('Existing cache model does not match the output manifest.')
                if any(number(row.get(name)) != value for name, value in features.items()):
                    raise ValueError('Existing output features do not match cached responses.')
        def guarded_sender(payload):
            response = send_request(payload)
            returned_model = response.get('model') if isinstance(response, dict) else None
            if not isinstance(returned_model, str) or not returned_model:
                raise ValueError('Decisions response has no model identifier.')
            if manifest.get('resolved_model') is not None and manifest['resolved_model'] != returned_model:
                raise ValueError('Decisions returned a different model identifier; use a new output and cache.')
            return response

        rows, new_count, hits = [], 0, 0
        for origin in origins:
            payload = make_request(series, origin, args.model, args.high_threshold, args.change_threshold, args.history_hours)
            key = request_key(payload)
            cached = (args.cache_dir / f'{key}.json').exists()
            if not cached and new_count >= args.max_new_requests:
                print(f'Paused at max-new-requests={args.max_new_requests}; rerun the same command to resume.', file=sys.stderr)
                break
            response, hit = get_or_create(payload, args.cache_dir, guarded_sender)
            returned_model = response.get('model')
            if not isinstance(returned_model, str) or not returned_model:
                raise ValueError('Decisions response has no model identifier.')
            if manifest.get('resolved_model') is None:
                manifest['resolved_model'] = returned_model
                atomic_json(manifest_path, manifest)
            elif manifest['resolved_model'] != returned_model:
                raise ValueError('Decisions returned a different model identifier; use a new output and cache.')
            hits += int(hit)
            new_count += int(not hit)
            rows.append({'origin_utc': stamp(origin), 'target_end_utc': stamp(origin + HOUR),
                         'request_sha256': key, **parse_response(response)})
            write_csv(args.output, rows)
            print(f'{len(rows)}/{len(origins)} target={stamp(origin + HOUR)} cache={"hit" if hit else "new"}', file=sys.stderr)
        print(f'Wrote {len(rows)} of {len(origins)} rows to {args.output}; {new_count} new calls, {hits} cached.')
        return 0 if len(rows) == len(origins) else 2
    except (ValueError, RuntimeError, OSError, json.JSONDecodeError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
