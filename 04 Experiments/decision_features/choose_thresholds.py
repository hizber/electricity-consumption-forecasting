#!/usr/bin/env python3
"""Choose thresholds from an explicit UTC fitting window; no API calls.

The high threshold is the 90th percentile of consumption. The change threshold
is the 90th percentile of absolute changes between adjacent recorded UTC hours.
Both timestamps in each change must be inside the fitting window.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import json
import math
from pathlib import Path
import sys

from decision_features import DEFAULT_INPUT, file_sha256, load_series, parse_time, stamp


def linear_quantile(values, fraction=0.9):
    """Empirical quantile using linear interpolation at index (n - 1) * q."""
    if not values or not 0 <= fraction <= 1:
        raise ValueError('Quantile needs nonempty values and a fraction between 0 and 1')
    ordered = sorted(values)
    if any(not math.isfinite(v) for v in ordered):
        raise ValueError('Values must be finite')
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + weight * (ordered[upper] - ordered[lower])


def calculate_thresholds(series, start, end):
    """Compute statistics from [start, end), without differences across the boundary."""
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError('Window boundaries must be timezone-aware')
    if start >= end:
        raise ValueError('Fitting start must be before fitting end')
    points = sorted((t, value) for t, value in series.items() if start <= t < end)
    if len(points) < 2:
        raise ValueError('Fitting window needs at least two readings')
    changes = [abs(current[1] - previous[1]) for previous, current in zip(points, points[1:])
               if current[0] - previous[0] == timedelta(hours=1)]
    if not changes:
        raise ValueError('Fitting window has no adjacent hourly pairs')
    readings = [value for _, value in points]
    high = linear_quantile(readings)
    change = linear_quantile(changes)
    if change <= 0:
        raise ValueError('The resulting change threshold is not positive; choose another method')
    return {
        'fit_start_utc_inclusive': stamp(start),
        'fit_end_utc_exclusive': stamp(end),
        'first_reading_utc': stamp(points[0][0]),
        'last_reading_utc': stamp(points[-1][0]),
        'quantile': 0.9,
        'interpolation': 'linear at (n - 1) * q',
        'unit': 'unspecified source consumption unit',
        'reading_count': len(readings),
        'adjacent_hourly_pair_count': len(changes),
        'high_consumption_threshold': high,
        'high_event_rule': 'target consumption > high_consumption_threshold',
        'high_event_count_in_fit': sum(v > high for v in readings),
        'absolute_change_threshold': change,
        'change_event_rule': 'absolute target-minus-previous consumption >= absolute_change_threshold',
        'change_event_count_in_fit': sum(v >= change for v in changes),
        'provenance_note': 'Provisional UTC boundaries; confirm assignment split timezone. Recompute within each training fold.',
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--start', required=True, help='Fitting start timestamp, inclusive, with timezone')
    parser.add_argument('--end', required=True, help='Fitting end timestamp, exclusive, with timezone')
    parser.add_argument('--output', type=Path, help='Create this JSON file; existing files are not overwritten')
    args = parser.parse_args(argv)
    try:
        start, end = parse_time(args.start), parse_time(args.end)
        if args.output and args.output.resolve() in {args.input.resolve(), Path(__file__).resolve(), (Path(__file__).parent / 'decision_features.py').resolve()}:
            raise ValueError('Output must not overwrite the dataset or program files')
        series = load_series(args.input)
        result = calculate_thresholds(series, start, end)
        result['input_sha256'] = file_sha256(args.input)
        result['threshold_script_sha256'] = file_sha256(__file__)
        result['reader_script_sha256'] = file_sha256(Path(__file__).parent / 'decision_features.py')
        encoded = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n'
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open('x', encoding='utf-8') as stream:
                stream.write(encoded)
        print(encoded, end='')
        return 0
    except (OSError, ValueError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
