"""Create chronological forecasting datasets without fitted preprocessing."""

import argparse
import csv
import hashlib
import json
import os
import platform
from datetime import datetime, timedelta, timezone
from pathlib import Path

from generate_features import SOURCE, feature_rows, read_source
from generate_daily_features import daily_rows as generate_daily_rows


def utc(value):
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


SPLITS = {
    'fit': (utc('2016-01-01'), utc('2019-10-01')),
    'validation': (utc('2019-10-01'), utc('2020-10-01')),
    'test': (utc('2020-10-01'), utc('2021-10-01')),
    'evaluation_1': (utc('2021-10-01'), utc('2022-01-01')),
    'evaluation_2': (utc('2022-01-01'), utc('2022-03-01')),
    'evaluation_3': (utc('2022-03-01'), utc('2022-06-01')),
}


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def split_for(first, last):
    for name, (start, end) in SPLITS.items():
        if start <= first and last < end:
            return name
    return None


def close_outputs(feature_file, label_file):
    """Fail if an output path was replaced while its file was open."""
    try:
        for stream in (feature_file, label_file):
            if stream.closed:
                continue
            stream.flush()
            opened = os.fstat(stream.fileno())
            current = os.stat(stream.name)
            if opened.st_nlink == 0 or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino):
                raise OSError(f'Output file was replaced during writing: {stream.name}')
    finally:
        feature_file.close()
        label_file.close()


def save_rows(output, name, rows):
    """Rows contain origin, target, final profile target, predictors and labels."""
    folder = output / name
    folder.mkdir()
    written_splits = set()
    active = None
    counts = {split: 0 for split in SPLITS}
    feature_names = None
    label_names = None
    skipped = 0
    try:
        for origin, target, last, features, labels in rows:
            split = split_for(origin + timedelta(hours=1), last)
            if split is None:
                skipped += 1
                continue
            if feature_names is None:
                feature_names = list(features)
                label_names = list(labels)
            if list(features) != feature_names or list(labels) != label_names:
                raise ValueError('Inconsistent dataset schema')
            if active is None or active[0] != split:
                if active is not None:
                    close_outputs(active[1], active[2])
                    active = None
                if split in written_splits:
                    raise ValueError(f'Rows return to an earlier split: {split}')
                directory = folder / split
                directory.mkdir()
                xf = (directory / 'features.csv').open('x', newline='', encoding='utf-8')
                yf = (directory / 'labels.csv').open('x', newline='', encoding='utf-8')
                xw = csv.DictWriter(xf, fieldnames=['origin_utc', 'target_end_utc', *feature_names])
                yw = csv.DictWriter(yf, fieldnames=['origin_utc', 'target_end_utc', *label_names])
                xw.writeheader()
                yw.writeheader()
                active = (split, xf, yf, xw, yw)
                written_splits.add(split)
            _, _, _, xw, yw = active
            metadata = {'origin_utc': origin.isoformat(), 'target_end_utc': target.isoformat()}
            xw.writerow({**metadata, **features})
            yw.writerow({**metadata, **labels})
            counts[split] += 1
    finally:
        if active is not None:
            close_outputs(active[1], active[2])
    if feature_names is None:
        raise ValueError(f'No eligible rows for {name}')
    for split in SPLITS:
        if split not in written_splits:
            directory = folder / split
            directory.mkdir()
            for filename, columns in (
                ('features.csv', feature_names), ('labels.csv', label_names)
            ):
                with (directory / filename).open('x', newline='', encoding='utf-8') as stream:
                    csv.writer(stream).writerow(['origin_utc', 'target_end_utc', *columns])
    return {'rows': counts, 'excluded_cross_boundary_or_outside_period': skipped,
            'predictor_columns': feature_names, 'label_columns': label_names}


def hourly_rows(records, weather_latency_hours=None):
    for origin, target, features, label in feature_rows(records, weather_latency_hours):
        yield origin, target, target, features, {'consumption': label}


def verify_saved_rows(output, task, counts):
    for split, expected in counts.items():
        folder = output / task / split
        with (folder / 'features.csv').open(newline='', encoding='utf-8') as xf, \
             (folder / 'labels.csv').open(newline='', encoding='utf-8') as yf:
            features = csv.DictReader(xf)
            labels = csv.DictReader(yf)
            if not features.fieldnames or not labels.fieldnames:
                raise ValueError(f'Missing CSV header in {task}/{split}')
            count = 0
            for x, y in zip(features, labels, strict=True):
                if (x['origin_utc'], x['target_end_utc']) != (y['origin_utc'], y['target_end_utc']):
                    raise ValueError(f'Feature and label mismatch in {task}/{split}')
                count += 1
            if count != expected:
                raise ValueError(f'Unexpected number of rows in {task}/{split}: {count} != {expected}')


def daily_rows(records, weather_latency_hours=None):
    for origin, target, step, features, label in generate_daily_rows(records, weather_latency_hours):
        yield origin, target, origin + timedelta(hours=24), {'horizon_hours': step, **features}, {'consumption': label}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=SOURCE)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--weather-latency-hours', type=int, default=None,
                        help='Opt in to observed weather using an assumed nonnegative publication delay in whole hours')
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('Output directory must not already exist')
    if args.weather_latency_hours is not None and args.weather_latency_hours < 0:
        parser.error('--weather-latency-hours must be nonnegative')
    records = read_source(args.input, args.weather_latency_hours is not None)
    args.output_dir.mkdir(parents=True)
    hourly = save_rows(args.output_dir, 'hourly', hourly_rows(records, args.weather_latency_hours))
    daily = save_rows(args.output_dir, 'daily', daily_rows(records, args.weather_latency_hours))
    verify_saved_rows(args.output_dir, 'hourly', hourly['rows'])
    verify_saved_rows(args.output_dir, 'daily', daily['rows'])
    scripts = [Path(__file__), Path(__file__).with_name('generate_features.py'),
               Path(__file__).with_name('generate_daily_features.py')]
    outputs = sorted(args.output_dir.rglob('*.csv'))
    manifest = {
        'input_sha256': file_hash(args.input),
        'script_sha256': {p.name: file_hash(p) for p in scripts},
        'output_sha256': {str(p.relative_to(args.output_dir)): file_hash(p) for p in outputs},
        'python_version': platform.python_version(),
        'split_boundaries_utc': {n: {'start_inclusive': s.isoformat(), 'end_exclusive': e.isoformat()} for n, (s, e) in SPLITS.items()},
        'hourly': hourly, 'daily': daily,
        'weather': ('excluded: historical availability and revision times unverified'
                    if args.weather_latency_hours is None else
                    'included conditionally: historical availability and revision times unverified'),
        'weather_availability': (None if args.weather_latency_hours is None else {
            'kind': 'observed_weather',
            'assumed_publication_delay_hours': args.weather_latency_hours,
            'historical_publication_and_revision_times_verified': False,
            'note': 'Conditional experiment only; delay is an assumption and observations may have later revisions.',
        }),
        'assumptions': ['meter reading available at origin', 'split boundaries use UTC provisionally',
                        'daily forecast at 00:00 UTC, 24 elapsed hours'],
    }
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'hourly': hourly['rows'], 'daily': daily['rows']}, indent=2))


if __name__ == '__main__':
    main()
