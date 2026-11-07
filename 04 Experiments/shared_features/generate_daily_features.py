"""Generate 24 target-hour rows per fixed daily forecast origin."""

import argparse
import csv
import statistics
from datetime import date, timedelta
from itertools import chain
from pathlib import Path

from generate_features import HOUR, LOCAL, SOURCE, WEATHER, WINDOWS, circular, finnish_public_holidays, read_source


def daily_rows(records, weather_latency_hours=None):
    """Yield features and labels for 24 hours issued together at 00:00 UTC.

    The weather latency is counted from each observation timestamp to the
    single origin. The function expects consecutive hourly input records.
    """
    if weather_latency_hours is not None and weather_latency_hours < 0:
        raise ValueError('weather latency must be nonnegative')
    holidays = {}
    for origin_index in range(167, len(records) - 24):
        origin = records[origin_index][0]
        if origin.hour != 0:
            continue
        history = [records[origin_index - lag][1] for lag in range(168)]
        labels = [records[origin_index + step][1] for step in range(1, 25)]
        if any(value is None for value in history + labels):
            continue
        base = {f'load_lag_{lag:03}': value for lag, value in enumerate(history, 1)}
        for width in WINDOWS:
            values = history[:width]
            base.update({
                f'load_mean_{width}h': statistics.fmean(values),
                f'load_std_{width}h': statistics.pstdev(values),
                f'load_min_{width}h': min(values),
                f'load_max_{width}h': max(values),
                f'load_change_{width}h': values[0] - values[-1],
            })
        if weather_latency_hours is not None:
            weather_index = origin_index - weather_latency_hours
            for key in WEATHER:
                latest = records[weather_index][2][key] if weather_index >= 0 else None
                base[f'{key}_latest'] = latest
                base[f'{key}_missing'] = int(latest is None)
            for width in WINDOWS:
                start = weather_index - width + 1
                values = [records[j][2]['temperature'] for j in range(max(0, start), weather_index + 1)] if weather_index >= 0 else []
                complete = len(values) == width and all(value is not None for value in values)
                base[f'temperature_mean_{width}h'] = statistics.fmean(values) if complete else None
                base[f'temperature_change_{width}h'] = values[-1] - values[0] if complete else None
            temp = base['temperature_latest']
            base['heating_degrees_15c'] = max(0.0, 15.0 - temp) if temp is not None else None
        for step, label in enumerate(labels, 1):
            target = origin + step * HOUR
            local = target.astimezone(LOCAL)
            if local.year not in holidays:
                holidays[local.year] = finnish_public_holidays(local.year)
            hour_sin, hour_cos = circular(local.hour, 24)
            weekday_sin, weekday_cos = circular(local.weekday(), 7)
            year_days = (date(local.year + 1, 1, 1) - date(local.year, 1, 1)).days
            year_sin, year_cos = circular(local.timetuple().tm_yday - 1, year_days)
            features = dict(base)
            features.update({
                'load_previous_day_target': history[24 - step],
                'load_previous_week_target': history[168 - step],
                'hour_sin': hour_sin, 'hour_cos': hour_cos,
                'weekday_sin': weekday_sin, 'weekday_cos': weekday_cos,
                'year_sin': year_sin, 'year_cos': year_cos,
                'is_weekend': int(local.weekday() >= 5),
                'is_finnish_public_holiday': int(local.date() in holidays[local.year]),
            })
            yield origin, target, step, features, label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=SOURCE)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--weather-latency-hours', type=int, default=None,
                        help='Assumed publication delay in whole hours. Default omits weather.')
    args = parser.parse_args()
    if args.weather_latency_hours is not None and args.weather_latency_hours < 0:
        parser.error('--weather-latency-hours must be nonnegative')
    records = read_source(args.input, args.weather_latency_hours is not None)
    rows = iter(daily_rows(records, args.weather_latency_hours))
    first = next(rows, None)
    if first is None:
        raise ValueError('No complete forecast profiles found')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / 'features.csv').open('x', newline='', encoding='utf-8') as xfile, \
         (args.output_dir / 'labels.csv').open('x', newline='', encoding='utf-8') as yfile:
        feature_writer = csv.DictWriter(xfile, fieldnames=['origin_utc', 'target_end_utc', 'horizon_hours', *first[3]])
        label_writer = csv.writer(yfile)
        feature_writer.writeheader()
        label_writer.writerow(('origin_utc', 'target_end_utc', 'horizon_hours', 'consumption'))
        count = 0
        for origin, target, step, features, label in chain((first,), rows):
            feature_writer.writerow({'origin_utc': origin.isoformat(), 'target_end_utc': target.isoformat(), 'horizon_hours': step, **features})
            label_writer.writerow((origin.isoformat(), target.isoformat(), step, label))
            count += 1
    print(f'Wrote {count} rows in {count // 24} complete profiles to {args.output_dir}')


if __name__ == '__main__':
    main()
