"""Score consumption-only baselines on a chronological validation window."""
from __future__ import annotations

import argparse
import csv
import json
from math import fsum, sqrt
from pathlib import Path

from decision_features.decision_features import DEFAULT_INPUT, HOUR, file_sha256, load_series, parse_time, stamp


HOURLY_LAGS = {'latest': 1, 'previous_day': 24, 'previous_week': 168}
DAILY_LAGS = {'previous_day': 24, 'previous_week': 168}


def forecast_profiles(series, start, end, horizon, lags):
    """Use common complete profiles, with target-end timestamps in [start, end)."""
    if any(lag < horizon for lag in lags.values()):
        raise ValueError('A baseline would use a reading after the forecast origin')
    profiles = []
    for origin in sorted(series):
        if horizon == 24 and origin.hour != 0:
            continue
        if origin + HOUR < start or origin + horizon * HOUR >= end:
            continue
        targets = [origin + lead * HOUR for lead in range(1, horizon + 1)]
        history = {name: [target - lag * HOUR for target in targets] for name, lag in lags.items()}
        if any(target not in series for target in targets):
            continue
        if any(time not in series for times in history.values() for time in times):
            continue
        profiles.append([
            {'origin_utc': stamp(origin), 'target_end_utc': stamp(target), 'lead_hours': lead + 1,
             'actual': series[target], **{name: series[times[lead]] for name, times in history.items()}}
            for lead, target in enumerate(targets)
        ])
    if not profiles:
        raise ValueError(f'No complete {horizon}-hour forecasts in the requested window')
    return profiles


def metrics(errors, actuals):
    percentage_errors = [abs(error / actual) for error, actual in zip(errors, actuals) if actual != 0]
    return {'mae': fsum(abs(error) for error in errors) / len(errors),
            'rmse': sqrt(fsum(error * error for error in errors) / len(errors)),
            'mape_percent': 100 * fsum(percentage_errors) / len(percentage_errors) if percentage_errors else None,
            'mape_zero_actuals_excluded': len(errors) - len(percentage_errors)}


def score(profiles, methods, *, include_totals=False):
    results = {}
    actuals = [row['actual'] for profile in profiles for row in profile]
    if include_totals:
        actual_totals = [fsum(row['actual'] for row in profile) for profile in profiles]
    for method in methods:
        errors = [[row[method] - row['actual'] for row in profile] for profile in profiles]
        result = {'hourly_values': metrics([error for profile in errors for error in profile], actuals)}
        if include_totals:
            result['24_hour_totals'] = metrics([fsum(profile) for profile in errors], actual_totals)
        results[method] = result
    return results


def write_predictions(path, profiles):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(profiles[0][0]))
        writer.writeheader()
        for profile in profiles:
            writer.writerows(profile)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--start', default='2019-10-01T00:00:00Z')
    parser.add_argument('--end', default='2020-10-01T00:00:00Z')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'baselines')
    args = parser.parse_args()
    start, end = parse_time(args.start), parse_time(args.end)
    if start >= end:
        parser.error('start must be before end')

    series = load_series(args.input)
    hourly = forecast_profiles(series, start, end, 1, HOURLY_LAGS)
    daily = forecast_profiles(series, start, end, 24, DAILY_LAGS)
    result = {
        'input_sha256': file_sha256(args.input),
        'validation_start_utc_inclusive': stamp(start),
        'validation_end_utc_exclusive': stamp(end),
        'consumption_unit': 'unspecified source unit',
        'next_hour': {
            'origins': len(hourly), 'first_target_end_utc': hourly[0][0]['target_end_utc'],
            'last_target_end_utc': hourly[-1][-1]['target_end_utc'],
            'methods': score(hourly, HOURLY_LAGS),
        },
        'next_24_hours': {
            'origins': len(daily), 'schedule': 'one origin per day at 00:00 UTC',
            'target_hours': 24 * len(daily), 'first_target_end_utc': daily[0][0]['target_end_utc'],
            'last_target_end_utc': daily[-1][-1]['target_end_utc'],
            'methods': score(daily, DAILY_LAGS, include_totals=True),
        },
    }
    output_dir = args.output_dir
    hourly_path = output_dir / 'next_hour_predictions.csv'
    daily_path = output_dir / 'next_24_hours_predictions.csv'
    summary_path = output_dir / 'scores.json'
    if args.input.resolve() in {path.resolve() for path in (hourly_path, daily_path, summary_path)}:
        parser.error('output must not overwrite the input dataset')
    output_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(hourly_path, hourly)
    write_predictions(daily_path, daily)
    summary = json.dumps(result, indent=2) + '\n'
    summary_path.write_text(summary, encoding='utf-8')
    print(summary, end='')


if __name__ == '__main__':
    main()
