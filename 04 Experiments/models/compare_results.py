"""Pool frozen evaluation metrics and compare each model with the baseline."""

import argparse
import json
import math
from pathlib import Path

from evaluate import _within_runs


PERIODS = ('test', 'evaluation_1', 'evaluation_2', 'evaluation_3')
METRICS = ('mae', 'rmse', 'mape_percent')


def pooled(scores):
    """Combine disjoint periods, weighting by the number of observations."""
    for item in scores:
        nonzero_count = item['count'] - item['mape_zero_actuals_excluded']
        if (item['count'] < 1 or not 0 <= item['mape_zero_actuals_excluded'] <= item['count']
                or (item['mape_percent'] is None) != (nonzero_count == 0)):
            raise ValueError('Invalid counts or MAPE in period metrics')
    count = sum(item['count'] for item in scores)
    excluded = sum(item['mape_zero_actuals_excluded'] for item in scores)
    eligible = count - excluded
    mape_sum = sum(item['mape_percent'] * (item['count'] - item['mape_zero_actuals_excluded'])
                   for item in scores if item['mape_percent'] is not None)
    return {'count': count,
            'mae': sum(item['mae'] * item['count'] for item in scores) / count,
            'rmse': math.sqrt(sum(item['rmse'] ** 2 * item['count'] for item in scores) / count),
            'mape_percent': mape_sum / eligible if eligible else None,
            'mape_zero_actuals_excluded': excluded}


def improvement(scores, baseline):
    """Return relative reductions in error; zero baseline has no defined reduction."""
    return {name: (100 * (baseline[name] - scores[name]) / baseline[name]
                   if scores[name] is not None and baseline[name] not in (None, 0) else None)
            for name in METRICS}


def period_scores(period, task):
    if task == 'hourly':
        return {'hourly': (period['model'], period['previous_day'])}
    profiles = period['profiles']
    result = {}
    for prefix, count in (('hourly', 24 * profiles), ('total', profiles)):
        def convert(source):
            return {'count': count, **{name: source[f'{prefix}_{name}']
                                      for name in (*METRICS, 'mape_zero_actuals_excluded')}}
        result[prefix] = (convert(period['model']), convert(period['previous_day']))
    return result


def compare(experiments, task):
    reports = {}
    for name, path in experiments:
        if name in reports:
            raise ValueError(f'Duplicate model name: {name}')
        report = json.loads(Path(path).read_text(encoding='utf-8'))
        if (report['task'] != task or report['model'].lower() != name.lower()
                or any(period not in report['periods'] for period in PERIODS)):
            raise ValueError(f'{name} lacks the required {task} evaluation periods')
        reports[name] = report
    if not reports:
        raise ValueError('At least one model is required')
    if len({report['dataset_manifest_sha256'] for report in reports.values()}) != 1:
        raise ValueError('Models must use the same frozen dataset manifest')
    groups = ('hourly',) if task == 'hourly' else ('hourly', 'total')
    result = {group: {'periods': {}, 'pooled': {}} for group in groups}
    for period in PERIODS:
        if len({report['periods'][period]['target_times_sha256'] for report in reports.values()}) != 1:
            raise ValueError(f'Target keys differ for {period}')
        measures = {name: period_scores(report['periods'][period], task) for name, report in reports.items()}
        for group in groups:
            baselines = [item[group][1] for item in measures.values()]
            if any(baseline != baselines[0] for baseline in baselines[1:]):
                raise ValueError(f'Baseline metrics differ for {period}: {group}')
            baseline = baselines[0]
            if any(values[group][0]['count'] != baseline['count'] for values in measures.values()):
                raise ValueError(f'Model and baseline counts differ for {period}: {group}')
            result[group]['periods'][period] = {
                'baseline': baseline,
                'models': {name: {'scores': values[group][0],
                                  'relative_error_reduction_percent': improvement(values[group][0], baseline)}
                           for name, values in measures.items()},
            }
    for group in groups:
        sections = result[group]['periods']
        baseline = pooled([sections[period]['baseline'] for period in PERIODS])
        models = {}
        for name in reports:
            scores = pooled([sections[period]['models'][name]['scores'] for period in PERIODS])
            models[name] = {'scores': scores, 'relative_error_reduction_percent': improvement(scores, baseline)}
        result[group]['pooled'] = {'baseline': baseline, 'models': models}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hourly', nargs=2, action='append', metavar=('NAME', 'JSON'), required=True)
    parser.add_argument('--daily', nargs=2, action='append', metavar=('NAME', 'JSON'), required=True)
    parser.add_argument('--output', type=Path, required=True, help='New JSON file inside models/runs')
    args = parser.parse_args()
    output = _within_runs(args.output)
    if output.exists():
        parser.error('output file already exists')
    report = {'hourly_forecast': compare(args.hourly, 'hourly'),
              'daily_forecast': compare(args.daily, 'daily')}
    with output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print('Compared frozen hourly and daily models on matching targets')


if __name__ == '__main__':
    main()
