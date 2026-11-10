"""Load and transform shared forecasting datasets without fitting on holdouts."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np


CALENDAR = ('hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos', 'year_sin',
            'year_cos', 'is_weekend', 'is_finnish_public_holiday')
WINDOWS = (3, 6, 24, 168)
STATISTICS = ('mean', 'std', 'min', 'max', 'change')
SPLITS = ('fit', 'validation', 'test', 'evaluation_1', 'evaluation_2', 'evaluation_3')
METADATA = ('origin_utc', 'target_end_utc')
WEATHER_TEMPERATURE = ('temperature_latest', 'temperature_missing', 'heating_degrees_15c')
WEATHER_HUMIDITY = ('humidity_latest', 'humidity_missing')
WEATHER_WIND = ('wind_speed_latest', 'wind_speed_missing')


@dataclass(frozen=True)
class FeatureConfig:
    """Input selection. The default uses 168 load readings and calendar fields."""

    history_hours: int = 168
    calendar: bool = True
    rolling: bool = False
    seasonal_lags: bool = False
    weather_temperature: bool = False
    weather_temperature_rolling: bool = False
    weather_humidity: bool = False
    weather_wind: bool = False

    def __post_init__(self):
        if self.history_hours not in (24, 72, 168):
            raise ValueError('history_hours must be 24, 72 or 168')
        for value in (self.calendar, self.rolling, self.seasonal_lags,
                      self.weather_temperature, self.weather_temperature_rolling,
                      self.weather_humidity, self.weather_wind):
            if not isinstance(value, bool):
                raise ValueError('feature group switches must be bool')
        if self.weather_temperature_rolling and not self.weather_temperature:
            raise ValueError('temperature rolling features require weather_temperature')

    def history_columns(self):
        return tuple(f'load_lag_{lag:03d}' for lag in range(self.history_hours, 0, -1))

    def context_columns(self, horizon):
        if horizon not in ('hourly', 'daily'):
            raise ValueError('horizon must be hourly or daily')
        columns = list(CALENDAR) if self.calendar else []
        if self.rolling:
            columns += [f'load_{stat}_{width}h' for width in WINDOWS
                        if width <= self.history_hours for stat in STATISTICS]
        if self.weather_temperature:
            columns.extend(WEATHER_TEMPERATURE)
        if self.weather_temperature_rolling:
            columns.extend(f'temperature_{stat}_{width}h' for width in WINDOWS
                           if width <= self.history_hours for stat in ('mean', 'change'))
        if self.weather_humidity:
            columns.extend(WEATHER_HUMIDITY)
        if self.weather_wind:
            columns.extend(WEATHER_WIND)
        if horizon == 'daily':
            columns.append('horizon_hours')
            if self.seasonal_lags:
                columns.append('load_previous_day_target')
                if self.history_hours >= 168:
                    columns.append('load_previous_week_target')
        return tuple(columns)


@dataclass
class Dataset:
    split: str
    horizon: str
    history_columns: tuple[str, ...]
    context_columns: tuple[str, ...]
    origins: np.ndarray
    targets: np.ndarray
    history: np.ndarray
    context: np.ndarray
    labels: np.ndarray


def _hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _time(value):
    try:
        stamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except (ValueError, AttributeError) as exc:
        raise ValueError(f'Invalid timestamp: {value!r}') from exc
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError('Timestamp must include a UTC offset')
    return stamp.astimezone(timezone.utc)


def _number(value, field, allow_missing=True):
    if value is None or not str(value).strip():
        if allow_missing:
            return np.nan
        raise ValueError(f'Missing {field}')
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(f'Invalid numeric value for {field}: {value!r}') from exc
    if not np.isfinite(number):
        raise ValueError(f'Nonfinite value for {field}')
    return number


def _reader(handle, required):
    reader = csv.DictReader(handle)
    names = reader.fieldnames
    if not names or len(names) != len(set(names)) or any(not name for name in names):
        raise ValueError('CSV headers must be present and unique')
    missing = set(required) - set(names)
    if missing:
        raise ValueError(f'Missing CSV columns: {sorted(missing)}')
    return reader


def _split_bounds(manifest, selected):
    boundaries = manifest['split_boundaries_utc']
    previous_end = None
    selected_bounds = None
    for split in SPLITS:
        if split not in boundaries:
            continue
        start = _time(boundaries[split]['start_inclusive'])
        end = _time(boundaries[split]['end_exclusive'])
        if start >= end:
            raise ValueError(f'Invalid split boundaries: {split}')
        if previous_end is not None and start < previous_end:
            raise ValueError(f'Split boundaries overlap or are out of order: {split}')
        if split == selected:
            selected_bounds = start, end
        previous_end = end
    if selected_bounds is None:
        raise ValueError(f'Missing split boundaries: {selected}')
    return selected_bounds


def load_split(root, horizon, split, config=FeatureConfig(), *, verify_hashes=True):
    """Load keyed rows, enforcing manifest hashes, split boundaries and horizons.

    Daily rows are checked and grouped as complete 24-hour profiles. Context
    columns may vary per future horizon, while history must remain unchanged.
    """
    if split not in SPLITS:
        raise ValueError(f'Unknown split: {split}')
    history_columns = config.history_columns()
    context_columns = config.context_columns(horizon)
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if config.weather_temperature or config.weather_humidity or config.weather_wind:
        weather = manifest.get('weather_availability')
        if (not isinstance(weather, dict) or weather.get('kind') != 'observed_weather'
                or isinstance(weather.get('assumed_publication_delay_hours'), bool)
                or not isinstance(weather.get('assumed_publication_delay_hours'), int)
                or weather['assumed_publication_delay_hours'] < 0):
            raise ValueError('Weather features require a manifest with an explicit nonnegative availability-delay assumption')
    paths = [root / horizon / split / filename for filename in ('features.csv', 'labels.csv')]
    if verify_hashes:
        for path in paths:
            key = path.relative_to(root).as_posix()
            if manifest['output_sha256'].get(key) != _hash(path):
                raise ValueError(f'Output hash mismatch: {key}')
    start, end = _split_bounds(manifest, split)
    expected = manifest[horizon]['rows'][split]
    origins, targets, histories, contexts, labels = [], [], [], [], []
    seen = set()
    with paths[0].open(newline='', encoding='utf-8') as xf, paths[1].open(newline='', encoding='utf-8') as yf:
        inputs = _reader(xf, (*METADATA, *history_columns, *context_columns))
        actuals = _reader(yf, (*METADATA, 'consumption'))
        for feature, label in zip(inputs, actuals, strict=True):
            if None in feature or None in label:
                raise ValueError('CSV row has more fields than the header')
            origin, target = (_time(feature[key]) for key in METADATA)
            if (origin, target) != tuple(_time(label[key]) for key in METADATA):
                raise ValueError('Feature and label keys do not match')
            if (origin, target) in seen:
                raise ValueError('Duplicate forecast key')
            seen.add((origin, target))
            if not start <= target < end:
                raise ValueError('Target outside its declared split')
            if horizon == 'hourly':
                if target != origin + timedelta(hours=1):
                    raise ValueError('Hourly target must be one hour after origin')
            else:
                step = int(_number(feature['horizon_hours'], 'horizon_hours', False))
                if (origin.hour or origin.minute or origin.second or not 1 <= step <= 24
                        or target != origin + timedelta(hours=step)
                        or origin + timedelta(hours=1) < start or origin + timedelta(hours=24) >= end):
                    raise ValueError('Invalid or cross-split daily forecast profile')
            origins.append(origin.isoformat())
            targets.append(target.isoformat())
            histories.append([_number(feature[c], c, False) for c in history_columns])
            contexts.append([_number(feature[c], c) for c in context_columns])
            labels.append(_number(label['consumption'], 'consumption', False))
    if len(labels) != expected:
        raise ValueError(f'Unexpected row count: {len(labels)} != {expected}')
    if not labels:
        raise ValueError('Dataset split is empty')
    history = np.asarray(histories, dtype=np.float64)
    context = np.asarray(contexts, dtype=np.float64).reshape(len(labels), len(context_columns))
    labels = np.asarray(labels, dtype=np.float64)
    origins, targets = np.asarray(origins), np.asarray(targets)
    if horizon == 'daily':
        if len(labels) % 24:
            raise ValueError('Daily rows must form complete 24-hour profiles')
        groups = len(labels) // 24
        origins, targets = origins.reshape(groups, 24), targets.reshape(groups, 24)
        history = history.reshape(groups, 24, len(history_columns))
        context = context.reshape(groups, 24, len(context_columns))
        labels = labels.reshape(groups, 24)
        if not np.all(origins == origins[:, :1]) or not np.all(history == history[:, :1, :]):
            raise ValueError('Daily profile must share one origin and history')
        weather_indices = [i for i, name in enumerate(context_columns)
                           if name.startswith(('temperature_', 'humidity_', 'wind_speed_'))
                           or name == 'heating_degrees_15c']
        if weather_indices and not np.array_equal(context[:, :, weather_indices],
                                                 np.broadcast_to(context[:, :1, weather_indices],
                                                                 context[:, :, weather_indices].shape),
                                                 equal_nan=True):
            raise ValueError('Daily observed weather must be fixed at the common origin')
        horizon_column = context_columns.index('horizon_hours')
        if not np.all(context[:, :, horizon_column] == np.arange(1, 25)):
            raise ValueError('Daily horizons must be ordered from 1 to 24')
        origins = origins[:, 0]
        history = history[:, 0, :]
    return Dataset(split, horizon, history_columns, context_columns, origins, targets,
                   history[:, :, np.newaxis], context, labels)


@dataclass
class Preprocessor:
    """Per-column mean imputation and standard scaling, learned on fit only."""

    horizon: str
    history_columns: tuple[str, ...]
    context_columns: tuple[str, ...]
    means: np.ndarray
    scales: np.ndarray
    label_mean: float
    label_scale: float

    @classmethod
    def fit(cls, dataset):
        if dataset.split != 'fit':
            raise ValueError('Preprocessing can only be fitted on the fit split')
        history = dataset.history[:, :, 0]
        context_rows = int(np.prod(dataset.context.shape[:-1]))
        context = dataset.context.reshape(context_rows, len(dataset.context_columns))
        if dataset.horizon == 'daily':
            # Each hourly target is a training example for known future covariates.
            history = np.repeat(history, 24, axis=0)
        values = np.concatenate([history, context], axis=1)
        if not np.isfinite(dataset.labels).all() or np.isinf(values).any():
            raise ValueError('Training data contains infinite or nonfinite labels')
        if np.isnan(values).all(axis=0).any():
            raise ValueError('A selected fitting feature is entirely missing')
        means = np.nanmean(values, axis=0)
        filled = np.where(np.isnan(values), means, values)
        scales = np.std(filled, axis=0)
        scales[scales == 0] = 1
        label_mean = float(np.mean(dataset.labels))
        label_scale = float(np.std(dataset.labels)) or 1.0
        return cls(dataset.horizon, dataset.history_columns, dataset.context_columns,
                   means, scales, label_mean, label_scale)

    def transform(self, dataset):
        if (dataset.horizon, dataset.history_columns, dataset.context_columns) != (
                self.horizon, self.history_columns, self.context_columns):
            raise ValueError('Dataset feature schema differs from fitted preprocessor')
        n_history = len(self.history_columns)
        history = dataset.history[:, :, 0]
        context = dataset.context
        if np.isinf(history).any() or np.isinf(context).any() or not np.isfinite(dataset.labels).all():
            raise ValueError('Dataset contains infinite or nonfinite labels')
        history = np.where(np.isnan(history), self.means[:n_history], history)
        context = np.where(np.isnan(context), self.means[n_history:], context)
        return ((history - self.means[:n_history])[:, :, np.newaxis] / self.scales[:n_history, np.newaxis],
                (context - self.means[n_history:]) / self.scales[n_history:],
                (dataset.labels - self.label_mean) / self.label_scale)

    def inverse_labels(self, values):
        return np.asarray(values) * self.label_scale + self.label_mean

    def save(self, path):
        payload = {'version': 1, 'horizon': self.horizon, 'history_columns': self.history_columns,
                   'context_columns': self.context_columns, 'means': self.means.tolist(),
                   'scales': self.scales.tolist(), 'label_mean': self.label_mean, 'label_scale': self.label_scale}
        Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        if data['version'] != 1:
            raise ValueError('Unsupported preprocessor version')
        result = cls(data['horizon'], tuple(data['history_columns']), tuple(data['context_columns']),
                     np.asarray(data['means'], dtype=np.float64), np.asarray(data['scales'], dtype=np.float64),
                     data['label_mean'], data['label_scale'])
        n = len(result.history_columns) + len(result.context_columns)
        if (result.means.shape != (n,) or result.scales.shape != (n,)
                or not np.isfinite(result.means).all() or not np.isfinite(result.scales).all()
                or (result.scales <= 0).any() or not np.isfinite(result.label_mean)
                or not np.isfinite(result.label_scale) or result.label_scale <= 0):
            raise ValueError('Invalid saved preprocessor')
        return result
