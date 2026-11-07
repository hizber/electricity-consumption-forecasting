"""Select explicitly chosen predictors and optionally add OpenAI proxy scores.

This module does not read a feature file unless the caller asks it to, make API
requests, write files, or fit any model. Input labels remain in the caller's
original rows; the returned dictionaries contain predictor columns only.
"""
from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
import math
from pathlib import Path
from collections.abc import Iterable, Mapping, Sequence

from decision_features import FEATURES

OPENAI_PREFIX = 'openai__'
_METADATA = frozenset({'origin_utc', 'target_end_utc', 'request_sha256'})
_FORBIDDEN_PREDICTORS = frozenset({
    'mtime', 'consumption', 'target', 'target_end_utc', 'origin_utc',
    'request_sha256', 'label', 'y', 'y_true', 'unnamed: 0',
})


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    """Load strings from CSV without silently accepting duplicate column names.

    The caller must convert or encode base predictors for the training model.
    """
    with Path(path).open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        names = reader.fieldnames
        if not names or any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError('CSV must have unique, nonempty column names')
        rows = list(reader)
    if any(None in row for row in rows):
        raise ValueError('CSV row contains more fields than the header')
    return rows


def _utc_time(value: object, name: str) -> datetime:
    if isinstance(value, datetime):
        time = value
    elif isinstance(value, str):
        try:
            time = datetime.fromisoformat(value.strip().replace('Z', '+00:00'))
        except ValueError as exc:
            raise ValueError(f'{name} must be a timezone-aware ISO timestamp') from exc
    else:
        raise ValueError(f'{name} must be a timezone-aware ISO timestamp')
    if time.tzinfo is None or time.utcoffset() is None:
        raise ValueError(f'{name} must include a timezone')
    return time.astimezone(timezone.utc)


def _score(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f'{name} must be a finite score in [0, 1]')
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be a finite score in [0, 1]') from exc
    if not math.isfinite(score) or not 0.0 <= score <= 1.0 + 1e-12:
        raise ValueError(f'{name} must be a finite score in [0, 1]')
    return min(score, 1.0)


def _feature_names(base_feature_columns: Sequence[str], join_key: str) -> list[str]:
    if isinstance(base_feature_columns, (str, bytes)):
        raise ValueError('Supply an explicit sequence of base feature names')
    columns = list(base_feature_columns)
    if not columns or any(not isinstance(name, str) or not name for name in columns):
        raise ValueError('Supply at least one explicit base feature name')
    if len(set(columns)) != len(columns):
        raise ValueError('Duplicate base feature name')
    for name in columns:
        if (name.lower() in _FORBIDDEN_PREDICTORS or name == join_key
                or name in FEATURES or name.startswith(OPENAI_PREFIX)):
            raise ValueError(f'{name!r} is metadata, a label, or an OpenAI score; not a base predictor')
    return columns


def _openai_index(rows: Iterable[Mapping[str, object]], join_key: str) -> dict[datetime, dict[str, float]]:
    if join_key != 'target_end_utc':
        raise ValueError('OpenAI generator scores must be joined on target_end_utc')
    index: dict[datetime, dict[str, float]] = {}
    for row in rows:
        missing = (_METADATA | set(FEATURES)) - row.keys()
        if missing:
            raise ValueError(f'OpenAI row is missing columns: {sorted(missing)}')
        extra = row.keys() - (_METADATA | set(FEATURES))
        if extra:
            raise ValueError(f'Unexpected OpenAI columns: {sorted(extra)}')
        target = _utc_time(row['target_end_utc'], 'target_end_utc')
        origin = _utc_time(row['origin_utc'], 'origin_utc')
        if target != origin + timedelta(hours=1):
            raise ValueError('OpenAI target timestamp must be one hour after origin')
        if target in index:
            raise ValueError('Duplicate OpenAI target timestamp')
        if not isinstance(row['request_sha256'], str) or not row['request_sha256']:
            raise ValueError('OpenAI row must have a request hash')
        index[target] = {OPENAI_PREFIX + name: _score(row[name], name) for name in FEATURES}
    return index


def build_training_features(
    base_rows: Iterable[Mapping[str, object]],
    base_feature_columns: Sequence[str],
    *,
    include_openai: bool = False,
    openai_rows: Iterable[Mapping[str, object]] | None = None,
    join_key: str = 'target_end_utc',
) -> list[dict[str, object]]:
    """Return predictors in base row order, with an explicit OpenAI on/off switch.

    Base feature names must be chosen by the caller. Metadata, target labels, raw
    generator score names and prefixed proxy scores cannot be selected as base
    features. When enabled, generator rows must match each base target timestamp
    exactly once and have the expected score schema. Any extra generator rows
    are ignored, so one table can cover multiple splits. No interpolation occurs.

    With include_openai=False, openai_rows is not read or iterated at all.
    Keep labels in a separate array in the same order as base_rows. This function
    cannot establish whether arbitrary caller-selected predictors are past-only;
    the caller must check their forecast-time availability.
    """
    if not isinstance(include_openai, bool):
        raise ValueError('include_openai must be True or False')
    columns = _feature_names(base_feature_columns, join_key)
    index = None
    if include_openai:
        if openai_rows is None:
            raise ValueError('OpenAI rows are required when include_openai=True')
        index = _openai_index(openai_rows, join_key)
    result: list[dict[str, object]] = []
    seen = set()
    for row in base_rows:
        missing = [name for name in columns if name not in row]
        if missing:
            raise ValueError(f'Base row is missing selected features: {missing}')
        predictors = {name: row[name] for name in columns}
        if include_openai:
            if any(name.startswith(OPENAI_PREFIX) for name in row):
                raise ValueError('Base row already contains OpenAI-prefixed columns')
            if join_key not in row:
                raise ValueError(f'Base row lacks join key {join_key}')
            target = _utc_time(row[join_key], join_key)
            if target in seen:
                raise ValueError('Duplicate base target timestamp')
            seen.add(target)
            if target not in index:
                raise ValueError(f'No OpenAI scores for target timestamp {target.isoformat()}')
            predictors.update(index[target])
        result.append(predictors)
    return result
