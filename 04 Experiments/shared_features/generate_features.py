"""Build shared next-hour features without fitting on validation or test data."""

import argparse
import csv
import math
import statistics
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


HOUR = timedelta(hours=1)
LOCAL = ZoneInfo("Europe/Helsinki")
SOURCE = Path(__file__).resolve().parents[2] / "02 Data/Electric-consumption/dataset_mansion.csv"
WINDOWS = (3, 6, 24, 168)
WEATHER = {
    "temperature": "AirTemperature(degC)",
    "humidity": "RelativeHumidity(%)",
    "wind_speed": "WindSpeed(m/s)",
}


def easter(year):
    """Gregorian Easter Sunday, used to derive Finnish movable holidays."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month = (h + ell - 7 * m + 114) // 31
    day = (h + ell - 7 * m + 114) % 31 + 1
    return date(year, month, day)


def finnish_public_holidays(year):
    sunday = easter(year)
    holidays = {date(year, month, day) for month, day in (
        (1, 1), (1, 6), (5, 1), (12, 6), (12, 25), (12, 26)
    )}
    holidays.update(sunday + timedelta(days=offset) for offset in (-2, 0, 1, 39, 49))
    for start in (date(year, 6, 20), date(year, 10, 31)):
        holidays.add(start + timedelta(days=(5 - start.weekday()) % 7))
    return holidays


def number(value, field):
    if value is None or not value.strip():
        return None
    try:
        parsed = float(value.strip().replace(",", "."))
    except ValueError as exc:
        raise ValueError(f"Invalid {field} value: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"Non-finite {field} value: {value!r}")
    return parsed


def read_source(path, include_weather):
    records = []
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"MTime", "Consumption"} | (set(WEATHER.values()) if include_weather else set())
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Missing required columns: {sorted(required - set(reader.fieldnames or []))}")
        for row in reader:
            stamp = datetime.fromisoformat(row["MTime"].replace("Z", "+00:00"))
            stamp = stamp.replace(tzinfo=timezone.utc) if stamp.tzinfo is None else stamp.astimezone(timezone.utc)
            if stamp.minute or stamp.second or stamp.microsecond:
                raise ValueError(f"Timestamp is not on an hour: {stamp}")
            if records and stamp - records[-1][0] != HOUR:
                raise ValueError(f"Timestamps are not strictly consecutive hourly values at {stamp}")
            consumption = number(row["Consumption"], "Consumption")
            weather = {key: number(row[column], column) for key, column in WEATHER.items()} if include_weather else {}
            records.append((stamp, consumption, weather))
    if len(records) < 169:
        raise ValueError("At least 169 consecutive hourly source rows are required")
    return records


def circular(value, period):
    angle = 2 * math.pi * value / period
    return math.sin(angle), math.cos(angle)


def feature_rows(records, weather_latency_hours=None):
    if weather_latency_hours is not None and (
        isinstance(weather_latency_hours, bool)
        or not isinstance(weather_latency_hours, int)
        or weather_latency_hours < 0
    ):
        raise ValueError("weather latency must be a nonnegative integer number of hours")
    holidays = {}
    for index in range(168, len(records)):
        target, label, _ = records[index]
        history = [records[index - lag][1] for lag in range(1, 169)]
        if label is None or any(value is None for value in history):
            continue
        local = target.astimezone(LOCAL)
        local_date = local.date()
        if local.year not in holidays:
            holidays[local.year] = finnish_public_holidays(local.year)
        hour_sin, hour_cos = circular(local.hour, 24)
        weekday_sin, weekday_cos = circular(local.weekday(), 7)
        year_days = (date(local.year + 1, 1, 1) - date(local.year, 1, 1)).days
        year_sin, year_cos = circular(local_date.timetuple().tm_yday - 1, year_days)
        features = {f"load_lag_{lag:03}": value for lag, value in enumerate(history, 1)}
        features.update({
            "hour_sin": hour_sin, "hour_cos": hour_cos,
            "weekday_sin": weekday_sin, "weekday_cos": weekday_cos,
            "year_sin": year_sin, "year_cos": year_cos,
            "is_weekend": int(local.weekday() >= 5),
            "is_finnish_public_holiday": int(local_date in holidays[local.year]),
        })
        for width in WINDOWS:
            values = history[:width]
            features.update({
                f"load_mean_{width}h": statistics.fmean(values),
                f"load_std_{width}h": statistics.pstdev(values),
                f"load_min_{width}h": min(values),
                f"load_max_{width}h": max(values),
                f"load_change_{width}h": values[0] - values[-1],
            })
        if weather_latency_hours is not None:
            # At target index i, origin is i-1. Latency 0 assumes the origin's
            # weather observation is published by the origin.
            weather_index = index - 1 - weather_latency_hours
            for key in WEATHER:
                latest = records[weather_index][2][key] if weather_index >= 0 else None
                features[f"{key}_latest"] = latest
                features[f"{key}_missing"] = int(latest is None)
            for width in WINDOWS:
                start = weather_index - width + 1
                values = [records[j][2]["temperature"] for j in range(max(0, start), weather_index + 1)] if weather_index >= 0 else []
                complete = len(values) == width and all(value is not None for value in values)
                features[f"temperature_mean_{width}h"] = statistics.fmean(values) if complete else None
                features[f"temperature_change_{width}h"] = values[-1] - values[0] if complete else None
            temp = features["temperature_latest"]
            features["heating_degrees_15c"] = max(0.0, 15.0 - temp) if temp is not None else None
        yield target - HOUR, target, features, label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for features.csv and labels.csv")
    parser.add_argument("--weather-latency-hours", type=int, default=None,
                        help="Include observations only after this many whole hours after their timestamp. Default: omit weather.")
    args = parser.parse_args()
    if args.weather_latency_hours is not None and args.weather_latency_hours < 0:
        parser.error("--weather-latency-hours must be nonnegative")
    records = read_source(args.input, args.weather_latency_hours is not None)
    rows = iter(feature_rows(records, args.weather_latency_hours))
    first = next(rows, None)
    if first is None:
        raise ValueError("No rows with a complete load history and target")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "features.csv").open("x", newline="", encoding="utf-8") as xfile, \
         (args.output_dir / "labels.csv").open("x", newline="", encoding="utf-8") as yfile:
        feature_writer = csv.DictWriter(xfile, fieldnames=["origin_utc", "target_end_utc", *first[2]])
        label_writer = csv.writer(yfile)
        feature_writer.writeheader()
        label_writer.writerow(("target_end_utc", "consumption"))
        count = 0
        from itertools import chain
        for origin, target, features, label in chain((first,), rows):
            feature_writer.writerow({"origin_utc": origin.isoformat(), "target_end_utc": target.isoformat(), **features})
            label_writer.writerow((target.isoformat(), label))
            count += 1
    print(f"Wrote {count} rows to {args.output_dir}")


if __name__ == "__main__":
    main()
