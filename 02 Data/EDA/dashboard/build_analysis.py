"""Rebuild exploratory dashboard statistics from the development fitting period."""
import hashlib
import json
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / 'Electric-consumption/dataset_mansion.csv'
START, END = '2016-01-01T00:00:00Z', '2019-10-01T00:00:00Z'
raw = pd.read_csv(SOURCE, dtype={'Consumption': str})
raw['time'] = pd.to_datetime(raw.MTime, utc=True)
raw['consumption'] = pd.to_numeric(raw.Consumption.str.replace(',', '.', regex=False), errors='coerce')
df = raw.loc[(raw.time >= START) & (raw.time < END)].copy().reset_index(drop=True)
if df.time.duplicated().any() or not df.time.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
    raise ValueError('Fitting period timestamps must be unique and hourly')
df['local'] = df.time.dt.tz_convert('Europe/Helsinki')
df['hour'] = df.local.dt.hour
df['weekday'] = df.local.dt.dayofweek
df['month'] = df.local.dt.month
weather = {'temperature': 'AirTemperature(degC)', 'humidity': 'RelativeHumidity(%)', 'wind': 'WindSpeed(m/s)'}
for label, col in weather.items():
    df[label] = pd.to_numeric(df[col], errors='coerce')
    # target ends at s, origin s-1; observation at s-2 assumes 1 h publication delay
    df[f'{label}_available_assumed'] = df[label].shift(2)
for lag in (1,2,3,6,12,24,48,72,168):
    df[f'load_lag_{lag}'] = df.consumption.shift(lag)
for window in (3,6,24,168):
    df[f'load_mean_{window}'] = df.consumption.shift(1).rolling(window, min_periods=window).mean()
df['hour_sin'] = np.sin(2 * np.pi * df.hour / 24)
df['hour_cos'] = np.cos(2 * np.pi * df.hour / 24)
df['load_change_1h'] = df.consumption - df.consumption.shift(1)

def n(v):
    if pd.isna(v): return None
    return float(v)

def summary(group, col):
    result = df.groupby(group, observed=True).consumption.agg(['mean','median','count']).reset_index()
    return [{col: int(row[0]), 'mean': n(row[1]), 'median': n(row[2]), 'count': int(row[3])} for row in result.itertuples(index=False, name=None)]

y = df.consumption
hist, edges = np.histogram(y.dropna(), bins=np.arange(0, 10.51, 0.5))
histogram = [{'bin_start': n(edges[i]), 'bin_end': n(edges[i+1]), 'count': int(count)} for i,count in enumerate(hist)]
# Preserve any future data outside the bins as an explicit bin.
over = int((y >= edges[-1]).sum())
if over:
    histogram[-1]['count'] -= int((y == edges[-1]).sum())
    histogram.append({'bin_start': n(edges[-1]), 'bin_end': None, 'count': over})
quantiles = [{'label': f'p{p}', 'value': n(y.quantile(p/100))} for p in (1,5,25,50,75,90,95,99)]
correlations = []
for feature, label, kind in [(f'load_lag_{i}', f'Consumption {i} h before target', 'load') for i in (1,2,3,6,12,24,48,72,168)] + [(f'load_mean_{i}', f'Past {i} h consumption mean', 'load_summary') for i in (3,6,24,168)] + [(f'{v}_available_assumed', f'{v.capitalize()} (assumed available)', 'weather') for v in weather]:
    pair = df[['consumption',feature]].dropna()
    correlations.append({'feature': feature, 'label': label, 'r': n(pair.corr().iloc[0,1]), 'n': len(pair), 'kind': kind})
missing = [{'name': label, 'count': int(df[col].isna().sum()), 'percent': n(100*df[col].isna().mean())} for label,col in [('Consumption','consumption'),('Temperature','temperature'),('Humidity','humidity'),('Wind speed','wind'),('Present weather','PresentWeather(rank)')]]
heat = df.groupby(['weekday','hour']).consumption.agg(['mean','count']).reset_index()
heatmap = [{'weekday':int(r.weekday),'hour':int(r.hour),'mean':n(r['mean']),'count':int(r['count'])} for _,r in heat.iterrows()]
starts = [-100,-20,-10,0,5,10,15,20,100]
labels = ['< −20','−20 to −10','−10 to 0','0 to 5','5 to 10','10 to 15','15 to 20','≥ 20']
df['temp_bin'] = pd.cut(df.temperature_available_assumed, bins=starts, labels=labels, right=False)
bins = df.groupby('temp_bin', observed=False).consumption.agg(['mean','count']).reset_index()
temperature_bins = [{'label':str(r.temp_bin),'mean':n(r['mean']),'n':int(r['count'])} for _,r in bins.iterrows()]
monthly_trend = df.set_index('time').consumption.resample('MS').agg(['mean','count'])
monthly_series = [{'month': idx.strftime('%Y-%m'),'mean': n(row['mean']),'count': int(row['count'])} for idx,row in monthly_trend.iterrows()]
seasons=[]
for name, months in [('Winter',[12,1,2]),('Spring',[3,4,5]),('Summer',[6,7,8]),('Autumn',[9,10,11])]:
    subset=df[df.month.isin(months)][['consumption','temperature_available_assumed']].dropna()
    seasons.append({'season':name,'r':n(subset.corr().iloc[0,1]),'n':len(subset)})

# The matrix includes representative predictors, not all 196/199 columns.
matrix_features = [
    {'key': 'load_lag_1', 'label': 'Load lag 1h', 'kind': 'load'},
    {'key': 'load_lag_2', 'label': 'Load lag 2h', 'kind': 'load'},
    {'key': 'load_lag_24', 'label': 'Load lag 24h', 'kind': 'load'},
    {'key': 'load_lag_168', 'label': 'Load lag 168h', 'kind': 'load'},
    {'key': 'load_mean_3', 'label': 'Load mean 3h', 'kind': 'load'},
    {'key': 'load_mean_24', 'label': 'Load mean 24h', 'kind': 'load'},
    {'key': 'load_mean_168', 'label': 'Load mean 168h', 'kind': 'load'},
    {'key': 'hour_sin', 'label': 'Hour sine', 'kind': 'calendar'},
    {'key': 'hour_cos', 'label': 'Hour cosine', 'kind': 'calendar'},
    {'key': 'temperature_available_assumed', 'label': 'Temperature (assumed)', 'kind': 'weather_assumed'},
    {'key': 'humidity_available_assumed', 'label': 'Humidity (assumed)', 'kind': 'weather_assumed'},
    {'key': 'wind_available_assumed', 'label': 'Wind (assumed)', 'kind': 'weather_assumed'},
]
matrix_cells = []
for left in matrix_features:
    for right in matrix_features:
        pairs = df[[left['key'], right['key']]].dropna() if left['key'] != right['key'] else df[[left['key']]].dropna()
        r = pairs.iloc[:, 0].corr(pairs.iloc[:, -1]) if len(pairs) > 1 else None
        matrix_cells.append({'x': left['key'], 'y': right['key'], 'r': n(r), 'n': len(pairs)})

# IQR fences highlight candidates for review, not confirmed errors or deletion.
anomaly_features = [
    {'key': 'consumption', 'label': 'Actual consumption', 'kind': 'target'},
    {'key': 'load_change_1h', 'label': 'Actual hour-to-hour consumption change', 'kind': 'target'},
    {'key': 'load_lag_1', 'label': 'Load lag 1h', 'kind': 'load'},
    {'key': 'load_mean_24', 'label': 'Load mean 24h', 'kind': 'load'},
    {'key': 'temperature_available_assumed', 'label': 'Temperature (assumed)', 'kind': 'weather_assumed'},
    {'key': 'humidity_available_assumed', 'label': 'Humidity (assumed)', 'kind': 'weather_assumed'},
    {'key': 'wind_available_assumed', 'label': 'Wind (assumed)', 'kind': 'weather_assumed'},
]
anomaly_rules, anomaly_examples, anomaly_events = [], [], []
for item in anomaly_features:
    values = df[item['key']].dropna()
    q1, q3 = values.quantile(.25), values.quantile(.75)
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    flagged = df.loc[(df[item['key']] < low) | (df[item['key']] > high), ['time', item['key']]].copy()
    anomaly_rules.append({**item, 'n': int(len(values)), 'q1': n(q1), 'q3': n(q3), 'low': n(low),
                          'high': n(high), 'flagged': len(flagged),
                          'percent': n(100 * len(flagged) / len(values)) if len(values) else None})
    # Save each flagged feature row with the timestamps of its source readings.
    if not flagged.empty:
        flagged['distance'] = np.maximum(low - flagged[item['key']], flagged[item['key']] - high)
        examples = []
        for row in flagged.itertuples(index=False, name=None):
            stamp, value, _ = row
            if item['kind'] == 'weather_assumed':
                first_reading = last_reading = stamp - pd.Timedelta(hours=2)
            elif item['key'] == 'load_lag_1':
                first_reading = last_reading = stamp - pd.Timedelta(hours=1)
            elif item['key'] == 'load_mean_24':
                first_reading, last_reading = stamp - pd.Timedelta(hours=24), stamp - pd.Timedelta(hours=1)
            elif item['key'] == 'load_change_1h':
                first_reading, last_reading = stamp - pd.Timedelta(hours=1), stamp
            else:
                first_reading = last_reading = stamp
            event = {**item, 'target_end_utc': stamp.isoformat(),
                     'origin_utc': (stamp - pd.Timedelta(hours=1)).isoformat(),
                     'source_first_reading_utc': first_reading.isoformat(),
                     'source_last_reading_utc': last_reading.isoformat(),
                     'value': n(value), 'low': n(low), 'high': n(high),
                     'direction': 'low' if value < low else 'high', 'rule': '1.5 x IQR'}
            anomaly_events.append(event)
        for row in flagged.nlargest(5, 'distance').itertuples(index=False, name=None):
            stamp, value, _ = row
            anomaly_examples.append({**item, 'target_end_utc': stamp.isoformat(),
                                     'origin_utc': (stamp - pd.Timedelta(hours=1)).isoformat(),
                                     'value': n(value), 'low': n(low), 'high': n(high),
                                     'direction': 'low' if value < low else 'high'})

with (HERE / 'anomalies.csv').open('w', newline='', encoding='utf-8') as handle:
    fields = ['key', 'label', 'kind', 'target_end_utc', 'origin_utc',
              'source_first_reading_utc', 'source_last_reading_utc', 'value', 'low', 'high',
              'direction', 'rule']
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(anomaly_events)

project = HERE.parents[2]
sys.path.insert(0, str(project / '04 Experiments/shared_features'))
from generate_features import feature_rows, read_source
from generate_daily_features import daily_rows

records = read_source(SOURCE, True)
feature_counts = {'hourly': len(next(feature_rows(records))[2]),
                  'daily': len(next(daily_rows(records))[3]) + 1,
                  'hourly_weather_experiment': len(next(feature_rows(records, 1))[2])}
sha = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
output = {
 'meta':{'start': START, 'end': END, 'scope': 'Fitting period only. Validation and official holdouts excluded.', 'timezone': 'UTC split; Europe/Helsinki for local calendar', 'source_sha256':sha, 'unit':'unspecified source unit', 'weather_assumption':'For each target, historical weather is shifted two hours, assuming it became available one hour after its timestamp. Historical publication and revision timing remain unverified.'},
 'kpis': {'rows':len(df), 'mean':n(y.mean()),'median':n(y.median()),'p95':n(y.quantile(.95)),'max':n(y.max()),'missing_temperature':int(df.temperature.isna().sum()),'start':df.time.min().isoformat(),'last':df.time.max().isoformat()},
 'monthly':summary('month','month'), 'hourly':summary('hour','hour'), 'weekday':summary('weekday','weekday'), 'heatmap':heatmap, 'histogram':histogram, 'quantiles':quantiles, 'correlations':correlations, 'temperature_bins':temperature_bins, 'missing':missing, 'monthly_series':monthly_series, 'seasonal_temperature_correlations':seasons,
 'feature_matrix': {'features': matrix_features, 'cells': matrix_cells},
 'anomalies': {'rules': anomaly_rules, 'examples': anomaly_examples,
               'note': 'IQR fences use the fitting-period distribution of each variable. Flags mark candidates for review, not verified errors. Load changes use target outcomes retrospectively. Weather availability is assumed.'},
 'feature_counts': feature_counts,
}
(HERE/'analysis.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps({'kpis':output['kpis'],'correlations':correlations,'missing':missing,'temperature_bins':temperature_bins,'seasons':seasons},indent=2))
