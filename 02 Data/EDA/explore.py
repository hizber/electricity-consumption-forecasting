"""Initial descriptive inspection of the verified Moodle CSV; no models are fitted.

Period boundaries below use UTC timestamp dates provisionally. Confirm with the teacher
whether the prescribed calendar dates refer to UTC or local time before final scoring.
"""
from pathlib import Path
import json
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / 'Electric-consumption' / 'dataset_mansion.csv'
df = pd.read_csv(DATA)
df['MTime'] = pd.to_datetime(df['MTime'], utc=True, errors='raise')
df['Consumption'] = pd.to_numeric(df['Consumption'].str.replace(',', '.', regex=False), errors='raise')
train = df[(df.MTime >= '2016-01-01') & (df.MTime < '2020-10-01')].copy()
assert not train.empty
train['local_time'] = train.MTime.dt.tz_convert('Europe/Helsinki')
train['hour'] = train.local_time.dt.hour
train['month'] = train.local_time.dt.month
train['weekday'] = train.local_time.dt.dayofweek
periods = {'training': ('2016-01-01','2020-10-01'), 'test': ('2020-10-01','2021-10-01'), 'evaluation_1': ('2021-10-01','2022-01-01'), 'evaluation_2': ('2022-01-01','2022-03-01'), 'evaluation_3': ('2022-03-01','2022-06-01')}
summary = {'row_count': len(df), 'first_utc':str(df.MTime.min()), 'last_utc':str(df.MTime.max()), 'duplicate_timestamps':int(df.MTime.duplicated().sum()), 'missing_timestamps_in_range':int(len(pd.date_range(df.MTime.min(),df.MTime.max(),freq='h').difference(df.MTime))), 'missing_by_column':{k:int(v) for k,v in df.isna().sum().items()}, 'rows_by_utc_period':{k: int(((df.MTime>=start)&(df.MTime<end)).sum()) for k,(start,end) in periods.items()}, 'training_consumption': {k:float(v) for k,v in train.Consumption.describe(percentiles=[.05,.5,.95,.99]).items()}, 'training_zero_values':int((train.Consumption==0).sum()), 'training_negative_values':int((train.Consumption<0).sum()), 'training_weather_available':int(train['AirTemperature(degC)'].notna().sum()), 'training_rows':len(train), 'consumption_unit':'not specified in supplied column description'}
summary['training_january_mean_utc'] = float(train.loc[train.MTime.dt.month == 1, 'Consumption'].mean())
summary['training_july_mean_utc'] = float(train.loc[train.MTime.dt.month == 7, 'Consumption'].mean())
summary['training_contemporaneous_temperature_correlation'] = float(train['Consumption'].corr(train['AirTemperature(degC)']))
summary['training_missing_present_weather_rank'] = int(train['PresentWeather(rank)'].isna().sum())
summary['temperature_2016_nonmissing'] = int(train.loc[train.MTime.dt.year == 2016, 'AirTemperature(degC)'].notna().sum())
(HERE/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary, indent=2))
