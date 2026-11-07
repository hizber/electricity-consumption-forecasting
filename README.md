# Electricity consumption forecasting

Group project on next-hour and next-24-hour electricity consumption forecasting using neural networks. The workspace contains shared feature-generation code, baselines, exploratory analysis and project notes.

## Data

The original course dataset, assignment PDF and generated datasets are excluded from this repository. Obtain the course data through Moodle and place `dataset_mansion.csv` in `02 Data/Electric-consumption/`. Do not commit that file, generated datasets, prediction tables or API caches. Some notes refer to local artifacts that are generated after the data is supplied.

## Rebuild the prepared datasets

Use Python 3.10 or later. From `04 Experiments/shared_features`, run:

```bash
python3 build_datasets.py --output-dir prepared_no_weather
python3 -m unittest -v test_generate_features test_generate_daily_features
```

The output has separate hourly and daily splits and labels, with source and output hashes. It assumes UTC split boundaries, a meter reading available at the forecast origin, and a 00:00 UTC daily forecast. The prepared dataset omits weather because historical publication and revision timing remain unverified. See [the data guide](04%20Experiments/shared_features/README.md) for the full assumptions and split definitions.

## Exploratory dashboard

From `02 Data/EDA/dashboard`, install pandas and NumPy in your Python environment, then run `python3 build_analysis.py` and `python3 build_dashboard.py`. Open `dashboard.html` locally. The dashboard's statistics cover the fitting period only. The generated dashboard, statistics and anomaly export contain derived household data and should remain local.

## Group work

See the [data preparation checklist](02%20Data/Data%20preparation%20checklist.md), [forecast protocol](04%20Experiments/Proposed%20forecast%20protocol.md) and [baseline results](04%20Experiments/Baseline%20results.md). Each student is responsible for their own neural-network implementation and records their AI use.
