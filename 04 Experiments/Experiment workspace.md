# Experiment workspace

Use this area for individual implementations, configurations, data versions, results, and reproducible evidence. The PDF requires a shared results table comparing the test period and three evaluation periods, and asks about large-error days.

| Experiment | Student | Model | Status | Results / evidence |
|---|---|---|---|---|
| Consumption baselines | Group benchmark | Next hour and next 24 hours | Validation scored | [[Baseline results]] |

## Shared next-hour features

The [shared feature generator and usage guide](shared_features/README.md) provide the four students with aligned load history, calendar and optional historical weather features. The [generated dataset](shared_features/generated_weather_1h/features.csv) and [separate labels](shared_features/generated_weather_1h/labels.csv) contain 56,063 aligned rows. The [manifest](shared_features/generated_weather_1h/manifest.json) records predictor names and hashes. Weather uses the assignment's FMI observations with a **provisional one-hour publication delay**, whose historical availability and revision timing have not been verified. See the guide before interpreting results and preserve the assignment's chronological holdouts.

## Prepared hourly and daily datasets

The [rebuild instructions](shared_features/README.md) and [dataset manifest](shared_features/prepared_no_weather/manifest.json) cover both next-hour forecasts and 24-hour forecasts issued once daily at 00:00 UTC. Each dataset has separate fit, validation, test and three evaluation splits with separately keyed features and labels. The 24-hour profiles remain wholly within one split. The prepared version omits weather because historical publication and revision times are unverified. The project owner confirmed immediate meter availability. UTC split boundaries and the daily forecast schedule remain provisional.

## Model experiments

See [[Starter results]] and the [shared model guide](models/README.md).

## Decisions feature generator

The [standalone generator and instructions](decision_features/README.md) produce experimental feature scores with cache-backed replay. A three-hour API sample succeeded. The OpenAI feature group is off for the current forecasting work.
