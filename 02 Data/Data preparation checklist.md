# Data preparation checklist

**Purpose:** Prepare a reproducible dataset for the group's next-hour and next-24-hour models. Priorities are proposed for group discussion. See [[Initial exploration]], [[Proposed forecast protocol]] and [shared dataset guide](../04%20Experiments/shared_features/README.md).

## P0 — Agree on the forecast contract

- [ ] Confirm when the hourly meter reading actually becomes available. The existing pipeline assumes the reading for the hour ending at the forecast origin is immediately available. Adjust lags and baselines if there is a delay.
- [ ] Confirm the timezone used for the assignment's date boundaries. The prepared splits use UTC provisionally.
- [ ] Agree whether the 24-hour forecast is issued once at 00:00 UTC for the next 24 elapsed hours. If a Helsinki calendar day is required, define daylight-saving handling and rebuild.
- [ ] Clarify the consumption unit for the report. Preserve the source's unspecified units until confirmed.

## P1 — Make the current datasets ready for consistent model input

- [x] Generate separate hourly and fixed-origin 24-hour datasets with chronological fit, validation, test and evaluation splits, and separate feature and label files. Record source, scripts, split settings and hashes in the [manifest](../04%20Experiments/shared_features/prepared_no_weather/manifest.json).
- [x] Check alignment, split boundaries, 24-hour profiles and a reproducible rebuild. The daily profiles crossing a split boundary are excluded.
- [ ] Agree on predictor groups and compare on the same eligible target hours. Start with consumption history and compare adding local calendar features. Track explicit column lists and model input shapes.
- [ ] Decide how each model will use the 168 history readings. If comparing shorter input windows, exclude longer lags and rolling summaries from those comparisons. For a 24-output model, group the 24 daily rows by origin and order by horizon.
- [ ] Implement or agree on preprocessing for each model. Fit any scaler and imputer on the fitting split only, reuse them on validation and later periods, and save their parameters and column order. Prepared values are not scaled.
- [ ] Verify that `origin_utc` and `target_end_utc` are metadata and that labels never enter the predictors. For daily models, never use actual consumption from later within the forecast horizon.
- [ ] Freeze the first shared dataset version, code hashes, split definitions, preprocessing settings and metrics so each student can reproduce the comparison. Keep a record if an assumption changes.

## P2 — Review quality and feature choices on fitting data

- [ ] Inspect the surrounding hours and source records for a sample of flagged peaks and sudden changes. An IQR flag is a review candidate, not proof of an error. Retain values unless a specific correction is supported and documented. See the [flagged events](EDA/dashboard/anomalies.csv).
- [ ] Review duplicate or missing timestamps, missing values, join alignment and daily-profile completeness whenever the source or pipeline changes. Preserve missingness rather than substituting future data.
- [ ] Use fitting data and chronological validation to compare groups such as rolling averages, daily/weekly lags and calendar features. High correlation between features is a reason to test redundancy, not a reason to discard a column automatically.
- [ ] Compare all models against the existing previous-day baseline on identical eligible timestamps. Use validation MAE and RMSE, with seasonal and hourly breakdowns. See [[Baseline results]].
- [ ] Record the final available evaluation timestamps. The original source ends at 20:00 UTC on 31 May 2022, so the last day is incomplete.

## P3 — Investigate weather as a separate experiment

- [ ] Check the provenance of 2016 weather values and whether historical weather publication times and revisions can be established. The assignment describes weather as beginning in 2017, but the supplied CSV contains earlier observations.
- [ ] Compare consumption-plus-calendar with the same inputs plus past temperature, and optionally humidity and wind. The existing hourly weather dataset assumes a one-hour delay; report the assumption and do not claim its historical availability is verified.
- [ ] Assess missingness and causal imputation on fitting data, including long rolling weather windows. Do not use observed target-hour weather or future values to fill gaps.
- [ ] For a daily weather experiment, use only weather history available at the common origin or a forecast vintage issued and published by that origin. Verify source coverage and timestamps before adding archived forecasts.

## Before final reporting

- [ ] Freeze feature choices, preprocessing and hyperparameters using fitting and validation data. Evaluate the official test and three evaluation periods only after decisions are fixed.
- [ ] Save the dataset manifest, preprocessing parameters, model configuration and per-period scores with each student's implementation. Document any unresolved assumptions and the treatment of missing or flagged data.
