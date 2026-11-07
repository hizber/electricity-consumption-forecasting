# Initial data exploration

The interactive [EDA dashboard](EDA/dashboard/dashboard.html) contains fitting-period KPIs, distributions, calendar patterns, correlations, weather comparisons and a feature-testing plan. Its [rebuild instructions](EDA/dashboard/README.md) document the narrower fitting period and timing assumptions.

**Status:** descriptive first pass, 6 November 2026. No model has been trained. The analysis uses the live Moodle dataset, compared by SHA-256 to the existing course archive. [[Source and open questions]] records the verification.

## Scope and reproducibility

The source CSV has 56,231 rows and 18 columns. [[explore.py]] reproduces the quality counts and training summary in [[summary.json]]. It uses UTC calendar boundaries for the assignment splits **as a provisional assumption**: the dataset description states UTC, but the assignment does not specify the split timezone. Confirm the interpretation with the instructor before final evaluation. Patterns below are calculated only on the designated training period (2016-01-01 00:00 to 2020-10-01 00:00 UTC, end exclusive; 41,640 rows). Quality checks cover the full file without fitting a model.

## Key findings

- The timestamps run hourly from 2015-12-31 22:00 to 2022-05-31 20:00 UTC, with no duplicate or missing hourly timestamps in that range. The final evaluation period ends three timestamp hours before the end of 31 May if UTC calendar dates are intended.
- Consumption is stored as text with decimal commas. After conversion, no target values are missing, zero, or negative across the file. In training, mean is 1.845, median 1.65, 95th percentile 4.39, and maximum 10.4. **The source does not specify the unit**; do not label values kWh without confirmation.
- In the full file, air temperature is missing in 208 rows and present-weather rank in 32,441 of 56,231 rows (57.7%). Missing weather is not the same as zero. In training, rank is missing in 24,083 of 41,640 rows (57.8%).
- The PDF says weather starts in 2017, but the file has recorded temperature from its first timestamp in December 2015; 2016 contains temperature for 8,782 of 8,784 UTC-dated rows. Investigate provenance and usability before deciding how to handle 2016 weather.
- In training, the mean consumption for January timestamps is 3.139 versus 0.812 for July timestamps (UTC months). Contemporaneous temperature and consumption have Pearson correlation −0.636. These are descriptive associations and do not prove causality or that future weather will be available for prediction.
- The exported `Unnamed: 0` column is a row index. Latitude, longitude, and place each have one nonmissing station value and should not be mistaken for varying house features.
- `MTime` is UTC according to the dictionary. The supplied data-generation script converts original Helsinki timestamps to UTC. For human calendar features, convert to `Europe/Helsinki` with daylight-saving handling; keep the split timezone question separate.

## Forecasting implications and next checks

1. Define the forecast origin and target precisely: a timestamp labels the **end** of the preceding consumption hour. Confirm when each measurement becomes available, then construct lag features without using information from after the forecast origin.
2. The provided weather is observed over each hour. Realized target-hour weather cannot be assumed available before that hour; distinguish historical observations from actual forecasts and document any allowed assumptions.
3. Preserve the assignment’s test and three evaluation periods. Set aside a chronological validation period **inside training** for development and fit imputation/scaling only on each training fold. Report results in the original target scale. Confirm which timezone defines boundaries.
4. Compare simple last-hour and same-hour-previous-day/week baselines, then each student’s neural network on the same forecast origins and metrics. Agree on a primary error metric; report MAE and RMSE as useful complementary measures, rather than calling regression accuracy a percentage without definition.
5. Inspect missing-data runs, unusual peaks and holidays; analyze errors by season, hour and high-error day after evaluating. For the optional 24-hour forecast, define whether all 24 predictions are made at once and what future inputs are available.

These are proposed next steps, not extra assignment requirements. No result above establishes which model will be best. Avoid repeatedly tuning on the required test and evaluation periods.
