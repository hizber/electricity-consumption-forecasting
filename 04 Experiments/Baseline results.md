# Baseline results

The previous-day baseline had the lowest MAE, RMSE and MAPE for both horizons on the development validation year. These are the scores to beat when comparing neural networks on the same timestamps.

| Forecast | Baseline | Hourly MAE | Hourly RMSE | Hourly MAPE | 24-hour total MAE | 24-hour total RMSE | 24-hour total MAPE |
|---|---|---:|---:|---:|---:|---:|---:|
| Next hour | Latest reading | 0.9705 | 1.2683 | 78.36% | — | — | — |
| Next hour | Previous day | **0.9181** | **1.2444** | **70.31%** | — | — | — |
| Next hour | Previous week | 0.9554 | 1.2816 | 75.79% | — | — | — |
| Next 24 hours | Previous day profile | **0.9188** | **1.2451** | **70.25%** | **7.2458** | **9.9777** | **16.37%** |
| Next 24 hours | Previous week profile | 0.9562 | 1.2826 | 75.77% | 9.7095 | 13.0200 | 23.19% |

MAPE averages the absolute error as a percentage of the actual consumption. The smallest actual hourly reading is 0.26, and every scored actual is nonzero. In the next-hour previous-day baseline, hours below 0.5 make up about 13% of the scored hours and contribute about 27% of the summed absolute percentage error. Keep MAE as the primary metric for model selection. For daily-total MAPE, sum the predicted and actual hourly values for each profile before calculating its percentage error. The script reports the count of zero actuals excluded if used on another window.

Both comparisons use validation targets from 1 October 2019 through 30 September 2020, with provisional UTC date boundaries. The next-hour comparison covers 8,784 forecasts. The 24-hour comparison covers 365 forecasts issued at 00:00 UTC, each with 24 hourly targets after the origin. A profile is scored only if all its target-end timestamps fall in the validation period. This excludes the profile starting on the final day, whose last target is at the start of the official test period. UTC midnight is the initial schedule for the fixed 24 elapsed-hour task. Confirm the schedule if the group chooses a Helsinki calendar-day forecast.

The 24-hour predictions are generated together using readings available at the origin. Previous-day and previous-week methods use readings 24 or 168 elapsed hours before each target respectively. For the total metrics, the 24 values are summed before calculating the error for that origin. The source does not specify the unit. The baselines use consumption only and assume that the last completed reading is available at the forecast origin.

Run `python3 run_baselines.py` from this folder to reproduce the [scores](baselines/scores.json), [next-hour predictions](baselines/next_hour_predictions.csv), and [24-hour predictions](baselines/next_24_hours_predictions.csv). The CSVs are evaluation tables and include the actual consumption for scoring. I checked the saved predictions against the source readings and independently recalculated the scores. The official test and evaluation periods are reserved for final reporting.

See [[Proposed forecast protocol]] and [[Experiment workspace]].
