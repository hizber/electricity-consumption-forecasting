# Proposed forecast protocol

**Status:** Proposal for group discussion, 6 November 2026. Both baseline comparisons are recorded in [[Baseline results]]. The live Moodle assignment and return-box listing were rechecked; no relevant change found, and the return boxes show 8 December 2026 at 23:59. The displayed page did not specify a timezone.

## Forecast definition

At the end of hour `t`, the consumption measurement for the hour ending at `t` is available, and the task is to predict consumption for the hour ending at `t + 1`. The project owner confirmed immediate meter-reading availability on 7 November 2026. The timezone of the assignment's split boundaries remains provisional and should be confirmed before final scoring. Use only information available at the forecast origin. Observed weather for the target hour is not available beforehand unless a suitable forecast is actually supplied.

For the optional next-day task, the group will provisionally issue one forecast at 00:00 UTC for the next 24 elapsed hours. The house is in Jyväskylä, Finland. UTC is used for chronological ordering and the provisional split and forecast schedule; calendar features use `Europe/Helsinki`, including daylight-saving transitions. This is not yet a confirmed local-calendar-day schedule.

## Development split

A proposed initial validation window is 1 October 2019 through 30 September 2020, inside the assignment's training period. Fit on earlier training data, then validate on this later period. Use UTC date boundaries provisionally; confirm the intended timezone. A year covers seasons, though it is only one historical year. Fit scalers and imputers on the fitting data only. Preserve the assigned test and three evaluation periods for final reporting after development decisions are frozen.

## Baselines

For the hour ending `t + 1`, compare the latest observed consumption `y[t]`, consumption 24 elapsed hours before the target `y[t − 23]`, and consumption 168 elapsed hours before the target `y[t − 167]`. Compare all methods on the same eligible timestamps. These use elapsed UTC hours; around daylight-saving transitions they need not represent the same local-clock hour. Establish that each input was available at prediction time.

## Metrics and group work

Use MAE as the primary checkpoint-selection metric and report RMSE and MAPE as complementary measures on identical targets. MAE and RMSE use the original source units, which remain unspecified. MAPE excludes zero actuals and reports their count. For the daily task, compute MAPE separately for individual hourly targets and for the summed 24-hour total. Each student studies and implements a neural network as required; assess the models against the baselines and each other, then report the required test and evaluation results and investigate large-error days. Avoid repeatedly tuning on test or evaluation sets. Record configurations, results, group decisions and individual AI use.

See [[Initial exploration]], [[Assignment requirements]] and [[Experiment workspace]].

The [starter results](Starter%20results.md) record the frozen shared-model settings and reference data checksums. The four hourly starters and one daily starter were scored on the test and evaluation periods on 10 November 2026, and their aggregate results are visible to the group. Those periods have therefore been exposed for subsequent model-development decisions. Further choices influenced by those results cannot be described as having been made without looking at the evaluation periods. Record such choices in the final report and use fit/validation for subsequent development.

The fixed-model evaluation reuses fitting-only preprocessing throughout each held-out period. Past readings can enter later forecasts after they become available, including readings from earlier held-out hours. They do not update the fitted model. A future rolling-retraining experiment would require a separate protocol and time-ordered refitting at each scheduled origin.
