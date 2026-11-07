# Proposed forecast protocol

**Status:** Proposal for group discussion, 6 November 2026. Both baseline comparisons are recorded in [[Baseline results]]. The live Moodle assignment and return-box listing were rechecked; no relevant change found, and the return boxes show 8 December 2026 at 23:59. The displayed page did not specify a timezone.

## Forecast definition

At the end of hour `t`, assume the consumption measurement for the hour ending at `t` is available, and predict consumption for the hour ending at `t + 1`. Confirm the measurement availability and assignment split timezone before final scoring. Use only information available at that forecast origin. Observed weather for the target hour is not available beforehand unless a suitable forecast is actually supplied.

## Development split

A proposed initial validation window is 1 October 2019 through 30 September 2020, inside the assignment's training period. Fit on earlier training data, then validate on this later period. Use UTC date boundaries provisionally; confirm the intended timezone. A year covers seasons, though it is only one historical year. Fit scalers and imputers on the fitting data only. Preserve the assigned test and three evaluation periods for final reporting after development decisions are frozen.

## Baselines

For the hour ending `t + 1`, compare the latest observed consumption `y[t]`, consumption 24 elapsed hours before the target `y[t − 23]`, and consumption 168 elapsed hours before the target `y[t − 167]`. Compare all methods on the same eligible timestamps. These use elapsed UTC hours; around daylight-saving transitions they need not represent the same local-clock hour. Establish that each input was available at prediction time.

## Metrics and group work

Propose MAE as the primary metric and RMSE as a complementary measure, in the original source units, which remain unspecified. Report the same metrics on identical timestamps for each approach. Each student studies and implements a neural network as required; assess the models against the baselines and each other, then report the required test and evaluation results and investigate large-error days. Avoid repeatedly tuning on test or evaluation sets. Record configurations, results, group decisions and individual AI use.

See [[Initial exploration]], [[Assignment requirements]] and [[Experiment workspace]].
