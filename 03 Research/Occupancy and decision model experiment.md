# Occupancy and OpenAI Decisions experiment

**Status:** A reproducible feature generator has been implemented and tested offline. No real API requests have been made, no occupancy ground truth has been identified, and no predictive improvement has been measured. See [generator instructions](../04%20Experiments/decision_features/README.md). The user's suggestion is to estimate the number of occupants and use those estimates as forecasting features. Course assignment and data were checked in Moodle on 6 November 2026.

## The idea

At the forecast origin `t`, use only consumption readings and any other inputs available by `t` to estimate a distribution over the number of people at home in the next target hour: 0, 1, 2, 3, 4 or 5 (assuming the assignment's five-person household and no guests). This is a modelling assumption: guests, household changes and partial-hour occupancy can violate it. Before testing, define whether the label refers to occupancy at the beginning, end, or any point within the target hour.

The [OpenAI Decisions API](https://developers.openai.com/api/docs/guides/decisions) accepts text, JSON and images, and returns predicate probabilities or choice/score distributions. A single mutually exclusive choice distribution is preferable to independently asking cumulative questions; derive `P(N≥1)` through `P(N≥4)` by summing the appropriate entries so probabilities are internally consistent. Its outputs would be model estimates, not measured or automatically calibrated occupancy probabilities. A structured output cannot establish that anyone was actually home.

The supplied hourly dataset has electricity and weather, with no identified per-hour occupancy labels. Heating, appliances, household behaviour and weather can produce similar aggregate consumption patterns under different occupancies. Count-level probabilities therefore cannot be validated or calibrated on this dataset alone; do not report them as known occupant counts or ground truth. Without labels one can still measure whether a frozen proxy feature helps **electricity forecasting** on genuinely later validation data. Observable sensor or occupancy labels for this same period, if actually available with provenance and permission, would enable separate occupancy and calibration evaluation; newly collected labels describe a later period and do not validate the historical counts automatically.

## Candidate decision features

All should use only origin-available data and be called **model-generated proxy scores** unless independently validated:

- Count distribution or coarse low/medium/high household-activity score; test whether finer categories add value.
- Likelihood of a high-usage episode in the next hour or an upcoming increase/decrease, with objective thresholds learned only on training data. These are directly checkable against future electricity readings on held-out data.
- Activity-regime or historical-pattern similarity labels based only on recent consumption and local hour, plus change or atypicality score. Measurable past-pattern criteria are safer than pretending to know the household's behaviour.
- Indicators of heating-sensitive versus short-spike dominated recent load may be tested as tentative pattern descriptions; no claim of actual appliance identity or occupancy without evidence.

A simple local alternative is to derive recent variability, ramps, rolling means, typical hour-of-week deviations, and clustering or a filtered latent-state model trained only on past fitting data. Latent states are unlabeled and do not establish occupancy counts. Historical published data, weather-vintage and household-calendar assumptions need the same availability checks.

## Controlled trial

1. Start with the proposed chronological protocol and persistence/daily/weekly baselines. Build a shared neural forecasting baseline with raw lags, calendar and available past weather.
2. Hold origin/target timestamps, preprocessing, input coverage, training window, architecture, tuning budget and MAE/RMSE evaluation fixed. Compare base model, base plus local proxy features, and base plus frozen Decisions-derived scores on exactly the same hours. Retain the same raw inputs in all groups so improvement is not just caused by extra information given only to one group.
3. Before generating scores, freeze the prompt, input formatting, model version and thresholds using only development data. Supply only a past window and known-at-origin information; never target-hour actual consumption, future weather or later labels. Document potential contamination from the external model's pretraining on historical public data; where it cannot be ruled out, qualify the results.
4. If any learned proxy model is fitted on this dataset, generate training-row features with time-ordered out-of-fold predictions; generate validation-row predictions using only earlier training data. Filtering, not retrospective smoothing, for a latent model. Refit models and preprocessing appropriately for each fold. A fixed pretrained Decisions call with no dataset fitting still must obey the same time, version and input rules.
5. Compare per-season and per-hour errors, repeated chronological folds, computational cost and latency. Test incremental value and stability, not just correlations; use simple alternative features as controls. Freeze choices before final test/Eval1–3; only the downstream consumption metrics establish whether the proxy helped that task. No user-approved API calls or spending have occurred.

The optional 24-hour forecast needs its own definition: generate all forecasts using information available at the common origin, not future actual loads or weather.

## Sources

- [OpenAI Decisions guide](https://developers.openai.com/api/docs/guides/decisions): predicate, choice and score outputs, and advice to use labeled examples for evaluation/thresholds.
- [Initial exploration](../02%20Data/Initial%20exploration.md) and [[Proposed forecast protocol]].

## Initial thresholds and feature switch (7 November 2026)

Proposed 90th-percentile fitting-period thresholds are **3.70** for consumption and **1.98** for absolute hourly change. They use 1 January 2016 through 30 September 2019 UTC, without validation or later values; UTC split interpretation remains provisional. [Generator instructions](../04%20Experiments/decision_features/README.md) describe their use and the separate `openai__` training-feature switch. No live API scores were generated during this update.
