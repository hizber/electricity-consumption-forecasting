# OpenAI Decisions feature generator

This standalone Python tool turns the last 24 completed hourly consumption measurements and known local calendar time into experimental OpenAI Decisions scores for the next hour. It implements the [Decisions API](https://developers.openai.com/api/docs/guides/decisions), using one request per target hour. Python 3.9+ and the operating system's IANA timezone database are required; no pip packages are needed. The default input is the verified course CSV in this vault.

**These are model-generated proxy scores, not verified occupancy measurements.** The data has no identified hourly occupancy labels, so we cannot validate occupancy calibration. Test whether these features improve the consumption forecast on time-ordered validation data. The assignment still requires each student's own neural-network implementation. No Decisions API calls were made while building or testing this tool.

## Feature definitions

Each row corresponds to a forecast made at `origin_utc` for the hour ending at `target_end_utc = origin_utc + 1 hour`. We assume that the measurement for the hour ending at the origin is already available; confirm the real latency. The input history contains only completed hours ending no later than the origin. The source dictionary describes timestamps as UTC. Target local date/time and weekday use `Europe/Helsinki`; the timezone of the assignment's calendar split boundaries remains unconfirmed.

- `occupancy_0` through `occupancy_4`, `occupancy_5_plus`: the model's scores for mutually exclusive estimates of how many people are present **at the beginning of the target hour**. Five or more allows guests. These are not proven calibrated; the only known household information is the assignment's five residents.
- `occupancy_ge_1` through `occupancy_ge_5`: sums of the appropriate mutually exclusive scores. `5_plus` cannot distinguish six or more.
- `activity_rising`, `activity_falling`, `activity_fluctuating`, `activity_stable`: model scores for the *observed history*, using the priority and change-threshold rules recorded in each request. These are load-pattern scores, not occupancy labels.
- `high_load`: model-estimated probability that the target-hour consumption is **strictly greater** than the chosen high-load threshold.
- `increase`: model-estimated probability that target-hour consumption is **strictly greater** than the last observed consumption.
- `large_change`: model-estimated probability that the absolute difference between target and last observed consumption is **at least** the chosen change threshold.
- `atypical`: model-estimated probability that target-hour consumption differs from the mean of the supplied history by **at least** the change threshold. This is relative to the recent history, not a learned same-hour-of-week profile.

Both thresholds are mandatory, in the source's **unspecified units**. Determine and freeze them from the fitting data of each chronological training fold, or predefine them independently. Never calculate them from validation/test targets. The numbers in the examples below are **illustrative placeholders**, not validated choices. The model receives no target-hour observation or target-hour weather. Weather is intentionally omitted because observation publication latency has not been established.

## Preview without making API calls

From this folder:

```bash
python3 decision_features.py \
  --start 2019-10-01T00:00:00Z --end 2019-10-01T03:00:00Z \
  --high-threshold 4 --change-threshold 1
```

This prints the first exact request and the number of eligible selected target hours. It makes **zero API calls and writes no feature output**. The requested start is inclusive and end exclusive, based on the *target end* timestamp. A complete hourly input window and a corresponding target timestamp in the source are required; incomplete histories are skipped. This is a retrospective dataset generator: it does not produce rows beyond the last timestamp in the input CSV. The target's actual consumption is never included in a request. The default `--limit` is five target hours.

## Run a small sample

With access to the Decisions API, provide the API key through your environment (do not put it into files, commands saved to history, or the vault). Then explicitly opt into calls:

```bash
python3 decision_features.py \
  --start 2019-10-01T00:00:00Z --end 2019-10-01T03:00:00Z \
  --high-threshold 4 --change-threshold 1 \
  --limit 3 --max-new-requests 3 --execute \
  --output results/validation-sample.csv --cache-dir results/cache
```

Each uncached hour creates one API request, may incur charges, and sends the recent household consumption and household context to OpenAI. `--max-new-requests` is a request-count cap, **not a monetary budget**. Increase `--limit` and the request cap deliberately after reviewing usage and the [current API pricing](https://developers.openai.com/api/docs/guides/decisions). Each successful response is validated and checkpointed before the next request. Tiny rounding differences in returned choice distributions are normalized to sum to one in the CSV; the cached raw response is unchanged. HTTP or connection failures stop the run without automatic retries; check the cause and rerun. Partial output exits with code 2 when the request cap is reached. Rerun the same command to resume from the cache. Changing date range, thresholds, model, input file, history, limit or script requires a different output path.

## Reproducibility and limitations

- The output CSV holds forecast/target timestamps, request hashes and scores; it intentionally does not include target consumption.
- Each `cache/<sha256>.json` file contains the exact request and complete response, returned model and usage fields, and cache time. The output manifest records the input and script hashes, frozen configuration and first returned model identifier; the tool rejects a different returned identifier later in the run. Keep the output, manifest and cache together. They contain household readings; choose storage and sharing accordingly.
- Rerunning with the same input, code, configuration and cache reproduces the saved results without new calls. `--execute --max-new-requests 0` replays existing cached rows without an API key and stops before a missing entry. An existing output is checked against its cached responses before it can be rewritten; restore missing cache files rather than regenerating them silently.
- The documented model is currently `gpt-6-luna`. OpenAI does not document a seed or immutable snapshot for this endpoint, so a new API call later is **not guaranteed** to reproduce a previous response. Pretraining contamination on the historical period cannot be ruled out. Preserve cached responses and qualify comparisons.
- A no-refusal response with all required probability categories and values in `[0,1]` is required. Failed, refused, incomplete or malformed responses are not silently cached or substituted. Probability-shaped results are not automatically calibrated or useful. Temperature features, fine-tuning, occupancy ground truth and a 24-hour common-origin forecast are not implemented.
- To join features to labels for evaluation, use `target_end_utc`; never use the future target value as an input. For stacking or thresholds learned on this dataset, use chronological fitting and forward out-of-fold feature generation. Decisions itself is used as a fixed external model here, not trained on this household's records.

## Tests

```bash
python3 -m unittest -v
```

The tests run offline with synthetic inputs and fake responses. They check causal windows, UTC and parsing, response validation, cumulative consistency, and caching; they cannot establish live API access or prediction quality.

## Candidate thresholds for the first validation experiment

Run `python3 choose_thresholds.py --start 2016-01-01T00:00:00Z --end 2019-10-01T00:00:00Z` in this folder to reproduce the calculation. The stored `candidate_thresholds.json` records its provenance.

The initial candidates are **3.70** for high consumption and **1.98** for absolute change, in the dataset's unspecified units. They are 90th percentiles calculated with linear interpolation from the earlier fitting period, 1 January 2016 through 30 September 2019, using UTC boundaries provisionally. Hourly changes use pairs where both readings are in that fitting period and exactly one hour apart. Recalculate thresholds independently for other chronological fitting folds; do not fit them on validation or test data. They are initial candidates, not guaranteed optimal.

## Turn the OpenAI group on or off

`feature_selection.py` provides a training helper. Base rows must already contain valid, past-only features and the target end time; labels must be kept separately. It does not train a model:

```python
from feature_selection import build_training_features, read_csv_rows

# `rows` and `labels` are your existing, time-aligned training data.
base_columns = ['lag_1', 'hour']  # example: replace with your actual predictors

# Off: no OpenAI feature file is read.
X_without_openai = build_training_features(rows, base_columns, include_openai=False)

# On: join the generated scores by target-end time.
openai_rows = read_csv_rows('results/validation-sample.csv')
X_with_openai = build_training_features(
    rows, base_columns, include_openai=True, openai_rows=openai_rows
)
```

When enabled, every score is named `openai__...` and each base row must have exactly one matching generated target timestamp. Restrict `rows` to timestamps covered by the generated file. The helper rejects missing or duplicate target matches, unexpected score columns, metadata used as predictors, and already-prefixed collisions. Explicitly selecting base columns helps avoid accidental labels, but **you must verify each base feature was available at the forecast origin**; a name check cannot establish causality. Keep target labels separately and in the same row order. For a fair comparison, use identical target hours, splits, preprocessing rules and training procedures for both versions.
