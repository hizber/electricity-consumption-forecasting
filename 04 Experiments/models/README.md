# Shared neural-network starting points

These four PyTorch models are starting points for the next-hour task. Each group member needs to study and adapt their chosen network, produce their own implementation and documentation, and record how they used AI. The code here does not establish an individual student's authorship or satisfy those deliverables by itself.

| Model | Structure | Points to investigate |
|---|---|---|
| LSTM | A recurrent LSTM encodes the consumption sequence. Its final output and the calendar context feed a linear prediction head. | Memory gates, hidden size and input history length. |
| GRU | A recurrent GRU encodes the sequence. Its final output and context feed a linear prediction head. | Update and reset gates, accuracy, parameter count and training time compared with LSTM. |
| TCN | Dilated causal convolutions use left padding and residual blocks. The last encoded step and context feed a linear head. | Dilation and receptive field. The default two-convolution blocks have a 253-hour receptive field; a shorter input still supplies only the chosen history. |
| Transformer | A Transformer encoder processes the historical sequence with sinusoidal positional encodings. The last encoded step and context feed a linear head. | Attention, positional information, training cost and overfitting. Attention covers the completed historical window. |

All models accept history in chronological order, oldest first, with shape `[batch, time, 1]`, plus forecast-time context of shape `[batch, context_features]`. They return a scalar prediction per sample with shape `[batch]`. The default hourly training command uses lagged consumption and calendar features. Rolling summaries and conditional past weather are opt-in. The target-hour calendar is known at the forecast origin. Models and context dimensions must match the saved preprocessing configuration.

## Setup and input data

Use Python 3.12 and the repository's pinned `requirements-lock.txt`. For example, from the repository root with `uv` installed:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-lock.txt
```

The lock records the versions used in the project's current environment. Package availability can differ by platform. The course CSV is excluded from GitHub; follow the root README and [dataset preparation guide](../shared_features/README.md) to obtain it and build a local verified dataset. Use a dataset directory containing `manifest.json` and its hourly fit and validation splits. The examples below use `rebuild_v2`, the verified build name in this workspace; substitute the local output directory in another checkout.

## Train and compare

Run from the repository root. The output directory must not exist. This command trains the LSTM with 24 completed hourly readings and calendar context:

```bash
.venv/bin/python '04 Experiments/models/train.py' \
  --dataset '04 Experiments/shared_features/prepared_no_weather/rebuild_v2' \
  --model lstm --history 24 --epochs 10 --patience 3 --seed 42 \
  --output '04 Experiments/models/runs/lstm-example'
```

Use `--model gru`, `--model tcn` or `--model transformer` for the other architectures and choose a distinct output directory for each run. `--history` accepts 24, 72 or 168 hours. The remaining options include `--batch-size` and `--learning-rate`. Keep target hours, splits and metrics consistent when comparing models. A history-length experiment should also exclude longer-context engineered features; the current command selects no rolling summaries.

The loader verifies split hashes, counts and feature-label alignment against the manifest. Preprocessing is fitted on the fitting split only. The training loop uses scaled-target MSE for gradient updates and selects the best epoch using validation MAE in the original source units, with early stopping controlled by patience. Validation and previous-day baseline MAE, RMSE and MAPE are also recorded. Test and evaluation splits are not opened by this command. Freeze feature and model decisions using fitting and validation data before opening held-out data.

MAPE is the mean absolute error divided by the absolute actual value, expressed as a percentage. Zero actuals are excluded from MAPE and counted in the report. When all actuals are zero, MAPE is undefined and stored as `null`. MAE and RMSE include all observations. MAPE can be strongly affected by small actual values. Daily reports show both MAPE over individual hourly targets and MAPE over 24-hour totals.

Each output folder contains `model.pt`, `preprocessor.json` and `metrics.json`, including selected columns, data source hash, configuration, epoch results, parameter count and elapsed training time. Run outputs are ignored under `04 Experiments/models/runs/` because fitted parameters and metrics are derived from household data. Keep all generated datasets, model outputs and course records local unless sharing has been approved. The source does not specify a consumption unit.

## Freeze and score holdouts

After development choices are recorded, run the two commands below from the repository root. Freeze every selected run before opening any holdout. The evaluator verifies the frozen configuration, data, checkpoint and environment and reports matched metrics by period, local hour and season. Largest-error days are ranked by daily MAE over complete UTC target days and remain in the ignored local report.

```bash
dataset='04 Experiments/shared_features/prepared_no_weather/rebuild_v2'
run='04 Experiments/models/runs/lstm-example'
.venv/bin/python '04 Experiments/models/evaluate.py' --create-freeze \
  --dataset "$dataset" --run "$run" --freeze-file "$run/freeze.json"
.venv/bin/python '04 Experiments/models/evaluate.py' --evaluate \
  --dataset "$dataset" --freeze-file "$run/freeze.json" --output "$run/heldout"
```

The previously reported starter freezes capture the code at commit `5495f62`. The subsequent split-boundary fix changes the preprocessing code hash, so the current evaluator intentionally rejects those old freezes. To replay an existing local freeze, export that commit into a temporary directory, link the local ignored runs directory, and use the original manifest and environment:

```bash
archive_dir="$(mktemp -d /private/tmp/electricity-frozen-XXXXXXXX)"
git archive 5495f62 | tar -x -C "$archive_dir"
ln -s "$PWD/04 Experiments/models/runs" "$archive_dir/04 Experiments/models/runs"
run="$PWD/04 Experiments/models/runs/gru_h24_verified"
dataset="$PWD/04 Experiments/shared_features/prepared_no_weather/rebuild_v2"
.venv/bin/python "$archive_dir/04 Experiments/models/evaluate.py" --evaluate \
  --dataset "$dataset" --freeze-file "$run/freeze.json" --output "$run/replay-example"
.venv/bin/python '04 Experiments/models/score_archived_freeze.py' \
  --snapshot "$archive_dir" --dataset "$dataset" --freeze-file "$run/freeze.json" \
  --output "$run/mape-replay.json"
```

The old manifest and model artifacts are local and excluded from the repository. A newly generated manifest may record different generator hashes even when its CSVs match, so it cannot replace the original manifest for this exact freeze check. Train and freeze new runs with the current code.

## Train a daily forecast

```bash
.venv/bin/python '04 Experiments/models/train_daily.py' \
  --dataset "$dataset" --model gru --history 24 --seasonal-lags --epochs 10 --patience 3 --seed 42 \
  --output '04 Experiments/models/runs/gru-daily-example'
```

The daily model applies a shared network across the 24 known horizon contexts using the same consumption history at one origin. With `--seasonal-lags` and 24 hours of history, each horizon also receives its target-aligned reading from the previous day. It reports errors for hourly values and summed profiles, with a previous-day baseline for both. Use the same freeze and evaluation commands with the daily run directory to score complete held-out profiles.

After scoring all frozen models, pool the four official periods and calculate relative reductions from the previous-day baseline:

```bash
runs='04 Experiments/models/runs'
.venv/bin/python '04 Experiments/models/compare_results.py' \
  --hourly lstm "$runs/lstm_h24_verified/mape_metrics_v2.json" \
  --hourly gru "$runs/gru_h24_verified/mape_metrics_v2.json" \
  --hourly tcn "$runs/tcn_h24_verified/mape_metrics_v2.json" \
  --hourly transformer "$runs/transformer_h24_verified/mape_metrics_v2.json" \
  --daily lstm "$runs/mape_daily_lstm_aligned/heldout_mape/evaluation.json" \
  --daily gru "$runs/mape_daily_gru_aligned/heldout_mape/evaluation.json" \
  --daily tcn "$runs/mape_daily_tcn_aligned/heldout_mape/evaluation.json" \
  --daily transformer "$runs/mape_daily_transformer_aligned/heldout_mape/evaluation.json" \
  --output "$runs/mape-comparison-example.json"
```

The comparison verifies model names, dataset manifests, target hashes, baseline scores and counts before combining periods. MAE and MAPE are weighted by their eligible counts. RMSE is combined from the squared period RMSEs. A negative reduction means that the model error exceeded the baseline error.

## Generate local notebook starters

```bash
.venv/bin/python '04 Experiments/models/generate_notebooks.py' --dataset "$dataset"
```

The generator executes each notebook from an empty working directory and exports its HTML in `models/notebooks/`. Use `--output-dir '04 Experiments/models/notebooks/mape_10'` to choose a new directory when the default already exists. Notebooks embed fitting and validation arrays and require the pinned libraries. The HTML removes the embedded data cell. Keep the generated notebooks and exports local and check course requirements before sharing them.

## Scope

See [the starter comparison](../Starter%20results.md) for results and limitations. UTC split boundaries and the optional 00:00 UTC daily schedule remain provisional. The project owner confirmed the completed meter reading is available at the origin. Historical weather publication and revision timing remain unverified.

## Reproduce feature and weather comparisons

The [starter results](../Starter%20results.md) compare eight hourly GRU configurations on identical validation targets. The following example starts at the repository root. It uses a fresh output name for each run. The weather build and both weather experiments are conditional on an unverified one-hour observation delay.

```bash
data='04 Experiments/shared_features/prepared_no_weather/group_rebuild'
weather='04 Experiments/shared_features/generated_weather_1h/group_rebuild'
runs='04 Experiments/models/runs'
build='04 Experiments/shared_features/build_datasets.py'
train='04 Experiments/models/train.py'

.venv/bin/python "$build" --output-dir "$data"
.venv/bin/python "$build" --output-dir "$weather" --weather-latency-hours 1

.venv/bin/python "$train" --dataset "$data" --model gru --history 24 --seed 42 --no-calendar --output "$runs/repro-history"
.venv/bin/python "$train" --dataset "$data" --model gru --history 24 --seed 42 --no-calendar --rolling --output "$runs/repro-history-rolling"
.venv/bin/python "$train" --dataset "$data" --model gru --history 24 --seed 42 --output "$runs/repro-calendar"
.venv/bin/python "$train" --dataset "$data" --model gru --history 24 --seed 42 --rolling --output "$runs/repro-calendar-rolling"
.venv/bin/python "$train" --dataset "$data" --model gru --history 72 --seed 42 --output "$runs/repro-calendar-72"
.venv/bin/python "$train" --dataset "$data" --model gru --history 168 --seed 42 --output "$runs/repro-calendar-168"
.venv/bin/python "$train" --dataset "$weather" --model gru --history 24 --seed 42 --weather-temperature --output "$runs/repro-temperature"
.venv/bin/python "$train" --dataset "$weather" --model gru --history 24 --seed 42 --weather-temperature --weather-temperature-rolling --weather-humidity --weather-wind --output "$runs/repro-weather"

.venv/bin/python '04 Experiments/models/summarize_validation.py' \
  --experiment history "$data" "$runs/repro-history" \
  --experiment history_rolling "$data" "$runs/repro-history-rolling" \
  --experiment calendar "$data" "$runs/repro-calendar" \
  --experiment calendar_rolling "$data" "$runs/repro-calendar-rolling" \
  --experiment calendar_72 "$data" "$runs/repro-calendar-72" \
  --experiment calendar_168 "$data" "$runs/repro-calendar-168" \
  --experiment temperature "$weather" "$runs/repro-temperature" \
  --experiment weather "$weather" "$runs/repro-weather" \
  --output "$runs/repro-validation-summary.json"
```

The trainer defaults to 10 epochs, patience 3, a batch size of 128 and learning rate 0.001. The summary verifies the fit and validation hashes, recorded training code and environment, feature schema and reproduced validation MAE. It refuses comparisons when target keys or labels differ. The ignored JSON includes MAE, RMSE, local-hour and seasonal slices, and missing selected weather inputs before fit-only imputation. Each run is deterministic under the tested environment, but the results are still from one seed. Record additional seeds and any changes to settings when repeating the comparison.

The official test and evaluation results for the shared starters are already published in [Starter results](../Starter%20results.md). Record whether those results influence any later student-model choices.
