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

The loader verifies split hashes, counts and feature-label alignment against the manifest. Preprocessing is fitted on the fitting split only. The training loop uses scaled-target MSE for gradient updates and selects the best epoch using validation MAE in the original source units, with early stopping controlled by patience. Validation RMSE and previous-day baseline MAE/RMSE are also recorded. Test and evaluation splits are not opened by this command. Freeze feature and model decisions using fitting and validation data before opening held-out data.

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

## Train a daily forecast

```bash
.venv/bin/python '04 Experiments/models/train_daily.py' \
  --dataset "$dataset" --model gru --history 24 --epochs 10 --patience 3 --seed 42 \
  --output '04 Experiments/models/runs/gru-daily-example'
```

The daily model applies a shared network across the 24 known horizon contexts using the same consumption history at one origin. It reports errors for hourly values and summed profiles, with a previous-day baseline for both. Use the same freeze and evaluation commands with the daily run directory to score complete held-out profiles.

## Generate local notebook starters

```bash
.venv/bin/python '04 Experiments/models/generate_notebooks.py' --dataset "$dataset"
```

The generator executes each notebook from an empty working directory and exports its HTML in `models/notebooks/`. Notebooks embed fitting and validation arrays and require the pinned libraries. The HTML removes the embedded data cell. Keep the generated notebooks and exports local and check course requirements before sharing them.

## Scope

See [the starter comparison](../Starter%20results.md) for results and limitations. UTC split boundaries and the optional 00:00 UTC daily schedule remain provisional. The project owner confirmed the completed meter reading is available at the origin. Historical weather publication and revision timing remain unverified.
