# Shared model inputs

`preprocessing.py` provides a NumPy-based loader and fitting-only preprocessing for the prepared splits. Install NumPy in the same Python environment as the model. Run the example from this directory after rebuilding the shared datasets.

```python
from preprocessing import FeatureConfig, Preprocessor, load_split

root = '../shared_features/prepared_no_weather/verified_rebuild'
config = FeatureConfig(history_hours=168, calendar=True, rolling=False)

fit = load_split(root, 'hourly', 'fit', config)
scaler = Preprocessor.fit(fit)
x_fit, calendar_fit, y_fit = scaler.transform(fit)

validation = load_split(root, 'hourly', 'validation', config)
x_val, calendar_val, y_val = scaler.transform(validation)
scaler.save(root + '/hourly_scaler.json')  # use an ignored local output directory
```

Set `root` to the directory containing the verified dataset manifest. The example names the independently verified rebuild available in the current workspace; choose the actual output directory in another checkout. Do not commit household-derived scalers, datasets or prediction tables. `load_split` verifies the manifest's CSV hashes and counts, split boundaries and feature-label keys by default. It rejects incomplete or cross-boundary daily profiles.

History arrays have shape `(examples, history_hours, 1)`, with readings ordered oldest to newest. Hourly context has shape `(examples, context_features)` and labels have shape `(examples,)`. For daily data, use `horizon='daily'`; history has the same shape, context is `(profiles, 24, context_features)`, and labels are `(profiles, 24)` in horizon order. `origins` and `targets` are metadata and are not included in the predictor arrays.

`FeatureConfig` offers 24, 72 or 168-hour histories, a calendar group, rolling consumption summaries, and for the daily task optional target-aligned previous-day and previous-week readings. A shorter history excludes rolling windows and seasonal lags that require more history. The daily horizon is always included as a known covariate. Weather flags default to off. To select past weather, use a manifest that explicitly records the assumed publication delay and keep the results conditional. The daily loader requires observed-weather values to be identical across the 24 horizons of a profile.

`Preprocessor.fit` accepts only the `fit` split. It learns one mean and standard deviation per selected input column, imputes missing inputs with the corresponding fit mean, and fits a target mean and standard deviation. Constant columns use a scale of one. `transform` reuses these parameters and returns scaled labels; call `inverse_labels` before reporting MAE or RMSE in the original source units. The saved JSON records feature order and scaling parameters; `Preprocessor.load` restores it. A different fold requires its own fitting and transformation.

From this directory, run `python3 -m unittest -v test_preprocessing` to check fit-only preprocessing, input ordering, hourly/daily shapes, serialization and tamper detection. All CSVs and model outputs contain household information and should remain local.
