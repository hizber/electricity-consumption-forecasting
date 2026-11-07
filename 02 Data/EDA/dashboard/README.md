# Electricity consumption EDA dashboard

Open [dashboard.html](dashboard.html) in a browser. It is self-contained and works offline. Charts use the fitting period from 1 January 2016 to 30 September 2019 UTC, keeping the later validation and official evaluation periods out of the analysis. Calendar views use Europe/Helsinki. Consumption units are unspecified in the source.

To rebuild, run the following commands from this folder with Python 3, pandas and NumPy installed. The first command uses the bundled runtime available in this workspace.

```bash
/Users/hizberhikmet/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 build_analysis.py
python3 build_dashboard.py
```

The analysis reads the original course CSV, writes `analysis.json` and records the input SHA-256. The dashboard builder embeds these statistics in `template.html` to produce the standalone HTML. Rebuild after changing the input or analysis. The fit period and UTC boundaries are fixed in `build_analysis.py`.

Lag correlations use complete pairs within the fitting period. Weather is shifted two hours before the target, corresponding to an assumed one-hour publication delay at the forecast origin. Historical publication and revision timing are unverified. Correlations are descriptive and should be assessed with feature-group ablations on chronological validation. No model is trained here and no correlation proves a feature improves predictions.

The feature matrix shows 12 selected representative variables with pairwise-complete sample counts. It does not cover every prepared predictor. The feature counts come from the prepared dataset manifests and exclude timestamps and labels. The EDA uses up to 32,856 raw fitting-period target hours, while the prepared fitting set has 32,690 rows after the 168-hour history requirement. Individual matrix counts vary with warmup and missingness.

The anomaly section uses each variable's fitting-period quartiles to flag values below Q1 − 1.5 × IQR or above Q3 + 1.5 × IQR. These are candidates for inspection, not confirmed errors. The global rule can flag valid seasonal peaks. Actual consumption and its hour-to-hour change are retrospective label checks, not model inputs. Weather flags inherit the unverified weather timing assumption. No observations are removed. The generated [anomalies.csv](anomalies.csv) records all flagged feature events, their bounds and forecast and source timestamps. A source reading can occur in multiple events. Keep the CSV beside the dashboard HTML for the download link to work.
