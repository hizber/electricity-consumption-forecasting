# Local data setup

The repository does not contain the course dataset or generated household records. Enrolled group members can obtain the electricity dataset from the course Moodle Datasets section, extract `dataset_mansion.csv`, and place it at `02 Data/Electric-consumption/dataset_mansion.csv` in this checkout. Do not commit or distribute the course dataset or derived household records without establishing permission and the applicable terms.

The dataset copied from Moodle on 6 November 2026 had SHA-256 `9e8334d93bc3f9f1391216f68f52eedb5c9ebd3c2a5856e97db0177cf76f70a0`. Check your local file with `shasum -a 256 '02 Data/Electric-consumption/dataset_mansion.csv'`. A different hash needs investigation before combining results; a hash proves identity with the recorded copy, not correctness or a licence to redistribute it.

From the repository root, Python 3.12 and `uv` can reproduce the project environment:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-lock.txt
```

From `04 Experiments/shared_features`, create a new output directory and run the checks:

```bash
../../.venv/bin/python build_datasets.py --output-dir prepared_no_weather
../../.venv/bin/python -m unittest -v test_build_datasets test_generate_features test_generate_daily_features
```

The script refuses an existing output directory. Inspect `prepared_no_weather/manifest.json` for row counts, source and script hashes and all output hashes. From inside `prepared_no_weather`, run `shasum -a 256 -c ../dataset-sha256.txt` to compare all 24 CSVs against the group reference. This passed on a fresh rebuild using Python 3.12.10 and the pinned environment. A failure leaves output for diagnosis. An earlier workspace build had output paths replaced while open. The writer now opens one chronological split at a time and checks for replacement. The original cause remains unconfirmed.

Check exclusions with `git check-ignore '02 Data/Electric-consumption/dataset_mansion.csv' '04 Experiments/shared_features/prepared_no_weather/manifest.json'`. Generated data, model artifacts and caches remain local. Repository access for other group members requires the owner to arrange access; access has not been granted by this setup guide.

The [forecast protocol](../04%20Experiments/Proposed%20forecast%20protocol.md) and [shared feature guide](../04%20Experiments/shared_features/README.md) explain timing, splits, validation and the unverified weather-availability assumption. For a different input file, provide `--input PATH` and record its provenance.
