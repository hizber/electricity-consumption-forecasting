"""Build local, portable notebooks from verified fit and validation arrays."""

import argparse
import base64
import io
import json
import os
import platform
import sys
import tempfile
from pathlib import Path

import nbformat as nbf
import numpy as np
import torch
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from traitlets.config import Config

from train import FeatureConfig, MODELS, Preprocessor, digest, load_split


HERE = Path(__file__).resolve().parent
DISCLOSURE = """**__Course materials__**

- Keras Functional API (4b)
- Neural-network Architecture (5)
- Dataset Splitting (7)
- Activation Functions (9)
- Training of Neural Networks (10)
- Loss Functions and Metrics (12)

I used ChatGPT to help writing the code, compare network configurations and explain the concepts. I used it as a learning assistant."""

TRAIN = """
import random
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(2)
model = Architecture(context_size=data['fit_context'].shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()

def batches(split, shuffle=False):
    '''Create batches of scaled history, context and labels.'''
    arrays = [torch.from_numpy(data[f'{split}_{name}']) for name in ('history', 'context', 'labels')]
    return DataLoader(TensorDataset(*arrays), batch_size=128, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(SEED) if shuffle else None)

training = batches('fit', shuffle=True)
validation = batches('validation')
best_mae = float('inf')
stale = 0

def mape(errors, actual):
    '''Return MAPE and the count of zero actuals excluded from it.'''
    nonzero = actual != 0
    value = float(np.abs(errors[nonzero] / actual[nonzero]).mean() * 100) if nonzero.any() else None
    return value, int((~nonzero).sum())

for epoch in range(1, EPOCHS + 1):
    model.train()
    for history, context, labels in training:
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(history, context), labels)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training loss')
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        scaled = np.concatenate([model(history, context).numpy() for history, context, _ in validation])
    errors = scaled * data['label_scale'] + data['label_mean'] - data['actual']
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mape_value, excluded = mape(errors, data['actual'])
    mape_text = f'{mape_value:.2f}%' if mape_value is not None else 'undefined'
    print(f'Epoch {epoch}: validation MAE {mae:.4f}, RMSE {rmse:.4f}, '
          f'MAPE {mape_text} ({excluded} zero actuals excluded)')
    if mae < best_mae:
        best_mae, best_rmse, best_mape, best_epoch, stale = mae, rmse, mape_text, epoch, 0
    else:
        stale += 1
        if stale >= 3:
            break
print(f'Best epoch {best_epoch}: MAE {best_mae:.4f}, RMSE {best_rmse:.4f}, MAPE {best_mape}')
baseline = data['baseline'] - data['actual']
baseline_mape, excluded = mape(baseline, data['actual'])
baseline_text = f'{baseline_mape:.2f}%' if baseline_mape is not None else 'undefined'
print(f'Previous-day baseline: MAE {np.abs(baseline).mean():.4f}, RMSE {np.sqrt(np.mean(baseline ** 2)):.4f}, '
      f'MAPE {baseline_text} ({excluded} zero actuals excluded)')
"""


def encoded_data(dataset):
    """Validate both development splits and encode transformed arrays."""
    config = FeatureConfig(history_hours=24)
    fit = load_split(dataset, 'hourly', 'fit', config)
    validation = load_split(dataset, 'hourly', 'validation', config)
    scaler = Preprocessor.fit(fit)
    arrays = {'actual': validation.labels, 'baseline': validation.history[:, -24, 0],
              'label_mean': scaler.label_mean, 'label_scale': scaler.label_scale}
    for split in (fit, validation):
        for name, array in zip(('history', 'context', 'labels'), scaler.transform(split), strict=True):
            arrays[f'{split.split}_{name}'] = array.astype(np.float32)
    with io.BytesIO() as handle:
        np.savez_compressed(handle, **arrays)
        return base64.b64encode(handle.getvalue()).decode('ascii')


def notebook(name, payload, dataset, epochs):
    """Assemble one next-hour notebook with local-only embedded arrays."""
    architecture = MODELS[name]
    source_path = HERE / ('recurrent.py' if name in ('gru', 'lstm') else 'temporal.py')
    source = source_path.read_text(encoding='utf-8')
    manifest = json.loads((dataset / 'manifest.json').read_text())
    cells = [
        nbf.v4.new_markdown_cell(
            f'# {name.upper()} next-hour forecasting starter\n\n'
            'Private course artifact containing derived household data. Keep it local. '
            f'Tested with Python {platform.python_version()}, NumPy {np.__version__} and PyTorch {torch.__version__}. '
            'The embedded arrays use 24 chronological '
            'load readings and known Helsinki calendar context. Preprocessing was fitted on '
            'the fitting split only. Validation selects the best epoch.\n\n'
            f'Source SHA-256: `{manifest["input_sha256"]}`. '
            f'Dataset manifest SHA-256: `{digest(dataset / "manifest.json")}`. '
            'Use this as a starting point and document your own changes and AI use. '
            'The final disclosure block is the course template. Its Keras reference '
            'does not describe this PyTorch implementation.'),
        nbf.v4.new_code_cell(
            'import base64\nimport io\nimport numpy as np\n'
            f'encoded = {payload!r}\n'
            'data = np.load(io.BytesIO(base64.b64decode(encoded)), allow_pickle=False)\n'
            "print(f\"Fit: {len(data['fit_labels']):,} hours. Validation: {len(data['actual']):,} hours.\")",
            metadata={'tags': ['private-data']}),
        nbf.v4.new_code_cell(source + f'\nArchitecture = {architecture.__name__}'),
        nbf.v4.new_code_cell(f'EPOCHS = {epochs}\n' + TRAIN),
        nbf.v4.new_markdown_cell(DISCLOSURE),
    ]
    return nbf.v4.new_notebook(cells=cells)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--model', choices=MODELS, action='append')
    parser.add_argument('--output-dir', type=Path, help='New output directory inside models/notebooks')
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    notebooks = (HERE / 'notebooks').resolve()
    output = args.output_dir.resolve() if args.output_dir else notebooks / f'epochs_{args.epochs}'
    if not output.is_relative_to(notebooks) or output == notebooks:
        parser.error('output directory must be inside models/notebooks')
    output.mkdir(parents=True, exist_ok=True)
    payload = encoded_data(args.dataset)
    for name in args.model or MODELS:
        target = output / f'{name}_starter'
        if target.with_suffix('.ipynb').exists() or target.with_suffix('.html').exists():
            raise FileExistsError(target)
        document = notebook(name, payload, args.dataset, args.epochs)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            kernel = root / 'jupyter/kernels/forecast'
            kernel.mkdir(parents=True)
            (kernel / 'kernel.json').write_text(json.dumps({
                'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
                'display_name': 'Forecast Python', 'language': 'python',
            }))
            env = {**os.environ, 'JUPYTER_PATH': str(root / 'jupyter'),
                   'JUPYTER_RUNTIME_DIR': str(root / 'runtime')}
            specs = KernelSpecManager(kernel_dirs=[str(kernel.parent)])
            manager = KernelManager(kernel_name='forecast', kernel_spec_manager=specs)
            NotebookClient(document, km=manager, timeout=900).execute(cwd=temporary, env=env)
        document.metadata.kernelspec = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
        nbf.validate(document)
        config = Config()
        config.TagRemovePreprocessor.remove_input_tags = {'private-data'}
        html, _ = HTMLExporter(config=config).from_notebook_node(document)
        nbf.write(document, target.with_suffix('.ipynb'))
        target.with_suffix('.html').write_text(html, encoding='utf-8')
        print(f'Executed and exported {name}', flush=True)


if __name__ == '__main__':
    main()
