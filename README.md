# ssl-probe

Tools for probing speech representations for frame-level acoustic and paralinguistic attributes.

The project uses [Quick Convert](https://github.com/benluks/quick-convert) for datasets, speech representations, and training infrastructure while keeping probing targets, frame alignment, statistics, and experiment workflows local to this package.

## Setup

The project uses Python 3.11+ and uv:

```bash
uv sync --group dev
```

Quick Convert is resolved from its GitHub repository by default. This source is part of the
package metadata, so installing the built wheel resolves the same dependency. For local
development against a Quick Convert checkout, install that checkout editable in the environment
after syncing.

## CLI

The installed command is `ssl-probe`:

```bash
ssl-probe --help
ssl-probe train --help
ssl-probe correlate --help
ssl-probe precompute opensmile --help
ssl-probe precompute knnvc --help
ssl-probe precompute speaker-stats --help
```

### Train a probe

```bash
ssl-probe train \\
  --root /path/to/LibriSpeech \\
  --train-split train-clean-100 \\
  --val-split dev-clean \\
  --target logf0 \\
  --smile-root features/opensmile/librispeech
```

The probe itself is a plain PyTorch module. Quick Convert's Lightning training adapter and pipeline infrastructure are used only for training orchestration.

### Precompute openSMILE features

```bash
ssl-probe precompute opensmile \\
  --root /path/to/LibriSpeech \\
  --dataset librispeech \\
  --splits train-clean-100 dev-clean
```

## Development

Formatting, linting, tests, and a wheel smoke test run in GitHub Actions. The corresponding local checks are:

```bash
uv run ruff format src/ssl_probe tests
uv run ruff check --select B,E4,E7,E9,F,I,UP src/ssl_probe tests
uv run pytest
```

The package uses a `src/` layout; importable code lives under `src/ssl_probe/`.
