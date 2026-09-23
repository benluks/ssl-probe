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
ssl-probe train \
  --root /path/to/LibriSpeech \
  --train-split train-clean-100 \
  --val-split dev-clean \
  --encoder wavlm \
  --encoder-kwargs '{"layer": 6}' \
  --target logf0 \
  --target-normalization standardize \
  --smile-root features/opensmile/librispeech
```

The openSMILE feature store defaults to `features/opensmile/<dataset>`. Pass
`--smile-root` when the features live elsewhere. Run names describe the target, dataset,
encoder, probe nonlinearity, and context size without assuming a conversion system. Use
`--experiment-label` to distinguish an explicit data or experiment variant:

```bash
ssl-probe train \
  --root /path/to/converted/LibriSpeech \
  --smile-root features/opensmile/knnvc-original-librispeech \
  --experiment-label knnvc-original \
  --encoder w2vbert \
  --target logf0
```

The probe itself is a plain PyTorch module. Quick Convert's Lightning training adapter and pipeline infrastructure are used only for training orchestration.

For datasets that need a custom train/validation partition, pass a pair of Quick Convert flat
manifests instead of `--root` and named splits. The manifests must contain `utt_id`, `path`, and
`split` columns. For CLAC, Quick Convert can build a flat manifest and split speakers disjointly:

```bash
export QUICK_CONVERT_CLAC_ROOT=/path/to/CLAC-Dataset

quick-convert build_flat_manifest_clac
python -m quick_convert.cli.split_manifest \
  --input outputs/clac/manifest.csv \
  --train-output outputs/clac/speaker_split/train.csv \
  --valid-output outputs/clac/speaker_split/val.csv \
  --strategy group-disjoint \
  --group-col spkid \
  --valid-fraction 0.1 \
  --seed 115

ssl-probe train \
  --dataset clac \
  --train-manifest outputs/clac/speaker_split/train.csv \
  --val-manifest outputs/clac/speaker_split/val.csv \
  --smile-root features/opensmile/clac \
  --encoder wavlm \
  --encoder-kwargs '{"layer": 6}' \
  --target logf0 \
  --target-normalization standardize \
  --context-size 3 \
  --hidden-dim 512 \
  --frame-batch-size 1024 \
  --max-steps 10000 \
  --val-check-interval 2000 \
  --spk-id-template '{path.stem}'
```

CLAC manifest IDs already have the form `elicitation/speaker`, so manifest-backed training uses
the common openSMILE root directly rather than appending one named split. Omit
`--target-normalization standardize` for classification targets such as the bin targets.

The target registry covers all 25 eGeMAPSv02 low-level descriptor columns. Derived pitch and
formant targets such as `smn_logf0`, `voiced`, `semitone`, and `f1_bin` remain available in
addition to the continuous source targets. For regression, `--target-normalization standardize`
computes mean and standard deviation from valid transformed training frames, trains against
z-scored values, and reports metrics on the unstandardized transformed scale. The fitted moments
are written to `target_standardization.json` in the run directory. Pearson correlation and
concordance correlation are reported alongside MAE, MSE, RMSE, and R².

The representation encoder is not tied to WavLM. Quick Convert owns the built-in aliases
`dac`, `emotion2vec`, `pros2vec`, `s3tokenizer`, `w2vbert`, and `wavlm`. Any Quick Convert
`ContentEncoder` subclass can also be selected by dotted class path:

```bash
ssl-probe train \
  --root /path/to/LibriSpeech \
  --encoder quick_convert.components.ssl.W2VBertContentEncoder \
  --encoder-kwargs '{"layer": 12}' \
  --target logf0
```

The encoder must expose its required `sample_rate` and `frame_hz`. Probe targets are sampled
at encoder-frame times using the encoder and openSMILE timebases; they are not stretched
proportionally to the encoder sequence length. Existing openSMILE outputs without frame-rate
metadata are treated as the standard 100 Hz low-level-descriptor stream.

For encoders that return multiple layer vectors per frame, either select a layer through the
encoder's own constructor arguments or train Quick Convert's weighted-sum fusion jointly with the
probe:

```bash
ssl-probe train \
  --root /path/to/LibriSpeech \
  --encoder s3tokenizer \
  --encoder-kwargs '{"representation": "encoder", "layer": -1}' \
  --layer-fusion weighted-sum \
  --target logf0
```

The layer count is inferred for Quick Convert's WavLM, W2V-BERT, and S3Tokenizer encoders. For an
arbitrary dotted-path encoder whose model metadata does not expose the count, pass
`--num-layers`. Fusion stays inside the plain `Probe` module, so its softmax-normalized layer
weights are learned with the probe rather than frozen during feature extraction.

During training, the normalized weights are logged as per-layer scalar histories and as a bar
chart through Quick Convert's media-logger abstraction. The default logging interval is 1,000
optimizer steps and can be changed with `--layer-log-interval`. At the end of training,
`layer_weights.json` is written to the prepared run directory with both the learned logits and
normalized weights. The weights also remain part of the probe checkpoint.

### Precompute openSMILE features

```bash
ssl-probe precompute opensmile \
  --root /path/to/LibriSpeech \
  --dataset librispeech \
  --splits train-clean-100 dev-clean
```

### CLAC KNN-VC resynthesis correlations

CLAC uses split-qualified utterance IDs such as `picnic/1234`. To resynthesize
WavLM layer 6 with the KNN-VC HiFiGAN checkpoint trained on original features,
then compare frame-level openSMILE features:

```bash
export QUICK_CONVERT_CLAC_ROOT=/path/to/clac
SPLITS=(picnic rainbow max_phonation smr)

ssl-probe precompute knnvc \
  --root "$QUICK_CONVERT_CLAC_ROOT" \
  --dataset clac \
  --splits "${SPLITS[@]}" \
  --hifigan-ckpt original \
  --out-root converted \
  --out-folder knnvc_l6_original

ssl-probe precompute opensmile \
  --root "$QUICK_CONVERT_CLAC_ROOT" \
  --dataset clac \
  --splits "${SPLITS[@]}" \
  --out-folder clac

ssl-probe precompute opensmile \
  --root converted/knnvc_l6_original/clac \
  --splits "${SPLITS[@]}" \
  --out-folder knnvc_l6_original_clac

for split in "${SPLITS[@]}"; do
  ssl-probe correlate "$split" \
    --feature-root features/opensmile \
    --original clac \
    --converted knnvc_l6_original_clac
done
```

The converted-audio extraction intentionally omits `--dataset clac`: KNN-VC
writes FLAC files, while the original CLAC dataset configuration selects WAV
inputs. The generic loader preserves the same split-relative output layout.

## Development

Formatting, linting, tests, and a wheel smoke test run in GitHub Actions. The corresponding local checks are:

```bash
uv run ruff format src/ssl_probe tests
uv run ruff check --select B,E4,E7,E9,F,I,UP src/ssl_probe tests
uv run pytest
```

### Model-backed smoke tests

Real-model tests are excluded from the fast suite because they install optional dependencies and
download checkpoints. To verify S3Tokenizer's real 12-layer output and trainable fusion path:

```bash
uv sync --group dev --group model-test
SSL_PROBE_RUN_MODEL_TESTS=1 uv run --no-sync pytest -m model tests/model
```

The S3Tokenizer ONNX checkpoint is downloaded to `~/.cache/s3tokenizer`. The same test can be
run on GitHub by manually dispatching the **CI** workflow; its `model-smoke` job caches that
checkpoint between runs.

The package uses a `src/` layout; importable code lives under `src/ssl_probe/`.

