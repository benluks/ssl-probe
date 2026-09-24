#!/usr/bin/env bash
set -euo pipefail

# Run one of the 56 encoder/target combinations. Task IDs 0-27 use
# W2V-BERT; task IDs 28-55 use S3Tokenizer.

task_id="${1:-}"
if [[ ! "${task_id}" =~ ^[0-9]+$ ]] || (( task_id < 0 || task_id >= 56 )); then
    echo "usage: $0 TASK_ID" >&2
    echo "TASK_ID must be an integer from 0 through 55" >&2
    exit 2
fi

regression_targets=(
    logf0
    loudness
    alpha_ratio
    hammarberg
    slope_0_500
    slope_500_1500
    spectral_flux
    mfcc1
    mfcc2
    mfcc3
    mfcc4
    jitter
    shimmer
    hnr
    h1_h2
    h1_a3
    f1
    f2
    f3
    f1_bandwidth
    f2_bandwidth
    f3_bandwidth
    f1_amplitude
    f2_amplitude
    f3_amplitude
)
classification_targets=(semitone voiced f1_bin)
targets=("${regression_targets[@]}" "${classification_targets[@]}")

target_count="${#targets[@]}"
encoder_index=$((task_id / target_count))
target_index=$((task_id % target_count))
target="${targets[target_index]}"

case "${encoder_index}" in
    0)
        encoder="w2vbert"
        encoder_label="w2vbert"
        encoder_args=()
        ;;
    1)
        encoder="s3tokenizer"
        encoder_label="s3tokenizer"
        encoder_args=(--encoder-kwargs '{"representation":"encoder","layer":-1}')
        ;;
esac

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="${PROJECT_ROOT:-$(cd -- "${script_dir}/.." && pwd)}"
probe_bin="${SSL_PROBE_BIN:-${project_root}/.venv/bin/ssl-probe}"
librispeech_root="${LIBRISPEECH_ROOT:-/cfs/collections/librispeech/LibriSpeech}"
smile_root="${SMILE_ROOT:-${project_root}/features/opensmile/librispeech}"
out_root="${OUT_ROOT:-${project_root}/outputs/sweeps/librispeech_weighted_sum}"
out_dir="${out_root}/${encoder_label}/${target}"
completion_marker="${out_dir}/.complete"

if [[ -f "${completion_marker}" && "${FORCE:-0}" != "1" ]]; then
    printf 'skip task %d/55: encoder=%s target=%s (already complete)\n' \
        "${task_id}" "${encoder}" "${target}"
    exit 0
fi

if [[ ! -x "${probe_bin}" ]]; then
    echo "error: ssl-probe executable not found at ${probe_bin}" >&2
    echo "set SSL_PROBE_BIN if the environment lives elsewhere" >&2
    exit 1
fi

normalization_args=()
if (( target_index < ${#regression_targets[@]} )); then
    normalization_args=(--target-normalization standardize)
fi

command=(
    "${probe_bin}" train
    --root "${librispeech_root}"
    --dataset librispeech
    --train-split train-clean-100
    --val-split dev-clean
    --smile-root "${smile_root}"
    --encoder "${encoder}"
    "${encoder_args[@]}"
    --layer-fusion weighted-sum
    --layer-log-interval "${LAYER_LOG_INTERVAL:-1000}"
    --target "${target}"
    "${normalization_args[@]}"
    --context-size "${CONTEXT_SIZE:-3}"
    --hidden-dim "${HIDDEN_DIM:-512}"
    --frame-batch-size "${FRAME_BATCH_SIZE:-256}"
    --lr "${LEARNING_RATE:-0.001}"
    --weight-decay "${WEIGHT_DECAY:-0.0001}"
    --warmup "${WARMUP:-0.05}"
    --max-steps "${MAX_STEPS:-10000}"
    --val-check-interval "${VAL_CHECK_INTERVAL:-2000}"
    --seed "${SEED:-115}"
    --experiment-label weighted-sum
    --out-dir "${out_dir}"
)

printf 'run task %d/55: encoder=%s target=%s\n' "${task_id}" "${encoder}" "${target}"
printf 'command:'
printf ' %q' "${command[@]}"
printf '\n'

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    exit 0
fi

mkdir -p "${out_dir}"
cd "${project_root}"
"${command[@]}"
touch "${completion_marker}"

