#!/usr/bin/env bash

set -e

features=(
    jitter
    shimmer
    spectral_flux
    hnr
    alpha_ratio
    hammarberg
)

for feature_name in "${features[@]}"; do
    echo "========================================"
    echo "Training target: $feature_name"
    echo "========================================"

    uv run -m src.train \
        --target "$feature_name" \
        --root /tmp/u036742/librispeech/LibriSpeech \
        --smile-root features/opensmile/librispeech/ \
        --frame-batch-size 1024 \
        --context-size 3 \
        --hidden-dim 512 \
        --val-check-interval 2000 \
        --conversion original
done