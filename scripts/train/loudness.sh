uv run src.train \
    --target loudness \
    --root /tmp/u036742/librispeech/LibriSpeech \
    --smile-root features/opensmile/knnvc_original_librispeech/ \
    --frame-batch-size 4096 \
    --context-size 1 \
    --hidden-dim 512 \
    --val-check-interval 2000
