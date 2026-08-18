uv run src.train \
    --target semitone \
    --root /tmp/u036742/librispeech/LibriSpeech \
    --smile-root features/opensmile/knnvc_original_librispeech/
    --frame-batch-size 1024 \
    --context-size 3 \
    --hidden-dim 512 \
    --val-check-interval 2000
