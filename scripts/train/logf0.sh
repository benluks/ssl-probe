uv run src.train \
    --target logf0 \
    --root /tmp/u036742/librispeech/LibriSpeech \
    --smile-root features/opensmile/knnvc_original_librispeech/ \
    --frame-batch-size 2048 \
    --context-size 3 \
    --hidden-dim 512 \
    --val-check-interval 2000
