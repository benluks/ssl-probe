uv run -m src.train \
    --target voiced \
    --root /tmp/u036742/librispeech/LibriSpeech \
    --smile-root features/opensmile/knnvc_original_librispeech/ \
    --frame-batch-size 1024 \
    --context-size 1 \
    --val-check-interval 2000 \
    # no hidden dim. Testing for linear recoverability
