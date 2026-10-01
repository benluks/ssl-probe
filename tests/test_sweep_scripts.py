import os
import subprocess
from pathlib import Path

import pytest

TASK = Path(__file__).resolve().parents[1] / "scripts/run_librispeech_weighted_sum_task.sh"


@pytest.mark.parametrize(
    ("encoder", "fusion", "kwargs"),
    [
        ("spear", "weighted-sum", '{"layer":null}'),
        ("emotion2vec", "weighted-sum", '{"layer":null,"granularity":"frame"}'),
        ("paseplus", "weighted-sum", '{"config_path":"a.cfg","checkpoint_path":"b.ckpt","layer":null}'),
    ],
)
def test_selected_sweep_constructor_and_fusion(tmp_path, encoder, fusion, kwargs):
    capture = tmp_path / "args"
    executable = tmp_path / "probe"
    executable.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$CAPTURE"\n')
    executable.chmod(0o755)
    env = {
        **os.environ,
        "SWEEP_ENCODER": encoder,
        "ENCODER_KWARGS": kwargs,
        "SSL_PROBE_BIN": str(executable),
        "OUT_ROOT": str(tmp_path / "outputs"),
        "CAPTURE": str(capture),
        "INFERENCE_FRAME_BUDGET": "300",
        "FORCE": "1",
        "DRY_RUN": "0",
    }
    subprocess.run([str(TASK), "27"], env=env, check=True, capture_output=True)
    args = capture.read_text().splitlines()
    assert args[args.index("--target") + 1] == "f1_bin"
    assert args[args.index("--encoder") + 1] == encoder
    assert args[args.index("--layer-fusion") + 1] == fusion
    assert args[args.index("--encoder-kwargs") + 1] == kwargs
    assert args[args.index("--inference-frame-budget") + 1] == "300"


def test_selected_sweep_rejects_legacy_task_id():
    result = subprocess.run(
        [str(TASK), "28"],
        env={**os.environ, "SWEEP_ENCODER": "spear"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "0 through 27" in result.stderr
