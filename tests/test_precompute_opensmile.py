from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from ssl_probe.commands.precompute_opensmile import main, parse_args


@pytest.mark.parametrize(
    "args",
    [
        ["--root", "/audio"],
        ["--dataset-config", "data.yaml"],
        ["--dataset-config", "data.yaml", "--out-folder", "x", "--splits", "train"],
        ["--root", "/audio", "--splits", "train", "--limit", "0"],
        ["--root", "/audio", "--splits", "train", "--limit", "-1"],
    ],
)
def test_invalid_sources_and_limits(args):
    with pytest.raises(SystemExit):
        parse_args(args)


def test_yaml_extraction_limit_unsplit_metadata_and_resume(tmp_path, monkeypatch):
    audio = tmp_path / "audio"
    audio.mkdir()
    for index in range(3):
        signal = 0.1 * np.sin(2 * np.pi * 150 * np.arange(16000) / 16000)
        sf.write(audio / f"sample-{index}.wav", signal, 16000)

    config = tmp_path / "private.yaml"
    config.write_text(
        f'audio_root: "{audio}"\n'
        "dataset:\n"
        "  _target_: quick_convert.data.BaseDataset\n"
        "  root: ${audio_root}\n"
        '  utt_id_template: "{path.stem}"\n'
        "  target_sr: 16000\n"
        "  load: [audio]\n"
        "unused_fold:\n"
        "  fold: ???\n"
    )
    output = tmp_path / "features"
    arguments = [
        "opensmile",
        "--dataset-config",
        str(config),
        "--out-folder",
        "indexed",
        "--out-root",
        str(output),
    ]
    monkeypatch.setattr(sys, "argv", [*arguments, "--limit", "2"])
    main()

    store = output / "indexed"
    first = sorted(store.glob("*.parquet"))
    assert len(first) == 2
    assert json.loads((store / "_metadata.json").read_text())["split"] is None
    frame = pd.read_parquet(first[0])
    assert {"start", "end", "jitterLocal_sma3nz", "shimmerLocaldB_sma3nz"} <= set(frame)
    assert len(frame) > 0
    timestamps = {path: path.stat().st_mtime_ns for path in first}

    monkeypatch.setattr(sys, "argv", arguments)
    main()
    assert len(list(store.glob("*.parquet"))) == 3
    assert all(path.stat().st_mtime_ns == stamp for path, stamp in timestamps.items())
