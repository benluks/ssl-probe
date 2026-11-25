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


@pytest.mark.parametrize("policy", ["raise", "skip"])
def test_failures_are_recorded_and_resume_does_not_decode_cached_audio(
    tmp_path, monkeypatch, policy
):
    from types import SimpleNamespace

    import torch
    from quick_convert.data import MetadataSample

    from ssl_probe.commands import precompute_opensmile as command

    class Dataset:
        rows = [
            MetadataSample(
                utt_id=name, path=tmp_path / f"{name}.mp4", split="test", start=0.0, end=1.0
            )
            for name in ("good", "broken", "later")
        ]
        broken = True

        def __getitem__(self, index):
            accessed.append(index)
            if index == 1 and self.broken:
                raise ValueError("Audio interval exceeds the source recording.")
            return SimpleNamespace(waveform=torch.zeros(16000), sample_rate=16000)

    dataset = Dataset()
    accessed = []
    monkeypatch.setattr(command, "load_dataset", lambda **kwargs: dataset)
    monkeypatch.setattr(
        command,
        "init_opensmile",
        lambda **kwargs: SimpleNamespace(
            process_signal=lambda waveform, sr: pd.DataFrame({"feature": [1.0]})
        ),
    )
    arguments = [
        "opensmile",
        "--root",
        str(tmp_path),
        "--splits",
        "test",
        "--out-folder",
        "store",
        "--out-root",
        str(tmp_path / "features"),
        "--on-error",
        policy,
    ]
    monkeypatch.setattr(sys, "argv", arguments)
    with pytest.raises(RuntimeError if policy == "raise" else SystemExit):
        main()

    store = tmp_path / "features" / "store"
    record = json.loads((store / "_errors.jsonl").read_text())
    assert record["utt_id"] == "broken"
    assert record["path"] == str(tmp_path / "broken.mp4")
    assert (record["start"], record["end"], record["stage"]) == (0.0, 1.0, "load")
    assert "ValueError" in record["traceback"]
    assert not (store / "test" / "broken.parquet").exists()
    assert accessed == ([0, 1] if policy == "raise" else [0, 1, 2])
    summary = json.loads((store / "_extraction_summary.json").read_text())
    assert summary["failed"] == 1
    assert summary["status"] == ("interrupted" if policy == "raise" else "completed_with_errors")

    dataset.broken = False
    accessed.clear()
    main()
    assert accessed == ([1, 2] if policy == "raise" else [1])
    assert len(list((store / "test").glob("*.parquet"))) == 3
    assert json.loads((store / "_extraction_summary.json").read_text())["status"] == "complete"
    assert len((store / "_errors.jsonl").read_text().splitlines()) == 1


def test_skip_policy_does_not_hide_storage_errors_or_leave_partial_cache(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import torch
    from quick_convert.data import MetadataSample

    from ssl_probe.commands import precompute_opensmile as command

    class Dataset:
        rows = [MetadataSample(utt_id="a", path=tmp_path / "a.wav", split="test")]

        def __getitem__(self, index):
            return SimpleNamespace(waveform=torch.zeros(16000), sample_rate=16000)

    monkeypatch.setattr(command, "load_dataset", lambda **kwargs: Dataset())
    monkeypatch.setattr(
        command,
        "init_opensmile",
        lambda **kwargs: SimpleNamespace(
            process_signal=lambda waveform, sr: pd.DataFrame({"feature": [1.0]})
        ),
    )

    def fail_write(self, path, **kwargs):
        path.write_bytes(b"partial parquet")
        raise OSError("storage full")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_write)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "opensmile",
            "--root",
            str(tmp_path),
            "--splits",
            "test",
            "--out-folder",
            "store",
            "--out-root",
            str(tmp_path / "features"),
            "--on-error",
            "skip",
        ],
    )
    with pytest.raises(OSError, match="storage full"):
        main()
    store = tmp_path / "features" / "store"
    assert not list((store / "test").glob("*.parquet"))
    assert not list((store / "test").glob("*.tmp"))
    assert not (store / "_errors.jsonl").exists()
