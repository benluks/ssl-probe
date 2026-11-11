import json
from types import SimpleNamespace

import pytest

from ssl_probe.commands import precompute_opensmile as command


@pytest.mark.parametrize(
    ("requested", "resolved"),
    [(None, ["train", "dev"]), (None, None), (["train"], ["train"])],
)
def test_metadata_uses_quick_convert_splits(tmp_path, monkeypatch, requested, resolved):
    arguments = [
        "--root",
        str(tmp_path / "audio"),
        "--out-folder",
        "features",
        "--out-root",
        str(tmp_path / "output"),
    ]
    if requested is not None:
        arguments += ["--splits", *requested]
    args = command.parse_args(arguments)
    monkeypatch.setattr(command, "parse_args", lambda: args)

    class EmptyDataset:
        splits = resolved

        def __iter__(self):
            return iter(())

        def __len__(self):
            return 0

    calls = []

    def load_dataset(**kwargs):
        calls.append(kwargs)
        return EmptyDataset()

    monkeypatch.setattr(command, "load_dataset", load_dataset)
    monkeypatch.setattr(command, "init_opensmile", lambda **kwargs: SimpleNamespace())
    command.main()
    assert calls[0]["splits"] == requested
    for split in resolved if resolved is not None else [None]:
        directory = tmp_path / "output/features"
        if split is not None:
            directory /= split
        metadata = json.loads((directory / "_metadata.json").read_text())
        assert metadata["split"] == split
        assert metadata["jitter_min_periods"] == 2


def test_explicit_missing_split_error_is_preserved(tmp_path, monkeypatch):
    args = command.parse_args(["--root", str(tmp_path), "--splits", "missing"])
    monkeypatch.setattr(command, "parse_args", lambda: args)

    def load_dataset(**kwargs):
        raise FileNotFoundError("Split directory does not exist")

    monkeypatch.setattr(command, "load_dataset", load_dataset)
    with pytest.raises(FileNotFoundError, match="Split directory"):
        command.main()
