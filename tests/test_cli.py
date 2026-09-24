from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from ssl_probe.commands.train import (
    build_run_name,
    parse_args,
    resolve_smile_root,
    resolve_split_smile_root,
    write_json_artifact,
)


@pytest.mark.parametrize(
    "arguments",
    [
        ("train", "--help"),
        ("correlate", "--help"),
        ("precompute", "opensmile", "--help"),
        ("precompute", "knnvc", "--help"),
        ("precompute", "speaker-stats", "--help"),
    ],
)
def test_command_help(arguments: tuple[str, ...]) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "ssl_probe.cli", *arguments],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_train_defaults_are_encoder_neutral() -> None:
    args = parse_args(["--root", "/audio"])

    assert resolve_smile_root(args.dataset, args.smile_root) == Path(
        "features/opensmile/librispeech"
    )
    assert args.experiment_label is None
    assert args.layer_fusion == "none"
    assert args.num_layers is None
    assert args.layer_log_interval == 1000
    assert args.target_normalization == "none"
    assert not hasattr(args, "conversion")


def test_train_accepts_manifest_pair_without_root() -> None:
    args = parse_args(
        [
            "--train-manifest",
            "train.csv",
            "--val-manifest",
            "val.csv",
        ]
    )

    assert args.root is None
    assert args.train_manifest == Path("train.csv")
    assert args.val_manifest == Path("val.csv")


@pytest.mark.parametrize(
    "arguments",
    [
        (),
        ("--train-manifest", "train.csv"),
        ("--val-manifest", "val.csv"),
        (
            "--root",
            "/audio",
            "--train-manifest",
            "train.csv",
            "--val-manifest",
            "val.csv",
        ),
    ],
)
def test_train_rejects_ambiguous_or_incomplete_data_sources(
    arguments: tuple[str, ...],
) -> None:
    with pytest.raises(SystemExit):
        parse_args(list(arguments))


def test_manifest_feature_root_is_not_split_twice() -> None:
    smile_root = Path("features/opensmile/clac")

    assert resolve_split_smile_root(smile_root, "picnic", Path("train.csv")) == smile_root
    assert resolve_split_smile_root(smile_root, "dev-clean", None) == (smile_root / "dev-clean")


def test_train_run_name_has_no_implicit_conversion() -> None:
    run_name = build_run_name(
        target="logf0",
        dataset="librispeech",
        encoder_slug="s3-tokenizer",
        nonlinearity="gelu",
        context_size=3,
        experiment_label=None,
    )

    assert run_name == Path("logf0/librispeech_s3-tokenizer_gelu_c3")


def test_train_run_name_accepts_explicit_experiment_label() -> None:
    run_name = build_run_name(
        target="logf0",
        dataset="librispeech",
        encoder_slug="wavlm-l6",
        nonlinearity="gelu",
        context_size=1,
        experiment_label="knnvc-original",
    )

    assert run_name == Path("logf0/knnvc-original_librispeech_wavlm-l6_gelu_c1")


def test_write_json_artifact_creates_prepared_run_directory(tmp_path: Path) -> None:
    artifact_path = tmp_path / "ssl-probe" / "run-id" / "target_standardization.json"

    write_json_artifact(artifact_path, {"mean": 1.25, "std": 0.5})

    assert artifact_path.read_text() == '{\n  "mean": 1.25,\n  "std": 0.5\n}\n'
