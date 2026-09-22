from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from ssl_probe.commands.train import build_run_name, parse_args, resolve_smile_root


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
    assert not hasattr(args, "conversion")


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
