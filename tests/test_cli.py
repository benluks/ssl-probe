from __future__ import annotations

import subprocess
import sys

import pytest


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
