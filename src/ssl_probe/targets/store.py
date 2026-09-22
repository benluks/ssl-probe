import json
from functools import cached_property
from os import PathLike
from pathlib import Path

import pandas as pd
import torch

from ..opensmile import OPENSMILE_LLD_FRAME_HZ, init_opensmile


class SmileParquetStore:
    def __init__(
        self,
        smile_root: PathLike | None = None,
        audio_root: PathLike | None = None,
        feature_column: str | None = None,
    ):
        self.smile_root = Path(smile_root)
        self.feature_column = feature_column
        self.audio_root = Path(audio_root) if audio_root is not None else None

    @cached_property
    def metadata(self) -> dict:
        path = self.smile_root / "_metadata.json"
        return json.loads(path.read_text()) if path.exists() else {}

    @property
    def frame_hz(self) -> float:
        frame_hz = self.metadata.get("frame_hz", OPENSMILE_LLD_FRAME_HZ)
        if not isinstance(frame_hz, (int, float)) or frame_hz <= 0:
            raise ValueError(f"Invalid openSMILE frame_hz metadata: {frame_hz!r}.")
        return float(frame_hz)

    @cached_property
    def smile(self):
        return init_opensmile()

    def _find_audio_path(self, utt_id: str) -> Path | None:
        if self.audio_root is None:
            return None

        matches = list(self.audio_root.glob(f"{utt_id}.*"))

        if not matches:
            return None

        if len(matches) > 1:
            raise RuntimeError(f"Found multiple audio files for {utt_id!r}: {matches}")

        return matches[0]

    def __getitem__(self, utt_id: str) -> torch.Tensor:
        feature_path = self.smile_root / f"{utt_id}.parquet"

        if feature_path.exists():
            df = pd.read_parquet(
                feature_path,
                columns=[self.feature_column],
            )

            return torch.as_tensor(
                df[self.feature_column].to_numpy(),
                dtype=torch.float32,
            )

        audio_path = self._find_audio_path(utt_id)

        if audio_path is not None:
            features = self.smile.process_file(audio_path)

            return torch.as_tensor(
                features[self.feature_column].to_numpy(),
                dtype=torch.float32,
            )

        raise FileNotFoundError(
            f"No precomputed features found at {feature_path}, "
            f"and no source audio was found for utterance {utt_id!r}."
        )
