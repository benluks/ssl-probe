from __future__ import annotations

import json
import math
import random
import warnings
from os import PathLike
from pathlib import Path

import torch
import torchaudio
from quick_convert.components.ssl import ContentEncoder
from quick_convert.data import AudioBatch, AudioSample
from quick_convert.data.loading import load_dataset
from quick_convert.data.resources import (
    ResourceCollection,
    ResourceRef,
    TemplateResourceProvider,
)
from torch.utils.data import DataLoader, IterableDataset, Sampler
from tqdm import tqdm

from .targets import FrameTarget, SmileParquetStore, TargetStandardization

warnings.filterwarnings(
    "ignore",
    message=r"Support for mismatched key_padding_mask and attn_mask is deprecated.*",
    category=UserWarning,
    module=r"torch\.nn\.functional",
)


class FrameBudgetBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        frame_lengths: list[int],
        max_frames: int,
        shuffle: bool = True,
    ):
        self.frame_lengths = frame_lengths
        self.max_frames = max_frames
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.frame_lengths)))

        if self.shuffle:
            random.shuffle(indices)

        batch = []
        max_len = 0

        for idx in indices:
            length = self.frame_lengths[idx]

            candidate_max = max(max_len, length)
            candidate_size = len(batch) + 1

            # Approximate padded encoder cost:
            # B * max(T_i)
            padded_frames = candidate_size * candidate_max

            if batch and padded_frames > self.max_frames:
                yield batch
                batch = [idx]
                max_len = length
            else:
                batch.append(idx)
                max_len = candidate_max

        if batch:
            yield batch


RESOURCE_PROVIDERS = {
    "spk_id": lambda spk_id_template: TemplateResourceProvider(
        name="spk_id", template=spk_id_template, kind="text"
    ),
}


class FrameDataset(IterableDataset):
    def __init__(
        self,
        content_encoder: ContentEncoder,
        target: FrameTarget,
        smile_root: PathLike,
        root="/Users/ben/librispeech/LibriSpeech",
        dataset_name: str | None = "librispeech",
        splits=("test-other",),
        context_size=1,
        inference_frame_budget=10_000,
        frame_batch_size=512,
        shuffle=True,
        audio_root=None,
        speaker_stats: PathLike | None = None,
        spk_id_template: str = "{path.parent.parent.stem}",
        preserve_layers: bool = False,
    ):
        super().__init__()

        self.content_encoder = content_encoder.eval()
        self.content_frame_hz = self._encoder_frame_hz()
        self.preserve_layers = preserve_layers
        self.target = target
        if self.target.transform_kwargs is not None and speaker_stats is None:
            raise ValueError(f"Target {self.target.name!r} requires speaker statistics.")

        self.context_size = context_size
        self.context_radius = context_size // 2

        additional_resource_providers = [
            RESOURCE_PROVIDERS[name](spk_id_template) for name in self.target.resources
        ]

        self.base_dataset = load_dataset(
            dataset_name,
            root=root,
            splits=list(splits),
            load=["audio"],
            target_sr=self._encoder_sample_rate(),
            additional_resource_providers=additional_resource_providers,
        )

        self.stores = {}

        if speaker_stats is not None:
            stats = json.loads(Path(speaker_stats).read_text())

            self.stores["speaker_stats"] = {
                spk_id: values["mean_log_f0"] for spk_id, values in stats["speakers"].items()
            }

        self.smile_features = SmileParquetStore(
            smile_root=smile_root,
            feature_column=self.target.source,
            audio_root=audio_root,
        )

        self.inference_frame_budget = inference_frame_budget
        self.frame_batch_size = frame_batch_size
        self.shuffle = shuffle

        self.frame_lengths = [
            self._estimate_reference_frames(sample.path) for sample in self.base_dataset.rows
        ]

    @property
    def feature_dim(self) -> int:
        return self.content_encoder.feature_dim

    def compute_target_standardization(self) -> TargetStandardization:
        """Compute transformed-target moments from valid training frames only."""
        total = 0.0
        total_squared = 0.0
        count = 0

        for sample in tqdm(
            self.base_dataset.rows,
            desc=f"target stats ({self.target.name})",
        ):
            raw_target = self.smile_features[sample.utt_id]
            kwargs = (
                self.target.transform_kwargs(sample, self.stores)
                if self.target.transform_kwargs is not None
                else {}
            )
            values, valid = self.target.apply(raw_target, **kwargs)
            values = values[valid].double()

            total += values.sum().item()
            total_squared += values.square().sum().item()
            count += values.numel()

        if count == 0:
            raise ValueError(f"Target {self.target.name!r} has no valid training frames.")

        mean = total / count
        variance = max(total_squared / count - mean**2, 0.0)
        std = math.sqrt(variance)

        if std == 0:
            raise ValueError(f"Target {self.target.name!r} has zero training variance.")

        return TargetStandardization(mean=mean, std=std, count=count)

    def _encoder_sample_rate(self) -> int:
        sample_rate = getattr(self.content_encoder, "sample_rate", None)
        if not isinstance(sample_rate, int) or sample_rate <= 0:
            raise TypeError(
                f"{type(self.content_encoder).__name__} must expose a positive integer "
                "`sample_rate` for batched audio loading."
            )
        return sample_rate

    def _encoder_frame_hz(self) -> float:
        frame_hz = self.content_encoder.frame_hz
        if not isinstance(frame_hz, (int, float)) or frame_hz <= 0:
            raise TypeError(
                f"{type(self.content_encoder).__name__} must expose a positive `frame_hz` "
                "for frame-level probing."
            )
        return float(frame_hz)

    def _estimate_reference_frames(self, path) -> int:
        """Estimate encoder frames from audio duration for batch budgeting."""
        info = torchaudio.info(path)
        return max(1, round(info.num_frames * self.content_frame_hz / info.sample_rate))

    @staticmethod
    def _align_target_frames(
        raw_target: torch.Tensor,
        output_frames: int,
        *,
        target_frame_hz: float,
        content_frame_hz: float,
    ) -> torch.Tensor:
        """Sample target frames at zero-origin content-frame times."""
        if output_frames < 0:
            raise ValueError("output_frames cannot be negative.")
        if target_frame_hz <= 0 or content_frame_hz <= 0:
            raise ValueError("Target and content frame rates must be positive.")
        if output_frames == 0:
            return raw_target[:0]
        if raw_target.shape[0] == 0:
            raise ValueError("Cannot align an empty target sequence to non-empty encoder output.")

        content_times = torch.arange(output_frames, dtype=torch.float64) / content_frame_hz
        indices = torch.floor(content_times * target_frame_hz).to(dtype=torch.long)
        indices = indices[indices < raw_target.shape[0]]
        return raw_target[indices]

    @staticmethod
    def _normalize_content_frames(
        values: torch.Tensor,
        *,
        preserve_layers: bool = False,
    ) -> torch.Tensor:
        """Normalize one utterance for single-layer probing or layer fusion."""
        if preserve_layers:
            if values.ndim != 3:
                raise ValueError(
                    "Layer fusion requires content with shape (frames, layers, features), "
                    f"but got {tuple(values.shape)}."
                )
            return values

        while values.ndim > 2 and values.shape[1] == 1:
            values = values.squeeze(1)
        if values.ndim != 2:
            raise ValueError(
                "Content encoders must return one feature vector per frame for probing. "
                f"Got an utterance tensor with shape {tuple(values.shape)}; select a single layer "
                "or enable layer fusion."
            )
        return values

    @torch.inference_mode()
    def extract_frames(
        self,
        samples: list[AudioSample],
    ) -> list[AudioSample]:
        audio_batch = self.base_dataset.collate_fn(samples)

        content = self.content_encoder(audio_batch)
        if content.frame_hz is None:
            raise ValueError(
                f"{type(self.content_encoder).__name__} returned features without a frame timebase."
            )
        if not math.isclose(content.frame_hz, self.content_frame_hz):
            raise ValueError(
                f"{type(self.content_encoder).__name__} declares frame_hz={self.content_frame_hz}, "
                f"but returned frame_hz={content.frame_hz}."
            )

        frame_samples = []

        for batch_idx, sample in enumerate(samples):
            n_content = int(content.lengths[batch_idx])
            utterance_content = content.values[batch_idx, :n_content]
            utterance_content = self._normalize_content_frames(
                utterance_content,
                preserve_layers=self.preserve_layers,
            )
            if utterance_content.shape[-1] != self.feature_dim:
                raise ValueError(
                    f"Encoder declares feature_dim={self.feature_dim}, but returned "
                    f"{utterance_content.shape[-1]} features per frame."
                )

            raw_target = self.smile_features[sample.utt_id]
            raw_target = self._align_target_frames(
                raw_target,
                n_content,
                target_frame_hz=self.smile_features.frame_hz,
                content_frame_hz=content.frame_hz,
            )

            n = min(
                utterance_content.shape[0],
                raw_target.shape[0],
            )

            utterance_content = utterance_content[:n]
            raw_target = raw_target[:n]

            kwargs = (
                self.target.transform_kwargs(sample, self.stores)
                if self.target.transform_kwargs is not None
                else {}
            )

            target_values, valid = self.target.apply(
                raw_target,
                **kwargs,
            )

            radius = self.context_radius

            # Only frames that have enough context on both sides can act as centers.
            for frame_idx in range(radius, n - radius):
                # The target decides whether this CENTER frame is a valid example.
                #
                # For log-F0, for example, unvoiced center frames are skipped.
                # Crucially, unvoiced neighboring frames are still retained as context.
                if not valid[frame_idx]:
                    continue

                context = utterance_content[frame_idx - radius : frame_idx + radius + 1]

                target_value = target_values[frame_idx]

                frame_samples.append(
                    AudioSample(
                        utt_id=f"{sample.utt_id}:{frame_idx:06d}",
                        path=sample.path,
                        split=sample.split,
                        resources=ResourceCollection.from_refs(
                            [
                                ResourceRef(
                                    name="content",
                                    kind="torch_tensor",
                                    value=context,
                                ),
                                ResourceRef(
                                    name=self.target.name,
                                    kind="torch_tensor",
                                    value=target_value.reshape(1),
                                ),
                            ]
                        ),
                    )
                )

        return frame_samples

    def __iter__(self):
        batch_sampler = FrameBudgetBatchSampler(
            frame_lengths=self.frame_lengths,
            max_frames=self.inference_frame_budget,
            shuffle=self.shuffle,
        )

        utterance_loader = DataLoader(
            self.base_dataset,
            batch_sampler=batch_sampler,
            collate_fn=lambda samples: samples,
            num_workers=0,
        )

        residual: list[AudioSample] = []

        for samples in utterance_loader:
            frames = self.extract_frames(samples)

            pool = residual + frames

            if self.shuffle:
                order = torch.randperm(len(pool)).tolist()
                pool = [pool[i] for i in order]

            n_full_batches = len(pool) // self.frame_batch_size

            for i in range(n_full_batches):
                start = i * self.frame_batch_size
                end = start + self.frame_batch_size

                yield AudioBatch.from_samples(pool[start:end])

            residual = pool[n_full_batches * self.frame_batch_size :]

        if residual:
            yield AudioBatch.from_samples(residual)

    def make_dataloader(self):
        return DataLoader(
            self,
            batch_size=None,
            num_workers=0,
        )
