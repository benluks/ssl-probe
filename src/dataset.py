from __future__ import annotations

import json
import random
import warnings
from os import PathLike
from pathlib import Path

import torch
import torchaudio
from quick_convert.components.ssl import WavLMContentEncoder
from quick_convert.data import AudioBatch, AudioSample, load_dataset
from quick_convert.data.resources import (
    ResourceCollection,
    ResourceRef,
    TemplateResourceProvider,
)
from torch.utils.data import DataLoader, IterableDataset, Sampler

from .targets import FrameTarget, SmileParquetStore

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
        target: FrameTarget,
        smile_root: PathLike,
        root="/Users/ben/librispeech/LibriSpeech",
        dataset_name: str | None = "librispeech",
        splits=("test-other",),
        layer=5,
        context_size=1,
        inference_frame_budget=10_000,
        frame_batch_size=512,
        shuffle=True,
        audio_root=None,
        speaker_stats: PathLike | None = None,
        spk_id_template: str = "{path.parent.parent.stem}",
    ):
        super().__init__()

        self.target = target
        if self.target.transform_kwargs is not None and speaker_stats is None:
            raise ValueError(
                f"Target {self.target.name!r} requires speaker statistics."
            )

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
            additional_resource_providers=additional_resource_providers,
        )

        self.stores = {}

        if speaker_stats is not None:
            stats = json.loads(Path(speaker_stats).read_text())

            self.stores["speaker_stats"] = {
                spk_id: values["mean_log_f0"]
                for spk_id, values in stats["speakers"].items()
            }

        self.content_encoder = WavLMContentEncoder(layer=layer)

        self.smile_features = SmileParquetStore(
            smile_root=smile_root,
            feature_column=self.target.source,
            audio_root=audio_root,
        )

        self.inference_frame_budget = inference_frame_budget
        self.frame_batch_size = frame_batch_size
        self.shuffle = shuffle

        self.frame_lengths = [
            self._estimate_ssl_frames(sample.path) for sample in self.base_dataset.rows
        ]

    def _estimate_ssl_frames(self, path) -> int:
        info = torchaudio.info(path)

        # Convert source duration to the encoder's required sample rate.
        input_length = round(
            info.num_frames * self.content_encoder.sample_rate / info.sample_rate
        )
        lengths = torch.tensor([input_length], dtype=torch.long)

        return int(self.content_encoder.output_lengths(lengths)[0])

    @torch.inference_mode()
    def extract_frames(
        self,
        samples: list[AudioSample],
    ) -> list[AudioSample]:
        audio_batch = self.base_dataset.collate_fn(samples)

        content = self.content_encoder(audio_batch)

        frame_samples = []

        for batch_idx, sample in enumerate(samples):
            n_content = int(content.lengths[batch_idx])
            utterance_content = content.values[batch_idx, :n_content]

            raw_target = self.smile_features[sample.utt_id]

            # openSMILE = 10 ms hop, WavLM = 20 ms hop.
            raw_target = raw_target[::2]

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


if __name__ == "__main__":
    fd = FrameDataset(
        utterance_batch_size=8,
        frame_batch_size=512,
    )

    dl = fd.make_dataloader()

    batch = next(iter(dl))

    print(len(batch))
