from __future__ import annotations

import math
import random
import warnings

import torch
import torchaudio
from quick_convert.components.feature_extractors.content import ContentFeatureExtractor
from quick_convert.components.ssl import WavLMContentEncoder
from quick_convert.data import AudioBatch, AudioSample, load_dataset
from quick_convert.data.resources import ResourceCollection, ResourceRef
from torch.utils.data import DataLoader, IterableDataset, Sampler

from targets.target import FrameTarget

from .opensmile_factory import init_opensmile

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

            # Approximate padded WavLM cost:
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


class FrameDataset(IterableDataset):
    def __init__(
        self,
        target: FrameTarget,
        root="/Users/ben/librispeech/LibriSpeech",
        splits=("test-other",),
        layer=5,
        context_size=1,
        inference_frame_budget=10_000,
        frame_batch_size=512,
        shuffle=True,
    ):
        super().__init__()

        self.target = target
        self.context_size = context_size
        self.context_radius = context_size // 2

        self.base_dataset = load_dataset(
            "librispeech",
            root=root,
            splits=list(splits),
            load=["audio"],
        )

        self.content_encoder = ContentFeatureExtractor(WavLMContentEncoder(layer=layer))

        self.smile = init_opensmile()

        self.inference_frame_budget = inference_frame_budget
        self.frame_batch_size = frame_batch_size
        self.shuffle = shuffle

        self.frame_lengths = [
            self._estimate_wavlm_frames(sample.path)
            for sample in self.base_dataset.rows
        ]

    @staticmethod
    def _estimate_wavlm_frames(path) -> int:
        info = torchaudio.info(path)

        # Convert source duration to equivalent 16 kHz sample count.
        n_16k = round(info.num_frames * 16_000 / info.sample_rate)

        # WavLM frontend stride = 320 samples = 20 ms.
        return math.ceil(n_16k / 320)

    @torch.inference_mode()
    def extract_frames(
        self,
        samples: list[AudioSample],
    ) -> list[AudioSample]:
        audio_batch = self.base_dataset.collate_fn(samples)

        content = self.content_encoder.extract_batch(audio_batch)

        frame_samples = []

        for batch_idx, sample in enumerate(samples):
            if hasattr(content, "values") and hasattr(content, "lengths"):
                n_content = int(content.lengths[batch_idx])
                utterance_content = content.values[batch_idx, :n_content]
            else:
                utterance_content = content[batch_idx]

            waveform = sample.waveform

            if waveform.ndim == 2 and waveform.shape[0] == 1:
                waveform = waveform.squeeze(0)

            smile_features = self.smile.process_signal(
                waveform.detach().cpu().numpy(),
                int(sample.sample_rate),
            )

            raw_target = torch.tensor(
                smile_features[self.target.source].to_numpy(),
                dtype=torch.float32,
            )

            # openSMILE = 10 ms hop, WavLM = 20 ms hop.
            raw_target = raw_target[::2]

            n = min(
                utterance_content.shape[0],
                raw_target.shape[0],
            )

            utterance_content = utterance_content[:n]
            raw_target = raw_target[:n]

            target_values, valid = self.target.apply(raw_target)

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
            collate_fn=self.base_dataset.collate_fn,
            num_workers=0,
        )

        residual: list[AudioSample] = []

        for utterance_batch in utterance_loader:
            samples = list(utterance_batch)

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
