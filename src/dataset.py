from __future__ import annotations

import math

import opensmile
import torch
from quick_convert.components.feature_extractors.content import ContentFeatureExtractor
from quick_convert.components.ssl import WavLMContentEncoder
from quick_convert.data import AudioBatch, AudioSample, load_dataset
from quick_convert.data.resources import ResourceCollection, ResourceRef
from torch.utils.data import DataLoader, IterableDataset


class FrameDataset(IterableDataset):
    PITCH_KEY = "F0semitoneFrom27.5Hz_sma3nz"

    def __init__(
        self,
        root="/Users/ben/librispeech/LibriSpeech",
        splits=("test-other",),
        layer=5,
        utterance_batch_size=8,
        frame_batch_size=512,
        shuffle=True,
    ):
        super().__init__()

        self.base_dataset = load_dataset(
            "librispeech",
            root=root,
            splits=list(splits),
            load=["audio"],
        )

        self.content_encoder = ContentFeatureExtractor(WavLMContentEncoder(layer=layer))

        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        )

        self.utterance_batch_size = utterance_batch_size
        self.frame_batch_size = frame_batch_size
        self.shuffle = shuffle

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

            f0_semitones = torch.tensor(
                smile_features[self.PITCH_KEY].to_numpy(),
                dtype=torch.float32,
            )

            # openSMILE: 10 ms hop
            # WavLM:     20 ms hop
            f0_semitones = f0_semitones[::2]

            n = min(
                utterance_content.shape[0],
                f0_semitones.shape[0],
            )

            utterance_content = utterance_content[:n]
            f0_semitones = f0_semitones[:n]

            voiced = f0_semitones > 0

            utterance_content = utterance_content[voiced]
            f0_semitones = f0_semitones[voiced]

            log_f0 = math.log(27.5) + f0_semitones * (math.log(2.0) / 12.0)

            original_frame_indices = torch.arange(n)[voiced]

            for encoding, pitch, frame_idx in zip(
                utterance_content,
                log_f0,
                original_frame_indices,
                strict=True,
            ):
                frame_samples.append(
                    AudioSample(
                        utt_id=f"{sample.utt_id}:{int(frame_idx):06d}",
                        path=sample.path,
                        split=sample.split,
                        resources=ResourceCollection.from_refs(
                            [
                                ResourceRef(
                                    name="content",
                                    kind="torch_tensor",
                                    value=encoding,
                                ),
                                ResourceRef(
                                    name="pitch",
                                    kind="torch_tensor",
                                    value=pitch.unsqueeze(0),
                                ),
                            ]
                        ),
                    )
                )

        return frame_samples

    def __iter__(self):
        utterance_loader = self.base_dataset.make_dataloader(
            batch_size=self.utterance_batch_size,
            shuffle=self.shuffle,
            num_workers=0,
        )

        residual: list[AudioSample] = []

        for utterance_batch in utterance_loader:
            # Reconstruct the individual utterance samples expected by our
            # extraction routine.
            samples = list(utterance_batch)

            frames = self.extract_frames(samples)

            pool = residual + frames

            # Shuffle whole frame samples, preserving content/pitch pairing.
            if self.shuffle:
                order = torch.randperm(len(pool)).tolist()
                pool = [pool[i] for i in order]

            n_full_batches = len(pool) // self.frame_batch_size

            for i in range(n_full_batches):
                start = i * self.frame_batch_size
                end = start + self.frame_batch_size

                yield AudioBatch.from_samples(pool[start:end])

            residual = pool[n_full_batches * self.frame_batch_size :]

        # Only the final batch of the epoch may be undersized.
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
