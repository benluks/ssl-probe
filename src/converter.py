import torch
from quick_convert.components.decoders import KnnVCHifiGanDecoder
from quick_convert.components.ssl import WavLMContentEncoder
from quick_convert.data import load_dataset
from scipy.stats import pearsonr
from tqdm import tqdm

from .opensmile_factory import init_opensmile

if __name__ == "__main__":
    wavlm = WavLMContentEncoder(layer=5)
    hifigan = KnnVCHifiGanDecoder.from_pretrained("original")

    dataset = load_dataset(
        "librispeech", root="/Users/ben/librispeech/LibriSpeech", splits=["test-other"]
    )
    smile = init_opensmile()

    for sample in tqdm(dataset):
        sample_path = sample.path

        orig_smile = smile.process_file(sample_path)

        with torch.inference_mode():
            waveform = hifigan(wavlm.encode_file(sample_path).values)
        resynth_smile = smile.process_signal(
            waveform.detach().cpu().numpy(),
            int(hifigan.sample_rate),
        )

        # Resynthesis may differ slightly in duration
        n_frames = min(len(orig_smile), len(resynth_smile))

        orig = orig_smile.iloc[:n_frames]
        resynth = resynth_smile.iloc[:n_frames]

        r, p = pearsonr(
            orig.to_numpy(),
            resynth.to_numpy(),
        )

        for feature, score in zip(orig.columns, r):
            print(f"{feature:<40} {score:>8.3f}")
