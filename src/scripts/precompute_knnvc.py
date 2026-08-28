import argparse
from pathlib import Path

import torch
import torchaudio
from quick_convert.components.decoders import KnnVCHifiGanDecoder
from quick_convert.components.ssl import WavLMContentEncoder
from quick_convert.data import load_dataset
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", required=True)
    parser.add_argument("--splits", nargs="+", required=True)
    parser.add_argument(
        "--out-root",
        default=Path("features"),
    )
    parser.add_argument("--out-folder", default=None)
    parser.add_argument(
        "--dataset",
        default="librispeech",
    )
    parser.add_argument(
        "--hifigan-ckpt", choices=["original", "prematched"], default="original"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    wavlm = WavLMContentEncoder(layer=6)
    hifigan = KnnVCHifiGanDecoder.from_pretrained(args.hifigan_ckpt)

    out_folder = args.out_folder or f"knnvc_{args.hifigan_ckpt}"
    out_dir = Path(args.out_root) / out_folder / args.dataset

    dataset = load_dataset(args.dataset, root=args.root, splits=args.splits)

    for sample in tqdm(dataset):
        with torch.inference_mode():
            waveform = hifigan(wavlm.encode_file(sample.path).values)

        outf = Path(out_dir) / sample.split / f"{sample.utt_id}.flac"
        outf.parent.mkdir(exist_ok=True, parents=True)

        torchaudio.save(outf, waveform.cpu(), sample_rate=hifigan.sample_rate)
