"""Inference-only evaluation of trusted local probe checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import types
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml
from quick_convert.data import AudioBatch

from ..dataset import FrameDataset
from ..encoders import build_content_encoder, build_layer_fusion
from ..probe import Probe
from ..targets import TARGETS, RegressionTask, TargetStandardization


class LegacyCheckpointUnpickler(pickle.Unpickler):
    """Translate the known pre-package target modules without global aliases."""

    def find_class(self, module, name):
        module = {
            "src.targets.target": "ssl_probe.targets.target",
            "src.targets.task": "ssl_probe.targets.task",
        }.get(module, module)
        return super().find_class(module, name)


def load_checkpoint(path):
    # torch.load expects a pickle module, including load/loads and Unpickler.
    compatibility_pickle = types.ModuleType("ssl_probe_checkpoint_pickle")
    compatibility_pickle.__dict__.update(vars(pickle))
    compatibility_pickle.Unpickler = LegacyCheckpointUnpickler
    return torch.load(
        path, map_location="cpu", weights_only=False, pickle_module=compatibility_pickle
    )


def read_config(checkpoint_path, explicit=None):
    candidates = (
        [Path(explicit)]
        if explicit
        else [
            parent / name
            for parent in list(Path(checkpoint_path).parents)[:4]
            for name in ("config.json", "config.yaml", "hparams.yaml")
        ]
    )
    for path in candidates:
        if path.is_file():
            payload = yaml.safe_load(path.read_text())
            # W&B's exported config wraps each entry in a value field.
            config = {
                key: value["value"] if isinstance(value, dict) and "value" in value else value
                for key, value in payload.items()
            }
            if all(key in config for key in ("encoder", "target", "context_size")):
                if (
                    config.get("target_normalization") == "standardize"
                    and "target_standardization" not in config
                ):
                    for parent in list(Path(checkpoint_path).parents)[:4]:
                        stats = parent / "target_standardization.json"
                        if stats.is_file():
                            config["target_standardization"] = json.loads(stats.read_text())
                            break
                return config
    raise ValueError(
        "No complete run configuration found; pass --config with the saved training arguments (JSON/YAML or W&B config export)."
    )


def restore_models(checkpoint, config, device="cpu"):
    saved_train_encoder = checkpoint.get("hyper_parameters", {}).get("train_encoder")
    if saved_train_encoder is True and config.get("train_encoder") is False:
        raise ValueError("Configuration conflicts with the checkpoint's fine-tuned encoder flag.")
    target = TARGETS[config["target"]]
    if not isinstance(target.task, RegressionTask):
        raise ValueError("Evaluation currently supports regression targets only.")
    if config.get("target_normalization", "none") == "standardize":
        stats = config.get("target_standardization")
        if stats is None:
            raise ValueError("Standardized checkpoint requires saved target_standardization.")
        target = replace(
            target, task=target.task.with_standardization(TargetStandardization(**stats))
        )
    context = config["context_size"]
    if context < 1 or context % 2 != 1:
        raise ValueError("context_size must be a positive odd number.")
    encoder = build_content_encoder(config["encoder"], config.get("encoder_kwargs", {}))
    fusion = build_layer_fusion(
        config.get("layer_fusion", "none"), encoder, config.get("num_layers")
    )
    probe = Probe(
        encoder.feature_dim * context,
        target.task.output_dim,
        config.get("hidden_dim", []),
        config.get("nonlinearity", "gelu"),
        fusion,
    )
    state = {
        key.replace("_orig_mod.", ""): value for key, value in checkpoint["state_dict"].items()
    }
    # Before packaging, the Lightning module's probe was a bare Sequential.
    legacy_probe = "probe.0.weight" in state
    if legacy_probe:
        state = {
            "probe.network." + key.removeprefix("probe.")
            if key.startswith("probe.")
            else key: value
            for key, value in state.items()
        }
    probe.load_state_dict(
        {
            key.removeprefix("probe."): value
            for key, value in state.items()
            if key.startswith("probe.")
        },
        strict=True,
    )
    encoder_state = {
        key.removeprefix("content_encoder."): value
        for key, value in state.items()
        if key.startswith("content_encoder.")
    }
    if encoder_state:
        encoder.load_state_dict(encoder_state, strict=True)
    elif config.get("train_encoder", False) or saved_train_encoder is True:
        raise ValueError("Fine-tuned checkpoint is missing encoder weights.")
    encoder.to(device).eval().requires_grad_(False)
    encoder.device = torch.device(device)
    probe.to(device).eval().requires_grad_(False)
    return encoder, probe, target


def regression_metrics(actual, predicted):
    actual, predicted = (
        np.asarray(actual, dtype=np.float64),
        np.asarray(predicted, dtype=np.float64),
    )
    if not len(actual):
        return {
            "count": 0,
            **dict.fromkeys(("r2", "pearson", "rmse", "mae", "ccc", "mean_baseline_rmse")),
        }
    error = predicted - actual
    variance = np.sum((actual - actual.mean()) ** 2)
    covariance = np.mean((actual - actual.mean()) * (predicted - predicted.mean()))
    denominator = actual.var() + predicted.var() + (actual.mean() - predicted.mean()) ** 2
    pearson_denominator = np.sqrt(actual.var() * predicted.var())
    return {
        "count": len(actual),
        "r2": float(1 - np.sum(error**2) / variance) if variance > 0 and len(actual) > 1 else None,
        "pearson": float(covariance / pearson_denominator) if pearson_denominator > 0 else None,
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "ccc": float(2 * covariance / denominator) if denominator > 0 else None,
        "mean_baseline_rmse": float(np.sqrt(actual.var())),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument("--root")
    data.add_argument("--manifest", type=Path)
    parser.add_argument("--dataset", default="clac")
    parser.add_argument("--split", default="max_phonation")
    parser.add_argument("--smile-root", type=Path)
    parser.add_argument("--speaker-stats", type=Path)
    parser.add_argument(
        "--spk-id-template",
        default=None,
        help="Speaker grouping template; verify for your dataset layout.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--frame-batch-size", type=int, default=512)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.frame_batch_size < 1:
        parser.error("--frame-batch-size must be positive")
    return args


@torch.inference_mode()
def main():
    args = parse_args()
    config = read_config(args.checkpoint, args.config)
    # Lightning checkpoints contain pickled metadata; use only trusted local files.
    checkpoint = load_checkpoint(args.checkpoint)
    encoder, probe, target = restore_models(checkpoint, config, args.device)
    smile_root = args.smile_root or Path("features/opensmile") / args.dataset
    args.spk_id_template = args.spk_id_template or (
        "{path.stem}" if args.dataset == "clac" else "{path.parent.parent.stem}"
    )
    dataset = FrameDataset(
        encoder,
        target,
        smile_root if args.manifest or args.dataset == "clac" else smile_root / args.split,
        root=args.root,
        dataset_name=args.dataset,
        splits=[args.split],
        manifest_path=args.manifest,
        context_size=config["context_size"],
        speaker_stats=args.speaker_stats,
        spk_id_template=args.spk_id_template,
        preserve_layers=probe.feature_transform is not None,
        shuffle=False,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    groups = {}
    actual, predicted = [], []
    skipped = []
    with (args.out_dir / "predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["utterance", "speaker", "frame", "time_seconds", "actual", "predicted"])
        # One utterance at a time avoids padding-dependent encoder predictions.
        for index in range(len(dataset.base_dataset)):
            sample = dataset.base_dataset[index]
            frames = dataset.extract_frames([sample])
            if not frames:
                skipped.append(sample.utt_id)
                continue
            speaker = args.spk_id_template.format(path=Path(sample.path), utt_id=sample.utt_id)
            group = groups.setdefault(speaker, ([], []))
            for start in range(0, len(frames), args.frame_batch_size):
                chunk = frames[start : start + args.frame_batch_size]
                batch = AudioBatch.from_samples(chunk)
                x = batch.resources["content"].values.to(args.device)
                y = batch.resources[target.name].values[:, 0].to(args.device)
                result = target.task.compute(probe(x), y)
                truth = result.metric_target.flatten().cpu().tolist()
                estimate = result.prediction.flatten().cpu().tolist()
                if not np.isfinite(truth).all() or not np.isfinite(estimate).all():
                    raise ValueError(f"Non-finite evaluation values in {sample.utt_id}")
                actual.extend(truth)
                predicted.extend(estimate)
                group[0].extend(truth)
                group[1].extend(estimate)
                for frame, value, prediction in zip(chunk, truth, estimate, strict=True):
                    frame_index = int(frame.utt_id.rsplit(":", 1)[1])
                    writer.writerow(
                        [
                            sample.utt_id,
                            speaker,
                            frame_index,
                            frame_index / dataset.content_frame_hz,
                            value,
                            prediction,
                        ]
                    )
    if not actual:
        raise ValueError("No valid evaluation frames found.")
    speakers = {speaker: regression_metrics(*values) for speaker, values in groups.items()}
    macro = {}
    for key in ("r2", "pearson", "rmse", "mae", "ccc"):
        values = [scores[key] for scores in speakers.values() if scores[key] is not None]
        macro[key] = float(np.mean(values)) if values else None
    payload = {
        "checkpoint": str(args.checkpoint),
        "target": target.name,
        "scale": "transformed target scale, with training standardization undone",
        "config": config,
        "pooled": regression_metrics(actual, predicted),
        "per_speaker": speakers,
        "speaker_macro": macro,
        "skipped_utterances": skipped,
        "baseline": "Evaluation-set mean (descriptive oracle constant baseline).",
    }
    (args.out_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n"
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.hexbin(actual, predicted, gridsize=60, mincnt=1, bins="log")
    low, high = min(min(actual), min(predicted)), max(max(actual), max(predicted))
    ax.plot([low, high], [low, high], "k--", linewidth=1)
    ax.set(
        xlabel=f"Measured {target.name}",
        ylabel=f"Predicted {target.name}",
        title=f"{args.dataset} / {args.split}",
    )
    fig.tight_layout()
    fig.savefig(args.out_dir / "scatter.png", dpi=160)
    plt.close(fig)
    print(json.dumps(payload["pooled"], indent=2))


if __name__ == "__main__":
    main()
