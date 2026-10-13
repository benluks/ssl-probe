"""Bounded frozen-encoder parity and batching diagnostic, without W&B logging."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch
from quick_convert.data import AudioBatch
from quick_convert.data.loading import load_dataset
from quick_convert.training.lightning.optim import LinearWarmup, Optimization

from ..dataset import FrameBudgetBatchSampler, FrameDataset
from ..encoders import build_content_encoder
from ..probe import Probe
from ..targets import TARGETS, RegressionTask, TargetStandardization
from ..training import OnlineProbeTrainingModule, ProbeTrainingModule


class QuietLogging:
    def log(self, *args, **kwargs):
        pass

    def log_dict(self, *args, **kwargs):
        pass


class QuietFrameModule(QuietLogging, ProbeTrainingModule):
    pass


class QuietOnlineModule(QuietLogging, OnlineProbeTrainingModule):
    pass


def tensor_check(left, right, *, atol, rtol):
    left, right = left.detach().cpu(), right.detach().cpu()
    if left.shape != right.shape:
        return {"pass": False, "left_shape": list(left.shape), "right_shape": list(right.shape)}
    delta = (left.double() - right.double()).abs()
    return {
        "pass": bool(torch.allclose(left, right, atol=atol, rtol=rtol)),
        "max_abs": float(delta.max()) if delta.numel() else 0.0,
        "mean_abs": float(delta.mean()) if delta.numel() else 0.0,
    }


def compare_frames(left, right, target, *, atol, rtol):
    left = {frame.utt_id: frame for frame in left}
    right = {frame.utt_id: frame for frame in right}
    same_ids = left.keys() == right.keys()
    checks = {"features": [], "targets": []}
    for key in left.keys() & right.keys():
        for name, resource in (("features", "content"), ("targets", target.name)):
            checks[name].append(
                tensor_check(
                    left[key].resources[resource].value,
                    right[key].resources[resource].value,
                    atol=atol,
                    rtol=rtol,
                )
            )
    result = {"same_frame_ids": same_ids, "left_frames": len(left), "right_frames": len(right)}
    for name, values in checks.items():
        result[name] = {
            "pass": same_ids and all(value["pass"] for value in values),
            "max_abs": max((value.get("max_abs", 0.0) for value in values), default=0.0),
        }
    result["pass"] = same_ids and all(result[name]["pass"] for name in checks)
    return result


def frame_counts(frames, batch_size, seed):
    shuffled = list(frames)
    random.Random(seed).shuffle(shuffled)
    return [
        {
            "frames": len(chunk),
            "utterances": len({frame.utt_id.rsplit(":", 1)[0] for frame in chunk}),
        }
        for start in range(0, len(shuffled), batch_size)
        if (chunk := shuffled[start : start + batch_size])
    ]


def batching_impact(left, right, probe, *, atol, rtol):
    """Compare aligned features and predictions in bounded chunks."""
    left = {frame.utt_id: frame for frame in left}
    right = {frame.utt_id: frame for frame in right}
    if left.keys() != right.keys():
        raise ValueError("Cannot measure batching impact with different frame IDs")
    device = next(probe.parameters()).device
    feature_errors, prediction_errors = [], []
    affected = 0
    keys = sorted(left)
    with torch.no_grad():
        for start in range(0, len(keys), 256):
            chunk = keys[start : start + 256]
            a, b = [
                torch.stack([frames[key].resources["content"].value for key in chunk]).to(device)
                for frames in (left, right)
            ]
            feature_errors.append((a.double() - b.double()).flatten().cpu())
            affected += int((~torch.isclose(a, b, atol=atol, rtol=rtol)).flatten(1).any(1).sum())
            prediction_errors.append((probe(a).double() - probe(b).double()).flatten().cpu())

    def summarize(values):
        delta = torch.cat(values).abs()
        return {
            "mean_abs": float(delta.mean()),
            "rms": float(delta.square().mean().sqrt()),
            "p95_abs": float(torch.quantile(delta, 0.95)),
            "max_abs": float(delta.max()),
        }

    return {
        "frames": len(keys),
        "affected_frames": affected,
        "affected_frame_fraction": affected / len(keys),
        "features": summarize(feature_errors),
        "predictions": summarize(prediction_errors),
    }


def run_diagnostic(
    dataset,
    samples,
    *,
    audio_budget=300,
    reference_audio_budget=10240,
    frame_batch_size=1024,
    updates=5,
    seed=115,
    lr=1e-3,
    weight_decay=1e-4,
    schedule_steps=10000,
    warmup=0.05,
    atol=1e-5,
    rtol=1e-4,
    probe_checkpoint=None,
):
    encoder = dataset.content_encoder.eval().requires_grad_(False)
    target = dataset.target
    groups = list(FrameBudgetBatchSampler(dataset.frame_lengths, audio_budget, shuffle=False))
    reference_groups = list(
        FrameBudgetBatchSampler(dataset.frame_lengths, reference_audio_budget, shuffle=False)
    )
    offline_frames, online_frames = [], []
    parity = []
    cached_groups = []
    for indices in groups:
        group = [samples[index] for index in indices]
        extracted = dataset.extract_frames(group)
        with torch.no_grad():
            content = encoder(dataset.base_dataset.collate_fn(group))
            online = dataset.frames_from_content(group, content)
        parity.append(compare_frames(extracted, online, target, atol=atol, rtol=rtol))
        offline_frames.extend(extracted)
        online_frames.extend(online)
        if extracted:
            cached_groups.append((group, extracted))
    if not offline_frames:
        raise ValueError("No valid target frames in the selected utterances.")
    reference = []
    for indices in reference_groups:
        reference.extend(dataset.extract_frames([samples[index] for index in indices]))
    padding = compare_frames(reference, offline_frames, target, atol=atol, rtol=rtol)

    # Use actual training modules with identical frame batches, initialization and schedules.
    # Logging is disabled; their native step implementations still run unchanged.
    torch.manual_seed(seed)
    probe = Probe(dataset.feature_dim * dataset.context_size, 1, [512], "gelu")
    if probe_checkpoint is not None:
        from .evaluate import load_checkpoint

        checkpoint = load_checkpoint(probe_checkpoint)
        state = checkpoint["state_dict"]
        probe.load_state_dict(
            {
                key.removeprefix("probe."): value
                for key, value in state.items()
                if key.startswith("probe.")
            },
            strict=True,
        )
    optimization = Optimization(
        optimizer_kwargs={"lr": lr, "weight_decay": weight_decay},
        lr_scheduler_kwargs={"T_max": schedule_steps},
        warmup=LinearWarmup(warmup),
    )
    device = encoder.device
    probe.to(device).eval()
    impact = batching_impact(reference, offline_frames, probe, atol=atol, rtol=rtol)
    impact["probe_source"] = str(probe_checkpoint) if probe_checkpoint else "seeded initialization"
    frame_module = QuietFrameModule(probe=probe, target=target, optimization=optimization).to(
        device
    )
    online_module = QuietOnlineModule(
        probe=copy.deepcopy(probe),
        target=target,
        optimization=optimization,
        content_encoder=encoder,
        train_dataset=dataset,
        val_dataset=dataset,
        train_encoder=False,
    ).to(device)
    for module in (frame_module, online_module):
        module._trainer = SimpleNamespace(estimated_stepping_batches=schedule_steps, global_step=0)
        module.train()
    frame_optim = frame_module.configure_optimizers()
    online_optim = online_module.configure_optimizers()
    encoder_versions = tuple(parameter._version for parameter in encoder.parameters())
    update_checks = []
    for step in range(updates):
        samples_for_step, frames = cached_groups[step % len(cached_groups)]
        frame_optim["optimizer"].zero_grad()
        online_optim["optimizer"].zero_grad()
        frame_output = frame_module._shared_step(AudioBatch.from_samples(frames), "train")
        online_output = online_module._shared_step(
            dataset.base_dataset.collate_fn(samples_for_step), "train"
        )
        frame_output.loss.backward()
        online_output.loss.backward()
        gradients = [
            tensor_check(left.grad, right.grad, atol=atol, rtol=rtol)
            for left, right in zip(
                frame_module.probe.parameters(), online_module.probe.parameters(), strict=True
            )
        ]
        frame_lr = frame_optim["optimizer"].param_groups[0]["lr"]
        online_lr = online_optim["optimizer"].param_groups[0]["lr"]
        frame_optim["optimizer"].step()
        online_optim["optimizer"].step()
        parameters = [
            tensor_check(left, right, atol=atol, rtol=rtol)
            for left, right in zip(
                frame_module.probe.parameters(), online_module.probe.parameters(), strict=True
            )
        ]
        check = {
            "update": step + 1,
            "frames": frame_output.batch_size,
            "same_frame_count": frame_output.batch_size == online_output.batch_size,
            "frame_loss": float(frame_output.loss.detach()),
            "online_loss": float(online_output.loss.detach()),
            "loss": tensor_check(frame_output.loss, online_output.loss, atol=atol, rtol=rtol),
            "predictions": tensor_check(
                frame_output.prediction, online_output.prediction, atol=atol, rtol=rtol
            ),
            "gradient_max_abs": max(value["max_abs"] for value in gradients),
            "parameter_max_abs": max(value["max_abs"] for value in parameters),
            "gradients_pass": all(value["pass"] for value in gradients),
            "parameters_pass": all(value["pass"] for value in parameters),
            "frame_lr": frame_lr,
            "online_lr": online_lr,
            "encoder_frozen": all(
                not p.requires_grad and p.grad is None for p in encoder.parameters()
            ),
            "encoder_eval": not encoder.training,
        }
        check["pass"] = (
            all(
                check[key]
                for key in (
                    "same_frame_count",
                    "gradients_pass",
                    "parameters_pass",
                    "encoder_frozen",
                    "encoder_eval",
                )
            )
            and check["loss"]["pass"]
            and check["predictions"]["pass"]
            and frame_lr == online_lr
        )
        update_checks.append(check)
        for optim in (frame_optim, online_optim):
            optim["lr_scheduler"]["scheduler"].step()
    extraction_pass = all(check["pass"] for check in parity)
    updates_pass = all(check["pass"] for check in update_checks)
    return {
        "controlled_parity_pass": extraction_pass
        and updates_pass
        and encoder_versions == tuple(parameter._version for parameter in encoder.parameters()),
        "encoder_weights_unchanged": encoder_versions
        == tuple(parameter._version for parameter in encoder.parameters()),
        "audio_batch_invariance_pass": padding["pass"],
        "selected_utterances": [sample.utt_id for sample in samples],
        "valid_frames": len(offline_frames),
        "same_audio_batch_extraction": parity,
        "different_audio_batch_extraction": padding,
        "audio_batching_impact": impact,
        "controlled_updates": update_checks,
        "native_batching": {
            "frame_batch_size": frame_batch_size,
            "frame_updates": frame_counts(reference, frame_batch_size, seed),
            "online_updates": [
                {"utterances": len(group), "frames": len(frames)} for group, frames in cached_groups
            ],
            "audio_budget": audio_budget,
            "reference_audio_budget": reference_audio_budget,
            "reference_audio_batches": [len(indices) for indices in reference_groups],
            "online_audio_batches": [len(indices) for indices in groups],
        },
        "scope": "Controlled updates intentionally use the same utterance-derived frame batches in both training modules. Native batching statistics describe only this selected subset; they do not establish full-run convergence parity. No encoder training or W&B logging.",
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--dataset", default="librispeech")
    parser.add_argument("--split", default="train-clean-100")
    parser.add_argument("--smile-root", type=Path, default=Path("features/opensmile/librispeech"))
    parser.add_argument("--encoder", default="wavlm")
    parser.add_argument(
        "--probe-checkpoint",
        type=Path,
        help="Optional frozen checkpoint with the diagnostic's encoder, target, context, 512/GELU head and normalization. Strict head loading; does not restore encoder weights or config.",
    )
    parser.add_argument("--encoder-kwargs", type=json.loads, default={"layer": 6})
    parser.add_argument("--target", choices=TARGETS, default="jitter")
    parser.add_argument("--target-standardization", type=json.loads)
    parser.add_argument("--context-size", type=int, default=3)
    parser.add_argument("--utterances", type=int, default=16)
    parser.add_argument("--updates", type=int, default=5)
    parser.add_argument("--inference-frame-budget", type=int, default=300)
    parser.add_argument("--reference-inference-frame-budget", type=int, default=10240)
    parser.add_argument("--frame-batch-size", type=int, default=1024)
    parser.add_argument("--schedule-steps", type=int, default=10000)
    parser.add_argument("--warmup", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=115)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    for key in (
        "utterances",
        "updates",
        "inference_frame_budget",
        "reference_inference_frame_budget",
        "frame_batch_size",
        "schedule_steps",
    ):
        if getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if args.context_size <= 0 or args.context_size % 2 == 0:
        parser.error("--context-size must be positive and odd")
    if args.lr <= 0 or args.weight_decay < 0 or args.atol < 0 or args.rtol < 0:
        parser.error("Invalid optimizer or tolerance value")
    if not 0 < args.warmup <= 1:
        parser.error("--warmup must be a fraction in (0, 1]")
    return args


def main():
    args = parse_args()
    print("Loading frozen encoder and selecting diagnostic utterances", flush=True)
    encoder = build_content_encoder(args.encoder, args.encoder_kwargs)
    encoder.to(args.device).eval().requires_grad_(False)
    encoder.device = torch.device(args.device)
    target = TARGETS[args.target]
    if not isinstance(target.task, RegressionTask) or target.transform_kwargs is not None:
        raise ValueError("Diagnostic supports regression targets without speaker statistics.")
    if args.target_standardization:
        target = replace(
            target,
            task=target.task.with_standardization(
                TargetStandardization(**args.target_standardization)
            ),
        )
    source = load_dataset(
        args.dataset,
        root=args.root,
        splits=[args.split],
        load=["audio"],
        target_sr=encoder.sample_rate,
    )
    rows = sorted(source.rows, key=lambda row: row.utt_id)
    selected = random.Random(args.seed).sample(rows, min(args.utterances, len(rows)))
    if not selected:
        raise ValueError("No audio found in the requested split.")
    # A bounded manifest avoids inspecting duration headers for the entire split.
    with tempfile.TemporaryDirectory() as temporary:
        manifest = Path(temporary) / "selected.csv"
        with manifest.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["utt_id", "path", "split"])
            writer.writerows(
                (row.utt_id, str(Path(row.path).resolve()), row.split) for row in selected
            )
        dataset = FrameDataset(
            encoder,
            target,
            args.smile_root if args.dataset == "clac" else args.smile_root / args.split,
            manifest_path=manifest,
            context_size=args.context_size,
            shuffle=False,
        )
        samples = [dataset.base_dataset[index] for index in range(len(selected))]
        print(
            f"Checking {len(samples)} utterances and {args.updates} controlled updates", flush=True
        )
        report = run_diagnostic(
            dataset,
            samples,
            audio_budget=args.inference_frame_budget,
            reference_audio_budget=args.reference_inference_frame_budget,
            frame_batch_size=args.frame_batch_size,
            updates=args.updates,
            seed=args.seed,
            lr=args.lr,
            weight_decay=args.weight_decay,
            schedule_steps=args.schedule_steps,
            warmup=args.warmup,
            atol=args.atol,
            rtol=args.rtol,
            probe_checkpoint=args.probe_checkpoint,
        )
    report["arguments"] = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("controlled_parity_pass", "audio_batch_invariance_pass", "valid_frames")
            },
            indent=2,
        ),
        flush=True,
    )
    print(f"Report: {args.out_dir / 'report.json'}", flush=True)
    if not report["controlled_parity_pass"] or not report["audio_batch_invariance_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
