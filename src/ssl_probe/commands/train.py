import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from quick_convert.pipelines.training.pipeline import TrainingPipeline
from quick_convert.training.lightning.optim import LinearWarmup, Optimization
from quick_convert.training.lightning.trainer import LightningTrainer

from ..dataset import FrameDataset
from ..encoders import build_content_encoder, build_layer_fusion, content_encoder_slug
from ..probe import Probe
from ..targets import TARGETS, RegressionTask
from ..training import ProbeTrainingModule


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()

    # Data / representation
    parser.add_argument("--root", required=True)
    parser.add_argument("--dataset", type=str, default="librispeech")
    parser.add_argument("--train-split", default="train-clean-100")
    parser.add_argument("--val-split", default="dev-clean")
    parser.add_argument(
        "--encoder",
        default="wavlm",
        help="Encoder alias (wavlm, w2vbert, dac) or dotted ContentEncoder class path.",
    )
    parser.add_argument(
        "--encoder-kwargs",
        type=json.loads,
        default={},
        metavar="JSON",
        help="JSON constructor arguments, for example '{\"layer\": 12}'.",
    )
    parser.add_argument("--context-size", type=int, default=1)
    parser.add_argument(
        "--layer-fusion",
        choices=["none", "weighted-sum"],
        default="none",
        help="How to combine multi-layer encoder output before probing.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Output layer count override for encoders whose metadata cannot be inspected.",
    )
    parser.add_argument(
        "--layer-log-interval",
        type=int,
        default=1000,
        help="Training-step interval for normalized layer-weight logging.",
    )

    parser.add_argument(
        "--target",
        choices=TARGETS,
        default="logf0",
    )

    parser.add_argument("--frame-batch-size", type=int, default=8192)
    parser.add_argument("--hidden-dim", type=int, nargs="+", default=[])
    parser.add_argument(
        "--target-normalization",
        choices=["none", "standardize"],
        default="none",
        help=(
            "Standardize regression loss targets with moments computed from valid training frames. "
            "Metrics remain on the target's transformed, unstandardized scale."
        ),
    )

    parser.add_argument(
        "--nonlinearity",
        choices=["relu", "gelu", "none"],
        default="gelu",
    )

    # Optimization
    parser.add_argument("--val-check-interval", type=int, default=1000)
    parser.add_argument("--check-val-every-n-epoch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--warmup", type=float, default=0.05)
    parser.add_argument(
        "--smile-root",
        type=Path,
        default=None,
        help="Feature-store root. Defaults to features/opensmile/<dataset>.",
    )

    # Output
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--seed", type=int, default=115)
    parser.add_argument(
        "--experiment-label",
        default=None,
        help="Optional data or experiment variant included in the run name.",
    )
    parser.add_argument(
        "--speaker-stats",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--spk-id-template",
        default="{path.parent.parent.stem}",
    )

    return parser.parse_args(argv)


def resolve_smile_root(dataset: str, smile_root: Path | None) -> Path:
    if smile_root is not None:
        return smile_root
    return Path("features") / "opensmile" / dataset


def build_run_name(
    *,
    target: str,
    dataset: str,
    encoder_slug: str,
    nonlinearity: str,
    context_size: int,
    experiment_label: str | None,
) -> Path:
    components = [
        experiment_label,
        dataset,
        encoder_slug,
        nonlinearity,
        f"c{context_size}",
    ]
    return Path(target) / "_".join(component for component in components if component)


def main():
    args = parse_args()
    args.smile_root = resolve_smile_root(args.dataset, args.smile_root)

    L.seed_everything(args.seed, workers=True)

    config = vars(args).copy()

    target = TARGETS[args.target]
    content_encoder = build_content_encoder(args.encoder, args.encoder_kwargs)
    layer_fusion = build_layer_fusion(
        args.layer_fusion,
        content_encoder,
        args.num_layers,
    )

    train_dataset = FrameDataset(
        content_encoder=content_encoder,
        root=args.root,
        dataset_name=args.dataset,
        splits=[args.train_split],
        smile_root=args.smile_root / args.train_split,
        speaker_stats=args.speaker_stats,
        spk_id_template=args.spk_id_template,
        target=target,
        context_size=args.context_size,
        frame_batch_size=args.frame_batch_size,
        inference_frame_budget=10 * args.frame_batch_size,
        preserve_layers=layer_fusion is not None,
    )

    target_standardization = None
    if args.target_normalization == "standardize":
        if not isinstance(target.task, RegressionTask):
            raise ValueError(
                "--target-normalization standardize is only valid for regression targets."
            )

        target_standardization = train_dataset.compute_target_standardization()
        target = replace(
            target,
            task=target.task.with_standardization(target_standardization),
        )
        train_dataset.target = target
        config["target_standardization"] = asdict(target_standardization)

    val_dataset = (
        None
        if args.val_split is None
        else FrameDataset(
            content_encoder=content_encoder,
            root=args.root,
            dataset_name=args.dataset,
            splits=[args.val_split],
            smile_root=args.smile_root / args.val_split,
            speaker_stats=args.speaker_stats,
            spk_id_template=args.spk_id_template,
            target=target,
            context_size=args.context_size,
            frame_batch_size=args.frame_batch_size,
            inference_frame_budget=10 * args.frame_batch_size,
            shuffle=False,
            preserve_layers=layer_fusion is not None,
        )
    )

    probe = Probe(
        input_dim=train_dataset.feature_dim * train_dataset.context_size,
        output_dim=target.task.output_dim,
        hidden_dim=args.hidden_dim,
        nonlinearity=args.nonlinearity,
        feature_transform=layer_fusion,
    )

    module = ProbeTrainingModule(
        probe=probe,
        target=target,
        layer_log_interval=args.layer_log_interval,
        optimization=Optimization(
            optimizer=torch.optim.AdamW,
            optimizer_kwargs={
                "lr": args.lr,
                "weight_decay": args.weight_decay,
            },
            lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR,
            lr_scheduler_kwargs={
                "T_max": args.max_steps,
            },
            warmup=LinearWarmup(args.warmup),
        ),
    )

    encoder_slug = content_encoder_slug(content_encoder, layer_fusion)
    run_name = build_run_name(
        target=args.target,
        dataset=args.dataset,
        encoder_slug=encoder_slug,
        nonlinearity=args.nonlinearity,
        context_size=args.context_size,
        experiment_label=args.experiment_label,
    )

    out_dir = args.out_dir or (Path("outputs") / run_name)

    trainer = LightningTrainer(
        module=module,
        train_dataloader_kwargs={},
        trainer_kwargs={
            "max_steps": args.max_steps,
            "accelerator": "auto",
            "val_check_interval": args.val_check_interval,
            "check_val_every_n_epoch": args.check_val_every_n_epoch,
            "callbacks": [
                ModelCheckpoint(
                    save_top_k=2,
                    monitor="step",
                    mode="max",
                    save_last=False,
                    save_on_train_epoch_end=False,
                ),
                ModelCheckpoint(monitor="val/loss", mode="min", save_last=False),
                LearningRateMonitor(logging_interval="step"),
            ],
            "logger": WandbLogger(name=str(run_name), save_dir=out_dir, project="ssl-probe"),
        },
    )

    pipeline = TrainingPipeline(
        trainer=trainer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        out_dir=out_dir,
    )
    run_dir = pipeline.prepare()

    if target_standardization is not None:
        stats_path = run_dir / "target_standardization.json"
        stats_path.write_text(json.dumps(asdict(target_standardization), indent=2) + "\n")

    trainer.pl_trainer.logger.log_hyperparams(config)
    pipeline.run()

    if trainer.pl_trainer.is_global_zero:
        module.export_layer_weights(run_dir / "layer_weights.json")


if __name__ == "__main__":
    main()
