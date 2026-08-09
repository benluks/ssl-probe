import argparse
from pathlib import Path

import lightning as L
import torch
from quick_convert.pipelines.training import (
    LightningTrainer,
    Optimization,
    TrainingPipeline,
)
from quick_convert.pipelines.training.optim.base import LinearWarmup

from .dataset import FrameDataset
from .module import Probe
from .targets import TARGETS


def parse_args():
    parser = argparse.ArgumentParser()

    # Data / representation
    parser.add_argument("--root", required=True)
    parser.add_argument("--train-split", default="train-clean-100")
    parser.add_argument("--val-split", default="dev-clean")
    parser.add_argument("--layer", type=int, default=5)
    parser.add_argument("--context-size", type=int, default=1)

    parser.add_argument(
        "--target",
        choices=TARGETS,
        default="pitch",
    )

    parser.add_argument("--frame-batch-size", type=int, default=8192)
    parser.add_argument("--hidden-dim", type=int, nargs="+", default=[])

    parser.add_argument(
        "--nonlinearity",
        choices=["relu", "gelu", "none"],
        default="gelu",
    )

    # Optimization
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--warmup", type=float, default=0.05)

    # Output
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--seed", type=int, default=115)

    return parser.parse_args()


def main():
    args = parse_args()

    L.seed_everything(args.seed, workers=True)

    config = vars(args).copy()

    target = TARGETS[args.target]

    train_dataset = FrameDataset(
        root=args.root,
        splits=[args.train_split],
        layer=args.layer,
        target=target,
        context_size=args.context_size,
        frame_batch_size=args.frame_batch_size,
        inference_frame_budget=10 * args.frame_batch_size,
    )

    val_dataset = (
        None
        if args.val_split is None
        else FrameDataset(
            root=args.root,
            splits=[args.val_split],
            layer=args.layer,
            target=target,
            context_size=args.context_size,
            frame_batch_size=args.frame_batch_size,
            inference_frame_budget=10 * args.frame_batch_size,
            shuffle=False,
        )
    )

    module = Probe(
        input_dim=train_dataset.content_encoder.encoder.feature_dim
        * train_dataset.context_size,
        target=target,
        hidden_dim=args.hidden_dim,
        nonlinearity=args.nonlinearity,
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

    trainer = LightningTrainer(
        module=module,
        train_dataloader_kwargs={},
        trainer_kwargs={
            "max_steps": args.max_steps,
            "accelerator": "auto",
        },
    )

    out_dir = args.out_dir or (
        Path("outputs")
        / f"{args.target}"
        / f"wavlm_l{args.layer + 1}_{args.nonlinearity}"
    )

    pipeline = TrainingPipeline(
        trainer=trainer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        out_dir=out_dir,
    )
    trainer.pl_trainer.logger.log_hyperparams(config)
    pipeline.run()


if __name__ == "__main__":
    main()
