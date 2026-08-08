import argparse

import torch
from quick_convert.pipelines.training import (
    LightningTrainer,
    Optimization,
    TrainingPipeline,
)
from quick_convert.pipelines.training.optim.base import LinearWarmup

from .dataset import FrameDataset
from .module import Probe


def parse_args():
    parser = argparse.ArgumentParser()

    # Data / representation
    parser.add_argument("--root", required=True)
    parser.add_argument("--split", default="train-clean-100")
    parser.add_argument("--layer", type=int, default=5)
    parser.add_argument("--frame-batch-size", type=int, default=16_382)

    # Optimization
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--warmup", type=float, default=0.05)

    # Output
    parser.add_argument("--out-dir", default="outputs/pitch")

    return parser.parse_args()


def main():
    args = parse_args()

    train_dataset = FrameDataset(
        root=args.root,
        splits=[args.split],
        layer=args.layer,
        frame_batch_size=args.frame_batch_size,
        inference_frame_budget=10 * args.frame_batch_size,
    )

    module = Probe(
        input_dim=train_dataset.content_encoder.encoder.feature_dim,
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

    pipeline = TrainingPipeline(
        trainer=trainer,
        train_dataset=train_dataset,
        out_dir=args.out_dir,
    )

    pipeline.run()


if __name__ == "__main__":
    main()
