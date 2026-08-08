import torch
from quick_convert.pipelines.training import (
    LightningTrainer,
    Optimization,
    TrainingPipeline,
)
from quick_convert.pipelines.training.optim.base import LinearWarmup

from .dataset import FrameDataset
from .module import Probe

MAX_STEPS = 10_000

train_dataset = FrameDataset(utterance_batch_size=4, frame_batch_size=512)
module = Probe(
    input_dim=train_dataset.content_encoder.encoder.feature_dim,
    optimization=Optimization(
        optimizer=torch.optim.AdamW,
        optimizer_kwargs={
            "lr": 1e-3,
            "weight_decay": 1e-4,
        },
        lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR,
        lr_scheduler_kwargs={"T_max": MAX_STEPS},
        warmup=LinearWarmup(0.05),
    ),
)

trainer = LightningTrainer(
    module=module,
    train_dataloader_kwargs={},
    trainer_kwargs={
        "max_epochs": 10,
        "max_steps": MAX_STEPS,
        "accelerator": "auto",
    },
)

pipeline = TrainingPipeline(
    trainer=trainer,
    train_dataset=train_dataset,
    out_dir="outputs/pitch_probe",
)

pipeline.run()
