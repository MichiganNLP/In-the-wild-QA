#!/usr/bin/env python
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from src.transformer_models.logger import LoggingCallback
from src.transformer_models.model import FineTuner


def transformer_train(args):
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.output_ckpt_dir, filename="{epoch}-{train_loss:.2f}", monitor="train_loss", mode="min",
        save_top_k=1)

    wandb_logger = WandbLogger(name=args.wandb_name, project=args.wandb_project, entity=args.wandb_entity, offline=True)

    train_params = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gpus=args.n_gpu,
        max_epochs=args.num_train_epochs,
        precision=16 if args.fp_16 else 32,
        amp_level=args.opt_level,
        gradient_clip_val=args.max_grad_norm,
        callbacks=[LoggingCallback(), checkpoint_callback],
        logger=wandb_logger,
        log_every_n_steps=1
    )

    model = FineTuner(args)
    trainer = pl.Trainer(**train_params)
    trainer.fit(model)
