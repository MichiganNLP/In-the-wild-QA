#!/usr/bin/env python
import argparse
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer

from src.transformer_models.logger import LoggingCallback
from src.transformer_models.model import FineTuner
from src.transformer_models.video_qa_with_evidence_dataset import VideoQAWithEvidenceForT5DataModule


def transformer_train(args: argparse.Namespace) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "0"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    data_module = VideoQAWithEvidenceForT5DataModule(args, tokenizer=tokenizer)

    model = FineTuner(tokenizer=tokenizer, **args.__dict__)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=args.output_ckpt_dir,
                                                       filename="{epoch}-{train_loss:.2f}", monitor="train_loss")

    wandb_logger = WandbLogger(name=args.wandb_name, project=args.wandb_project, entity=args.wandb_entity,
                               offline=args.wandb_offline)

    trainer = pl.Trainer(accumulate_grad_batches=args.gradient_accumulation_steps, gpus=args.n_gpu,
                         max_epochs=args.num_train_epochs, precision=16 if args.fp_16 else 32,
                         amp_level=args.opt_level, gradient_clip_val=args.max_grad_norm,
                         callbacks=[RichProgressBar(), LoggingCallback(), checkpoint_callback], logger=wandb_logger,
                         log_every_n_steps=1)

    trainer.fit(model, datamodule=data_module)
