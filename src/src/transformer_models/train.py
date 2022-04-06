#!/usr/bin/env python
import argparse
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from transformers import AutoTokenizer

from src.transformer_models.logger import LoggingCallback
from src.transformer_models.model import AnswerWithEvidenceModule
from src.utils.timer import duration
from src.video_qa_with_evidence_dataset import VideoQAWithEvidenceDataModule


def transformer_train(args: argparse.Namespace) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "0"
    if args.model_type == "violet_decoder":
        tokenizer = {
            "encoder": AutoTokenizer.from_pretrained("bert-base-uncased"),
            "decoder": AutoTokenizer.from_pretrained(args.model_name_or_path),
        }
    elif args.model_type == "clip_decoder":
        tokenizer = {
            "encoder": AutoTokenizer.from_pretrained(args.pretrained_clip_ckpt_path),
            "decoder": AutoTokenizer.from_pretrained(args.model_name_or_path),
        }
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    data_module = VideoQAWithEvidenceDataModule(args, tokenizer=tokenizer)

    duration("Loading the data…", args.timer)

    model = AnswerWithEvidenceModule(tokenizer=tokenizer, **args.__dict__)

    duration("Initializing the model…", args.timer)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=args.output_ckpt_dir,
                                                       filename="{epoch}-{loss/train:.2f}", monitor="loss/train")

    loggers = [
        WandbLogger(name=args.wandb_name, project=args.wandb_project, entity=args.wandb_entity,
                    offline=args.wandb_offline),
        TensorBoardLogger(save_dir="."),
    ]

    trainer = pl.Trainer(accumulate_grad_batches=args.gradient_accumulation_steps, gpus=args.n_gpu,
                         max_epochs=args.num_train_epochs, precision=16 if args.fp_16 else 32,
                         amp_level=args.opt_level, gradient_clip_val=args.max_grad_norm, logger=loggers,
                         callbacks=[RichProgressBar(), LoggingCallback(), checkpoint_callback], log_every_n_steps=1)

    duration("Initializing the trainer…", args.timer)
    trainer.fit(model, datamodule=data_module)
    duration("Finish training…", args.timer)

    if args.test_after_train:
        trainer.test(model, datamodule=data_module)
