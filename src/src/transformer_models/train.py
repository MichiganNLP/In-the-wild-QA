#!/usr/bin/env python
import argparse
import logging
import os
import re

import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from transformers import AutoTokenizer

from src.transformer_models.logger import LoggingCallback
from src.transformer_models.model import AnswerWithEvidenceModule
from src.video_qa_with_evidence_dataset import VideoQAWithEvidenceDataModule


class ShouldTrainFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno != logging.WARNING \
               or re.match(r"^Some weights of \w+ were not initialized from the model .+", record.getMessage()) is None


def train_transformer(args: argparse.Namespace) -> None:
    should_train = args.model_type != "T5_zero_shot"

    os.environ["TOKENIZERS_PARALLELISM"] = "0"
    if args.model_type == "violet_decoder":
        encoder_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        decoder_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    elif args.model_type == "clip_decoder":
        encoder_tokenizer = AutoTokenizer.from_pretrained(args.pretrained_clip_ckpt_path)
        decoder_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    else:
        encoder_tokenizer = decoder_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    data_module = VideoQAWithEvidenceDataModule(args, encoder_tokenizer=encoder_tokenizer,
                                                decoder_tokenizer=decoder_tokenizer)

    if should_train:
        logging.getLogger("transformers.modeling_utils").addFilter(ShouldTrainFilter())
    model = AnswerWithEvidenceModule(decoder_tokenizer=decoder_tokenizer, **args.__dict__)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=args.output_ckpt_dir,
                                                       filename="{epoch}-{loss/train:.2f}", monitor="loss/train")

    loggers = [
        WandbLogger(name=args.wandb_name, project=args.wandb_project, entity=args.wandb_entity,
                    offline=args.wandb_offline),
        TensorBoardLogger(save_dir="."),
    ]

    callbacks = [RichProgressBar(), LoggingCallback(), checkpoint_callback]

    trainer = pl.Trainer(accumulate_grad_batches=args.gradient_accumulation_steps, gpus=args.n_gpu,
                         max_epochs=args.num_train_epochs, precision=16 if args.fp_16 else 32,
                         amp_level=args.opt_level, gradient_clip_val=args.max_grad_norm, profiler=args.profiler,
                         log_every_n_steps=1, logger=loggers, callbacks=callbacks)

    if should_train:
        trainer.fit(model, datamodule=data_module)
    else:
        trainer.validate(model, datamodule=data_module)

    if args.test_after_train:
        trainer.test(model, datamodule=data_module)
