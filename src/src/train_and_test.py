#!/usr/bin/env python
import argparse
import logging
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from transformers import AutoTokenizer

from src.closest_rtr.closest_rtr import ClosestAnswerWithEvidenceModule
from src.mca.mca import MostCommonAnswerWithEvidenceModule
from src.model import AnswerWithEvidenceModule
from src.rdm.random import RandomAnswerWithEvidenceModule
from src.transformer_models.model import TransformersAnswerWithEvidenceModule
from src.utils.logger_utils import UninitializedWeightsFilter
from src.video_qa_with_evidence_dataset import VideoQAWithEvidenceDataModule


def model_type_to_class(model_type: str) -> type[AnswerWithEvidenceModule]:  # noqa
    if model_type == "random":
        return RandomAnswerWithEvidenceModule
    elif model_type == "most_common_ans":
        return MostCommonAnswerWithEvidenceModule
    elif model_type == "closest_rtr":
        return ClosestAnswerWithEvidenceModule
    else:
        return TransformersAnswerWithEvidenceModule


def train_and_test(args: argparse.Namespace) -> None:
    should_train = args.model_type not in {"random", "most_common_ans", "closest_rtr", "t5_zero_shot"}

    os.environ["TOKENIZERS_PARALLELISM"] = "0"
    if args.model_type == "violet_decoder":
        encoder_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        decoder_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    elif args.model_type == "clip_decoder":
        encoder_tokenizer = AutoTokenizer.from_pretrained(args.pretrained_clip_ckpt_path)
        decoder_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    elif args.model_type in {"random", "most_common_ans", "closest_rtr"}:
        encoder_tokenizer = decoder_tokenizer = None
    else:
        encoder_tokenizer = decoder_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    data_module = VideoQAWithEvidenceDataModule(args, encoder_tokenizer=encoder_tokenizer,
                                                decoder_tokenizer=decoder_tokenizer)

    if args.model_type in {"random", "most_common_ans", "closest_rtr"}:
        args.train_instances = data_module.train_dataloader().dataset
    else:
        args.decoder_tokenizer = decoder_tokenizer

    if should_train:
        uninitialized_weights_filter = UninitializedWeightsFilter()
        logging.getLogger("transformers.modeling_utils").addFilter(uninitialized_weights_filter)
    else:
        uninitialized_weights_filter = None

    model_class = model_type_to_class(args.model_type)
    model = model_class(**args.__dict__)

    if uninitialized_weights_filter:
        logging.getLogger("transformers.modeling_utils").removeFilter(uninitialized_weights_filter)

    loggers = [
        WandbLogger(name=args.wandb_name, project=args.wandb_project, entity=args.wandb_entity,
                    offline=args.wandb_offline),
        TensorBoardLogger(save_dir="transformer_models"),
    ]

    callbacks = [
        RichProgressBar(),
        pl.callbacks.ModelCheckpoint(dirpath=args.output_ckpt_dir,
                                     filename="{epoch}-{loss/train:.2f}", monitor="loss/train"),
    ]

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
