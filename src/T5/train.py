#!/usr/bin/env python
import argparse
import glob
import os
import json
import time
import logging
import random
import re
from itertools import chain
from string import punctuation

import pytorch_lightning as pl
import pandas as pd
import numpy as np
import torch

from pytorch_lightning.loggers import WandbLogger

from logger import LoggingCallback
from model import T5FineTuner


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default='t5-base')
    parser.add_argument("--tokenizer_name_or_path", default='t5-base')
    parser.add_argument("--max_seq_length", default=512)
    parser.add_argument("--learning_rate", default=3e-4)
    parser.add_argument("--weight_decay", default=0.0)
    parser.add_argument("--adam_epsilon", default=1e-8)
    parser.add_argument("--warmup_steps", default=0)
    parser.add_argument("--train_batch_size", default=8)
    parser.add_argument("--eval_batch_size", default=8)
    parser.add_argument("--num_train_epochs", default=100)
    parser.add_argument("--gradient_accumulation_steps", default=16)
    parser.add_argument("--n_gpu", default=1)
    parser.add_argument("--early_stop_callback", default=False)

    # if you want to enable 16-bit training then install apex and set this to true
    parser.add_argument("--fp_16", default=False)

     # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
    parser.add_argument("--opt_level", default='01')

    # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
    parser.add_argument("--max_grad_norm", default=1.0)
    parser.add_argument("--seed", default=42)
    parser.add_argument("--use_tpu", default=False)

    parser.add_argument("--wandb_project", default='question-suggestion-T5')
    parser.add_argument("--wandb_name", required=True)

    args = parser.parse_args()

    return args




args = parse_args()

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath=args.output_dir, filename="{epoch}-{val_acc:.2f}", monitor="val_acc", mode="max", save_top_k=1
)

wandb_logger = WandbLogger(name=args.wandb_name,project=args.wandb_project)

train_params = dict(
    accumulate_grad_batches=args.gradient_accumulation_steps,
    gpus=args.n_gpu,
    max_epochs=args.num_train_epochs,
    precision= 16 if args.fp_16 else 32,
    amp_level=args.opt_level,
    gradient_clip_val=args.max_grad_norm,
    callbacks=[LoggingCallback(), checkpoint_callback],
    logger = wandb_logger,
    log_every_n_steps=1
)


model = T5FineTuner(args)
trainer = pl.Trainer(**train_params)
trainer.fit(model)
