#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

output_ckpt_dir=ckpts/t5/text_finetuned/
num_train_epochs=10
wandb_name=t5
log_dir=logs/t5/
log_path=logs/t5/baseline.log

mkdir -p ${output_ckpt_dir}
mkdir -p ${log_dir}

python -m src t5_train \
    --output_ckpt_dir ${output_ckpt_dir} \
    --num_train_epochs ${num_train_epochs} \
    --wandb_name ${wandb_name} \
    | tee ${log_path}
