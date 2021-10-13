#!/bin/bash

# params needs to be modified
CUDA_ID=0


# params based on the params at the top
output_ckpt_dir=T5/ckpt/
num_train_epochs=10
train_batch_size=2
eval_batch_size=2
wandb_name=T5
log_path=T5/logs/baseline.log
gradient_accumulation_steps=16

mkdir -p ${output_ckpt_dir}


CUDA_VISIBLE_DEVICES=${CUDA_ID} python -W ignore main.py T5_train\
    --train_data example_data/train.json \
    --dev_data example_data/dev.json \
    --test_data example_data/test.json \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --output_ckpt_dir ${output_ckpt_dir} \
    --num_train_epochs ${num_train_epochs} \
    --train_batch_size ${train_batch_size} \
    --eval_batch_size ${eval_batch_size} \
    --wandb_name ${wandb_name} > ${log_path}