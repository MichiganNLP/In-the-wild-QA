#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

output_ckpt_dir=ckpts/t5/text_finetuned/
num_train_epochs=10
train_batch_size=2
eval_batch_size=2
wandb_name=t5
wandb_entity=in-the-wild-vqa-um
log_dir=logs/t5/
log_path=logs/t5/baseline.log
gradient_accumulation_steps=16

mkdir -p ${output_ckpt_dir}
mkdir -p ${log_dir}


python -m src t5_train \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --output_ckpt_dir ${output_ckpt_dir} \
    --num_train_epochs ${num_train_epochs} \
    --train_batch_size ${train_batch_size} \
    --eval_batch_size ${eval_batch_size} \
    --wandb_name ${wandb_name} \
    --wandb_entity ${wandb_entity} 
    
# > ${log_path}