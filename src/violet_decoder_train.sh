#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

output_ckpt_dir=ckpts/violet_decoder/
num_train_epochs=10
train_batch_size=1
eval_batch_size=1
log_dir=logs/violet_decoder/
log_path=logs/violet_decoder/baseline.log
gradient_accumulation_steps=16
max_seq_length=128
max_vid_length=2048
sample_rate=60
seed=42
wandb_name=t5_violet_decoder-${sample_rate}_mvl-${max_vid_length}_sd-${seed}
wandb_entity=in-the-wild-vqa-um

mkdir -p ${output_ckpt_dir}
mkdir -p ${log_dir}

python -m src violet_decoder \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --output_ckpt_dir ${output_ckpt_dir} \
    --num_train_epochs ${num_train_epochs} \
    --train_batch_size ${train_batch_size} \
    --eval_batch_size ${eval_batch_size} \
    --max_seq_length ${max_seq_length} \
    --seed ${seed} \
    --wandb_name ${wandb_name} \
    --wandb_entity ${wandb_entity} \
    --max_vid_length ${max_vid_length} \
    --sample_rate ${sample_rate}

# > ${log_path}
