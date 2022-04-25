#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

output_ckpt_dir=ckpts/t5/text_visual_finetuned/
num_train_epochs=10
log_dir=logs/t5_text_visual/
log_path=logs/t5_text_visual/baseline.log
max_seq_length=128
max_vid_length=2048
visual_avg_pool_size=60
seed=42
wandb_name=t5_text_visual_sr-${visual_avg_pool_size}_mvl-${max_vid_length}_sd-${seed}

mkdir -p ${output_ckpt_dir}
mkdir -p ${log_dir}

python -m src t5_text_and_visual \
    --output_ckpt_dir ${output_ckpt_dir} \
    --num_train_epochs ${num_train_epochs} \
    --max_seq_length ${max_seq_length} \
    --seed ${seed} \
    --wandb_name ${wandb_name} \
    --max_vid_length ${max_vid_length} \
    --visual_avg_pool_size ${visual_avg_pool_size} \
    | tee ${log_path}
