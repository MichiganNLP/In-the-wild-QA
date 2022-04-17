#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

output_ckpt_dir=ckpts/t5/text_visual_finetuned/
num_train_epochs=10
train_batch_size=2
eval_batch_size=2
log_dir=logs/t5_text_visual/
log_path=logs/t5_text_visual/baseline.log
gradient_accumulation_steps=16
max_seq_length=128
max_vid_length=2048
visual_avg_pool_size=60
seed=42
wandb_name=t5_text_visual_sr-${visual_avg_pool_size}_mvl-${max_vid_length}_sd-${seed}
wandb_entity=in-the-wild-vqa-um

mkdir -p ${output_ckpt_dir}
mkdir -p ${log_dir}

python -m src t5_text_and_visual \
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
    --visual_avg_pool_size ${visual_avg_pool_size}

# > ${log_path}
