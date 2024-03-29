#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

output_ckpt_dir=ckpts/t5/evidence_finetuned/
num_train_epochs=50
train_batch_size=16
eval_batch_size=16
log_dir=logs/t5_evidence/
log_path=logs/t5_evidence/baseline.log
max_seq_length=512
max_vid_length=2048
visual_avg_pool_size=1
seed=42
wandb_name=t5_evidence_sr-${visual_avg_pool_size}_mvl-${max_vid_length}_sd-${seed}

mkdir -p ${output_ckpt_dir}
mkdir -p ${log_dir}

python -m src t5_evidence \
    --output_ckpt_dir ${output_ckpt_dir} \
    --num_train_epochs ${num_train_epochs} \
    --train_batch_size ${train_batch_size} \
    --eval_batch_size ${eval_batch_size} \
    --wandb_name ${wandb_name} \
    --max_seq_length ${max_seq_length} \
    --max_vid_length ${max_vid_length} \
    --visual_avg_pool_size ${visual_avg_pool_size} \
    --seed ${seed}

# > ${log_path}
