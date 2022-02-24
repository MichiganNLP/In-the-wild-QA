#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

output_ckpt_dir=ckpts/T5/evidence_IO_finetuned/
num_train_epochs=50
train_batch_size=1
eval_batch_size=1
log_dir=logs/T5_evidence_IO/
log_path=logs/T5_evidence_IO/baseline.log
gradient_accumulation_steps=16
max_seq_length=512
max_vid_length=2048
visual_avg_pool_size=1
seed=42
wandb_name=T5_evidence_sr-${visual_avg_pool_size}_mvl-${max_vid_length}_sd-${seed}
wandb_entity=in-the-wild-vqa-um
wandb_project=trial

mkdir -p ${output_ckpt_dir}
mkdir -p ${log_dir}

python -m src T5_evidence_IO \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --output_ckpt_dir ${output_ckpt_dir} \
    --num_train_epochs ${num_train_epochs} \
    --train_batch_size ${train_batch_size} \
    --eval_batch_size ${eval_batch_size} \
    --wandb_project ${wandb_project} \
    --wandb_name ${wandb_name} \
    --wandb_entity ${wandb_entity} \
    --max_seq_length ${max_seq_length} \
    --max_vid_length ${max_vid_length} \
    --visual_avg_pool_size ${visual_avg_pool_size} \
    --seed ${seed}

# > ${log_path}
