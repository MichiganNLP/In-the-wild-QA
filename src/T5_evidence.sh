#!/bin/bash

# params needs to be modified
CUDA_ID=0


# params based on the params at the top
output_ckpt_dir=ckpts/T5/evidence_finetuned/
num_train_epochs=10
train_batch_size=2
eval_batch_size=2
log_dir=logs/T5_evidence/
log_path=logs/T5_evidence/baseline.log
gradient_accumulation_steps=16
path_to_visual=video_features/features/WildQA_I3D_avg_pool.hdf5
visual_size=1024
max_seq_length=128
max_vid_length=2048
sample_rate=1
seed=42
wandb_name=T5_evidence_sr-${sample_rate}_mvl-${max_vid_length}_sd-${seed}
wandb_entity=in-the-wild-vqa-um
wandb_project=trial

mkdir -p ${output_ckpt_dir}
mkdir -p ${log_dir}


CUDA_VISIBLE_DEVICES=${CUDA_ID} CUDA_LAUNCH_BLOCKING=1 python -W ignore -m src.main T5_evidence \
    --train_data example_data/wildQA-data/train.json \
    --dev_data example_data/wildQA-data/dev.json \
    --test_data example_data/wildQA-data/test.json \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --output_ckpt_dir ${output_ckpt_dir} \
    --num_train_epochs ${num_train_epochs} \
    --train_batch_size ${train_batch_size} \
    --eval_batch_size ${eval_batch_size} \
    --wandb_project ${wandb_project} \
    --wandb_name ${wandb_name} \
    --wandb_entity ${wandb_entity} \
    --path_to_visual_file ${path_to_visual} \
    --visual_size ${visual_size} \
    --max_seq_length ${max_seq_length} \
    --max_vid_length ${max_vid_length} \
    --sample_rate ${sample_rate} \
    --seed ${seed}

# > ${log_path}