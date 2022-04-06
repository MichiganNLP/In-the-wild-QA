#!/usr/bin/env bash

# params needs to be modified
CUDA_ID=0

# params based on the params at the top
output_ckpt_dir=ckpts/T5/text_visual_finetuned/
num_train_epochs=10
train_batch_size=1
eval_batch_size=1
log_dir=logs/T5_text_visual/
log_path=logs/T5_text_visual/baseline.log
gradient_accumulation_steps=16
path_to_visual=/home/dnaihao/In-the-wild-QA/src/video_features/features/WildQA_I3D_avg_pool.hdf5
visual_size=1024
max_seq_length=128
max_vid_length=2048
sample_rate=60
seed=42
wandb_name=T5_violet_decoder-${sample_rate}_mvl-${max_vid_length}_sd-${seed}
wandb_entity=in-the-wild-vqa-um
data_dir=wildQA-data
n_gpu=1

mkdir -p ${output_ckpt_dir}
mkdir -p ${log_dir}

# PL_TORCH_DISTRIBUTED_BACKEND=gloo
CUDA_VISIBLE_DEVICES=${CUDA_ID} python -m src violet_decoder\
    --train_data example_data/${data_dir}/train.json \
    --dev_data example_data/${data_dir}/dev.json \
    --test_data example_data/${data_dir}/test.json \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --output_ckpt_dir ${output_ckpt_dir} \
    --num_train_epochs ${num_train_epochs} \
    --train_batch_size ${train_batch_size} \
    --eval_batch_size ${eval_batch_size} \
    --max_seq_length ${max_seq_length} \
    --n_gpu ${n_gpu} \
    --seed ${seed} \
    --wandb_offline \
    --wandb_name ${wandb_name} \
    --wandb_entity ${wandb_entity} \
    --path_to_visual_file ${path_to_visual} \
    --visual_size ${visual_size} \
    --max_vid_length ${max_vid_length} \
    --sample_rate ${sample_rate}
# > ${log_path}
