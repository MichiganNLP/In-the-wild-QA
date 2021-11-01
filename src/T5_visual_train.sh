#!/bin/bash

# params needs to be modified
CUDA_ID=0


# params based on the params at the top
output_ckpt_dir=ckpts/T5/text_visual_finetuned/
num_train_epochs=10
train_batch_size=1
eval_batch_size=1
wandb_name=T5_text_visual
log_dir=logs/T5_text_visual/
log_path=logs/T5_text_visual/baseline.log
gradient_accumulation_steps=16
path_to_visual=example_data/LifeQA_I3D_avg_pool.hdf5
visual_size=1024
max_seq_length=128
max_vid_length=2048


mkdir -p ${output_ckpt_dir}
mkdir -p ${log_dir}


CUDA_VISIBLE_DEVICES=${CUDA_ID} python -W ignore -m src.main T5_text_and_visual\
    --train_data example_data/train.json \
    --dev_data example_data/dev.json \
    --test_data example_data/test.json \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --output_ckpt_dir ${output_ckpt_dir} \
    --num_train_epochs ${num_train_epochs} \
    --train_batch_size ${train_batch_size} \
    --eval_batch_size ${eval_batch_size} \
    --wandb_name ${wandb_name} \
    --path_to_visual_file ${path_to_visual} \
    --visual_size ${visual_size} \
    --max_seq_length ${max_seq_length} \
    --max_vid_length ${max_vid_length}
    
# > ${log_path}