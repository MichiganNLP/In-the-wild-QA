#!/usr/bin/env bash

output_ckpt_dir=ckpts/clip_decoder/text_visual_source_trained/
num_train_epochs=100
train_batch_size=8
eval_batch_size=16
log_dir=logs/clip_decoder/
log_path=logs/clip_decoder/source_training_parallel_baseline.log
max_seq_length=16
max_vid_length=2048
seed=42
wandb_name=t5_clip_decoder-source-train-parallel-_mvl-${max_vid_length}_sd-${seed}

mkdir -p ${output_ckpt_dir}
mkdir -p ${log_dir}

python -m src clip_decoder \
    --train_data_path example_data/tvqa-data/train.json \
    --dev_data_path example_data/tvqa-data/val.json \
    --test_data_path example_data/tvqa-data/val.json \
    --frames_path video_features/tvqa_frames_hq \
    --is_tvqa \
    --n_gpu 4 \
    --accelerator ddp \
    --output_ckpt_dir ${output_ckpt_dir} \
    --num_train_epochs ${num_train_epochs} \
    --train_batch_size ${train_batch_size} \
    --eval_batch_size ${eval_batch_size} \
    --max_seq_length ${max_seq_length} \
    --seed ${seed} \
    --wandb_name ${wandb_name} \
    --max_vid_length ${max_vid_length}

# > ${log_path}
