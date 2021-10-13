#!/bin/bash

ckpt_path=T5/ckpt/epoch=0-val_loss=8.81.ckpt
pred_out_dir=T5/preds
pred_num=1
beam_size=5
CUDA_ID=0
batch_size=8

CUDA_VISIBLE_DEVICES=${CUDA_ID} python -W ignore main.py T5_eval \
    --train_data example_data/train.json \
    --dev_data example_data/dev.json \
    --test_data example_data/test.json \
    --ckpt_path ${ckpt_path} \
    --batch_size ${batch_size} \
    --max_seq_length 512 \
    --pred_out_dir ${pred_out_dir} \
    --beam_size ${beam_size} \
    --pred_num ${pred_num}