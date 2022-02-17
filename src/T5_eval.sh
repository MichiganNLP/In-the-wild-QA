#!/usr/bin/env bash

data_dir=wildQA-data
ckpt_path=ckpts/T5/text_finetuned/epoch=6-train_loss=1.98.ckpt
pred_out_dir=preds/T5/text_finetuned/
pred_name=preds-T5_eval
pred_num=1
beam_size=5
CUDA_ID=0
batch_size=8

CUDA_VISIBLE_DEVICES=${CUDA_ID} python -W ignore -m src T5_eval \
    --train_data example_data/${data_dir}/train.json \
    --dev_data example_data/${data_dir}/dev.json \
    --test_data example_data/${data_dir}/test.json \
    --ckpt_path ${ckpt_path} \
    --batch_size ${batch_size} \
    --max_seq_length 512 \
    --pred_out_dir ${pred_out_dir} \
    --beam_size ${beam_size} \
    --pred_num ${pred_num}

python -m src.utils.post_process_T5 \
    --pred ${pred_out_dir}/${pred_name}-${pred_num}.txt \
    --processed_pred ${pred_out_dir}/processed_pred-${pred_num}.txt \
    --test_data example_data/${data_dir}/test.json
