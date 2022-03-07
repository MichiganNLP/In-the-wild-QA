#!/usr/bin/env bash

data_dir=wildQA-data
pred_out_dir=preds/T5/text_zero_shot
pred_num=1
beam_size=5
CUDA_ID=0
batch_size=8

CUDA_VISIBLE_DEVICES=${CUDA_ID} python -m src T5_zero_shot \
    --train_data example_data/${data_dir}/train.json \
    --dev_data example_data/${data_dir}/dev.json \
    --test_data example_data/${data_dir}/test.json \
    --batch_size ${batch_size} \
    --max_seq_length 512 \
    --pred_out_dir ${pred_out_dir} \
    --beam_size ${beam_size} \
    --pred_num ${pred_num}

python -m src.utils.post_process_T5 \
    --pred ${pred_out_dir}/preds-T5_zero_shot-${pred_num}.txt \
    --processed_pred ${pred_out_dir}/processed_pred-${pred_num}.txt \
    --test_data example_data/${data_dir}/test.json
