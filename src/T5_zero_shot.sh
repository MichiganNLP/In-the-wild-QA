#!/bin/bash

pred_out_dir=preds/T5/text_zero_shot
pred_num=1
beam_size=5
CUDA_ID=0
batch_size=8

CUDA_VISIBLE_DEVICES=${CUDA_ID} python -W ignore -m src.main T5_zero_shot \
    --train_data example_data/train.json \
    --dev_data example_data/dev.json \
    --test_data example_data/test.json \
    --batch_size ${batch_size} \
    --max_seq_length 512 \
    --pred_out_dir ${pred_out_dir} \
    --beam_size ${beam_size} \
    --pred_num ${pred_num}

# python -m utils.post_process_T5 \
#     --pred ${pred_out_dir}/preds-${pred_num}.txt \
#     --processed_pred ${pred_out_dir}/processed_pred-${pred_num}.txt \
#     --test_data example_data/test.json
