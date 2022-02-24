#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

pred_out_dir=preds/T5/text_zero_shot
pred_num=1
beam_size=5

python -m src T5_zero_shot \
    --max_seq_length 512 \
    --pred_out_dir ${pred_out_dir} \
    --beam_size ${beam_size} \
    --pred_num ${pred_num}

python -m src.utils.post_process_T5 \
    --pred ${pred_out_dir}/preds-T5_zero_shot-${pred_num}.txt \
    --processed_pred ${pred_out_dir}/processed_pred-${pred_num}.txt
