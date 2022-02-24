#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

ckpt_path=ckpts/T5/text_finetuned/epoch=5-train_loss=0.36.ckpt
pred_out_dir=preds/T5/text_finetuned/
pred_name=preds-T5_eval
pred_num=1
beam_size=5

python -m src T5_eval \
    --ckpt_path ${ckpt_path} \
    --max_seq_length 512 \
    --pred_out_dir ${pred_out_dir} \
    --beam_size ${beam_size} \
    --pred_num ${pred_num}

python -m src.utils.post_process_T5 \
    --pred ${pred_out_dir}/${pred_name}-${pred_num}.txt \
    --processed_pred ${pred_out_dir}/processed_pred-${pred_num}.txt
