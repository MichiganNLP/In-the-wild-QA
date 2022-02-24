#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

ckpt_path=ckpts/T5/text_visual_finetuned/epoch=2-train_loss=0.85.ckpt
pred_out_dir=preds/T5/text_visual/
pred_name=preds-T5_text_visual_eval
pred_num=1
beam_size=5
batch_size=1
max_seq_length=128
visual_avg_pool_size=60

python -m src T5_text_visual_eval \
    --ckpt_path ${ckpt_path} \
    --batch_size ${batch_size} \
    --max_seq_length ${max_seq_length} \
    --pred_out_dir ${pred_out_dir} \
    --beam_size ${beam_size} \
    --pred_num ${pred_num} \
    --visual_avg_pool_size ${visual_avg_pool_size}

python -m src.utils.post_process_T5 \
    --pred ${pred_out_dir}/${pred_name}-${pred_num}.txt \
    --processed_pred ${pred_out_dir}/processed_pred-${pred_num}.txt \
    --model_name T5_Text+Visual
