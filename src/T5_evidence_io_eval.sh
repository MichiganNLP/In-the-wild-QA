#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

pred_out_dir=preds/T5/evidence_IO/
pred_name=preds-T5_evidence_IO_eval
pred_num=1
#ckpt_path=ckpts/T5/evidence_IO_finetuned/epoch=38-train_loss=4.52.ckpt
#beam_size=5
#batch_size=1
#max_seq_length=128
#visual_avg_pool_size=1

#python -m src T5_evidence_IO_eval \
#    --ckpt_path ${ckpt_path} \
#    --batch_size ${batch_size} \
#    --max_seq_length ${max_seq_length} \
#    --pred_out_dir ${pred_out_dir} \
#    --beam_size ${beam_size} \
#    --pred_num ${pred_num} \
#    --visual_avg_pool_size ${visual_avg_pool_size}

python -m src.utils.post_process_T5_evidence \
    --pred ${pred_out_dir}/${pred_name}-${pred_num}.txt \
    --processed_pred ${pred_out_dir}/processed_pred-${pred_num}.txt \
    --model_name T5_evidence_IO
