#!/bin/bash

CUDA_ID=0


ckpt_path=ckpts/T5/evidence_IO_finetuned/epoch=38-train_loss=4.52.ckpt
pred_out_dir=preds/T5/evidence_IO/
pred_name=preds-T5_evidence_IO_eval
pred_num=1
beam_size=5
batch_size=1
visual_size=1024
max_seq_length=128
max_vid_length=2048
path_to_visual=video_features/features/WildQA_I3D_avg_pool.hdf5
sample_rate=1
data_dir=wildQA-data


CUDA_VISIBLE_DEVICES=${CUDA_ID} python -W ignore -m src T5_evidence_IO_eval \
    --train_data example_data/${data_dir}/train.json \
    --dev_data example_data/${data_dir}/dev.json \
    --test_data example_data/${data_dir}/test.json \
    --ckpt_path ${ckpt_path} \
    --batch_size ${batch_size} \
    --max_seq_length ${max_seq_length} \
    --pred_out_dir ${pred_out_dir} \
    --beam_size ${beam_size} \
    --pred_num ${pred_num} \
    --max_vid_length ${max_vid_length} \
    --path_to_visual_file ${path_to_visual} \
    --sample_rate ${sample_rate} \
    --visual_size ${visual_size}


python -m src.utils.post_process_T5_evidence \
    --pred ${pred_out_dir}/${pred_name}-${pred_num}.txt \
    --processed_pred ${pred_out_dir}/processed_pred-${pred_num}.txt \
    --test_data example_data/${data_dir}/test.json \
    --model_name T5_evidence_IO