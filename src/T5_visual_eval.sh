#!/usr/bin/env bash

ckpt_path=ckpts/T5/text_visual_finetuned/epoch=5-train_loss=1.98.ckpt
pred_out_dir=preds/T5/text_visual/
pred_name=preds-T5_text_visual_eval
pred_num=1
beam_size=5
CUDA_ID=0
batch_size=1
visual_size=1024
max_seq_length=128
max_vid_length=2048
path_to_visual=/home/dnaihao/In-the-wild-QA/src/video_features/features/WildQA_I3D_avg_pool.hdf5
data_dir=wildQA-data
sample_rate=60

CUDA_VISIBLE_DEVICES=${CUDA_ID} python -W ignore -m src T5_text_visual_eval \
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

python -m src.utils.post_process_T5 \
    --pred ${pred_out_dir}/${pred_name}-${pred_num}.txt \
    --processed_pred ${pred_out_dir}/processed_pred-${pred_num}.txt \
    --test_data example_data/${data_dir}/test.json \
    --model_name T5_Text+Visual
