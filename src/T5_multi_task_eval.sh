#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

ckpt_path=ckpts/T5/T5_multi_task/epoch=46-train_loss=0.11.ckpt
pred_out_dir=preds/T5/multi_task/
pred_name=preds-T5_multi_task_eval
pred_num=1
beam_size=5
batch_size=1
max_seq_length=128
max_vid_length=2048
path_to_visual=video_features/features/WildQA_I3D_avg_pool.hdf5
visual_avg_pool_size=1
data_dir=wildQA-data

python -m src T5_multi_task_eval \
    --ckpt_path ${ckpt_path} \
    --batch_size ${batch_size} \
    --max_seq_length ${max_seq_length} \
    --pred_out_dir ${pred_out_dir} \
    --beam_size ${beam_size} \
    --pred_num ${pred_num} \
    --max_vid_length ${max_vid_length} \
    --path_to_visual_file ${path_to_visual} \
    --visual_avg_pool_size ${visual_avg_pool_size}

python -m src.utils.post_process_T5_evidence \
    --pred ${pred_out_dir}/${pred_name}-evidence-${pred_num}.txt \
    --processed_pred ${pred_out_dir}/processed_pred-evidence-${pred_num}.txt \
    --test_data example_data/${data_dir}/test.json \
    --model_name T5_multi_task_evidence

python -m src.utils.post_process_T5 \
    --pred ${pred_out_dir}/${pred_name}-vqa-${pred_num}.txt \
    --processed_pred ${pred_out_dir}/processed_pred-vqa-${pred_num}.txt \
    --test_data example_data/${data_dir}/test.json \
    --model_name T5_multi_task_vqa
