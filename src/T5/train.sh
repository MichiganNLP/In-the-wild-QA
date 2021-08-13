#!/bin/bash

# params needs to be modified
SPLIT=question-split
DB=geography
CUDA_ID=0


# params based on the params at the top
data_dir=dataset/${SPLIT}/${DB}
output_dir=ckpt/${SPLIT}/${DB}
wandb_name=${DB}-${SPLIT}
log_path=logs/${DB}-${SPLIT}.log
mkdir -p ${output_dir}


CUDA_VISIBLE_DEVICES=${CUDA_ID} python -W ignore train.py \
    --data_dir ${data_dir} \
    --output_dir ${output_dir} \
    --wandb_name ${wandb_name} > ${log_path}