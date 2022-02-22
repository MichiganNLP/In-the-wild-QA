#!/usr/bin/env bash

pred_num=5
data_dir=wildQA-data

python -m src random_evidence \
    --train_data example_data/${data_dir}/train.json \
    --dev_data example_data/${data_dir}/dev.json \
    --test_data example_data/${data_dir}/test.json \
    --pred_num ${pred_num}
