#!/usr/bin/env bash

data_dir=wildQA-data

python -m src most_common_ans \
    --train_data example_data/${data_dir}/train.json \
    --dev_data example_data/${data_dir}/dev.json \
    --test_data example_data/${data_dir}/test.json
