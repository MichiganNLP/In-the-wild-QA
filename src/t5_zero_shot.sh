#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

python -m src t5_zero_shot --beam_size 5
