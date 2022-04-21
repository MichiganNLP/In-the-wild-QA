#!/usr/bin/env bash

#SBATCH --job-name=clip_decoder_source_training_parallel
#SBATCH --mail-user=dnaihao@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --cpus-per-task=4
#SBATCH --mem=46G
#SBATCH --gres=gpu:4
#SBATCH --time=5-00:00:00
#SBATCH --account=mihalcea0
#SBATCH --partition=spgpu
#SBATCH --output=/home/%u/In-the-wild-QA/src/%x-%j.log

source /etc/profile.d/http_proxy.sh
bash clip_decoder_source_training.sh
