#!/usr/bin/env bash

#SBATCH --job-name=t5_multi_task_train
#SBATCH --mail-user=dnaihao@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --account=mihalcea0
#SBATCH --partition=spgpu
#SBATCH --output=/home/%u/In-the-wild-QA/src/%x-%j.log

source /etc/profile.d/http_proxy.sh
bash t5_multi_task_train.sh
