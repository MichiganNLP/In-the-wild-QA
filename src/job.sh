#!/bin/bash

# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=T5-text-visual
#SBATCH --mail-user=dnaihao@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --account=mihalcea0
#SBATCH --partition=gpu
#SBATCH --output=/home/%u/In-the-wild-QA/src/%x-%j.log

# The application(s) to execute along with its input arguments and options: 
source /etc/profile.d/http_proxy.sh
bash T5_visual_train.sh