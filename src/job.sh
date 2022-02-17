#!/usr/bin/env bash

# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=visual_feature_extraction
#SBATCH --mail-user=dnaihao@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --account=mihalcea0
#SBATCH --partition=gpu
#SBATCH --output=/home/%u/In-the-wild-QA/src/%x-%j.log

# The application(s) to execute along with its input arguments and options: 
# source /etc/profile.d/http_proxy.sh
bash extract_features.sh