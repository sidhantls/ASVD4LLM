#!/bin/bash

# Grid Engine options (lines prefixed with #$)
# Runtime limit of 10 hour:
#$ -l h_rt=5:00:00
#
# Set working directory to the directory where the job is submitted from:
#$ -cwd
#
# Request one GPU in the gpu queue:
#$ -q gpu 
#$ -pe gpu-a100 1
#
# Request 4 GB system RAM 
# the total system RAM available to the job is the value specified here multiplied by 
# the number of requested GPUs (above)
#$ -l h_vmem=120G

# Initialise the environment modules and load CUDA version 11.0.2
. /etc/profile.d/modules.sh
module load cuda
#module load cuda/12.1.1
module load anaconda 
conda config --add envs_dirs /exports/eddie/scratch/s2593541/anaconda/envs
conda config --add pkgs_dirs /exports/eddie/scratch/s2593541/anaconda/pkgs
conda activate lrd2
nvidia-smi
export HF_DATASETS_CACHE="/exports/eddie/scratch/s2593541/cache/lm_eval"
export TOKENIZERS_PARALLELISM=false


python asvd.py --model_id=meta-llama/Llama-2-7b-chat-hf --cache_dir=/exports/eddie/scratch/s2593541/lrd/cache_train_llama2 --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --param_ratio_target 0.9 --use_cache 
