#!/bin/bash

# Grid Engine options (lines prefixed with #$)
# Runtime limit of 10 hour:
#$ -l h_rt=12:00:00
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
conda activate lrd3_new
nvidia-smi
export HF_DATASETS_CACHE="/exports/eddie/scratch/s2593541/cache/lm_eval"
export TOKENIZERS_PARALLELISM=false

MODEL=meta-llama/Meta-Llama-3-8B
CACHE_DIR=/exports/eddie/scratch/s2593541/lrd/cache_train_llama_8b
USE_BOS=false 

#MODEL=meta-llama/Llama-2-13b-hf
#CACHE_DIR=/exports/eddie/scratch/s2593541/lrd/cache_train_llama13
#USE_BOS=false 

#MODEL=google/gemma-7b
#CACHE_DIR=/exports/eddie/scratch/s2593541/lrd/cache_train_llama_gemma
#USE_BOS=true

EVAL_BS=4
COMP_VALUES=(0.90 0.85 0.80)

# Loop over the COMP values
for i in ${!COMP_VALUES[@]}; do
    COMP=${COMP_VALUES[$i]}
    EXP_NAME="strs_${MODEL#*/}_${COMP}"

    # Check if it's the first iteration
    if [ $i -eq 0 ]; then
        # Command for the first iteration without extra arguments
        python asvd.py --model_id=$MODEL --eval_bs=$EVAL_BS --cache_dir=$CACHE_DIR --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --param_ratio_target $COMP --exp_name=$EXP_NAME
    else
        python asvd.py --model_id=$MODEL --eval_bs=$EVAL_BS --cache_dir=$CACHE_DIR --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --param_ratio_target $COMP --exp_name=$EXP_NAME --use_cache

    fi
done


#EXP_NAME=llama13_strs_asvd_85_full
#python asvd.py --model_id=$MODEL --eval_bs=4 --cache_dir=/exports/eddie/scratch/s2593541/lrd/cache_train_llama3 --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --param_ratio_target 0.85 --exp_name=$EXP_NAME

#EXP_NAME=strs_asvd_80_full
#python asvd.py --model_id=meta-llama/Llama-2-7b-hf --eval_bs=8 --cache_dir=/exports/eddie/scratch/s2593541/lrd/cache_train_llama3 --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --param_ratio_target 0.80 --exp_name=$EXP_NAME --use_cache

#EXP_NAME=strs_asvd_70
#python asvd.py --model_id=meta-llama/Llama-2-7b-hf --cache_dir=/exports/eddie/scratch/s2593541/lrd/cache_train_llama3 --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --param_ratio_target 0.70 --exp_name=$EXP_NAME --use_cache

#EXP_NAME=strs_asvd_50
#`#python asvd.py --model_id=meta-llama/Llama-2-7b-hf --cache_dir=/exports/eddie/scratch/s2593541/lrd/cache_train_llama3 --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --param_ratio_target 0.50 --exp_name=$EXP_NAME --use_cache
