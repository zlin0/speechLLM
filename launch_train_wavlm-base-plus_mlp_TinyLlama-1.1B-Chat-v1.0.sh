#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16000
#SBATCH --job-name=mlp_train_sllm #job name
#SBATCH --nodes=1  #number of nodes requested
#SBATCH --gpus=1  #number of gpus requested
#SBATCH --partition=gpu-a100   #queue
#SBATCH --account=a100acct
#SBATCH --mail-user="lzhan268@jh.edu"  #email for reporting
#SBATCH --mail-type=END,FAIL  #report types
#SBATCH --error=logs/train_wavlm-base-plus_mlp1_TinyLlama-1.1B-Chat-v1.0.%j.log
#SBATCH --output=logs/train_wavlm-base-plus_mlp1_TinyLlama-1.1B-Chat-v1.0.%j.log

#SBATCH --exclude=e03

source ~/.bashrc
export HF_HOME=./hf_cache/
export HF_DATASETS_CACHE=./hf_cache/

echo `date`

conda activate /home/tthebau1/miniconda3/envs/speechllm

python3 train.py \
    --encoder 'microsoft/wavlm-base-plus' \
    --connector 'mlp' \
    --llm 'TinyLlama-1.1B-Chat-v1.0' \
    --connector-k '1' 
