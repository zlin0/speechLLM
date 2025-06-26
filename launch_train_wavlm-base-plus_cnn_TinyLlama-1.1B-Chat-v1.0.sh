#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16000
#SBATCH --job-name=cnn_train_sllm #job name
#SBATCH --nodes=1  #number of nodes requested
#SBATCH --gpus=1  #number of gpus requested
#SBATCH --partition=gpu-a100   #queue
#SBATCH --account=a100acct
#SBATCH --mail-user="lzhan268@jh.edu"  #email for reporting
#SBATCH --mail-type=END,FAIL  #report types
#SBATCH --error=logs/train_wavlm-base-plus_cnn_TinyLlama-1.1B-Chat-v1.0.%j.log
#SBATCH --output=logs/train_wavlm-base-plus_cnn_TinyLlama-1.1B-Chat-v1.0.%j.log
#SBATCH --exclude=e03

echo `date`

source ~/.bashrc
export HF_HOME=./hf_cache/
export HF_DATASETS_CACHE=./hf_cache/
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'
export TORCH_DISTRIBUTED_DEBUG=DETAIL # 如果用到分布式，可看详细挂起栈
export PYTHONUNBUFFERED=1             # 确保 print/log 不被缓存，立即输出

conda activate /home/tthebau1/miniconda3/envs/speechllm
python3 train.py \
    --encoder 'microsoft/wavlm-base-plus' \
    --connector 'cnn' \
    --llm 'TinyLlama-1.1B-Chat-v1.0' \
    --batch-size 128
