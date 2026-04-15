#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:a40:1
#SBATCH --account=eecs542w26s001_class
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err


# To run on great lakes
#Step 1: salloc --account=ece567w26_class --partition=spgpu --gres=gpu:a40:1 --time=01:00:00 --mem=32G
#Step 2: bash run_test.sh

source ~/.bashrc
conda activate akg
module load cuda
# Add tokens here (if required)
export HF_TOKEN=""
export GEMINI_API_KEY=""
export HF_HOME=/scratch/engin_root/engin1/arshiv/ml/hf_cache
export MODEL_HOME=/scratch/engin_root/engin1/arshiv/ml/hf_models

mkdir -p logs

# Workload Generation

# SIZES=(5 20 100 1000)
# HOT_RATIOS=(0.1 0.5 0.8 0.95)

# for n in "${SIZES[@]}"; do
#   for hot in "${HOT_RATIOS[@]}"; do
#     out_file="workloads/pilot/w_${n}_hot_${hot}.json"
#     python workload_gen.py --n $n --hot $hot --out $out_file
#   done
# done

# E2E Run

python run_experiment_v2.py
# echo "All workloads generated."
# ./neo4j_server/bin/neo4j start
# python run_experiment.py --workload workloads/w_20_hot_0.8.json --store memory
# ./neo4j_server/bin/neo4j stop
