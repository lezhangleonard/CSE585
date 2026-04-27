#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:a40:1
#SBATCH --account=eecs542w26s001_class
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source ~/.bashrc
#Update conda environment here if necessary
conda activate akg
module load cuda
# Add tokens here (if required)
export HF_TOKEN=""
export GEMINI_API_KEY=""
export HF_HOME=/scratch/engin_root/engin1/arshiv/ml/hf_cache

export MODEL_PATH="/scratch/engin_root/engin1/arshiv/ml/hf_models/qwen2.5-7b-instruct"

mkdir -p logs

echo "Starting vLLM server..."
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.75 \
    --max-num-seqs 64 \
    --enforce-eager \
    --port 8000 &

VLLM_PID=$!

echo "Waiting for server to become active..."
while ! curl -s http://localhost:8000/v1/models > /dev/null; do
    sleep 10
done
echo "vLLM Server is ONLINE!"

# Workloads
WORKLOADS=(
    "workloads/real/w_5_hot_0.1.json"
    "workloads/synthetic/w_5_hot_0.1.json"
)

# Run with vllm + memory store, resolve_pronouns=false, verbose=false
python run_experiment.py \
    --backend vllm \
    --store memory \
    --workloads "${WORKLOADS[@]}"
    # --resolve_pronouns    # commented out = False
    # --verbose             # commented out = False


kill $VLLM_PID
echo "Job complete."

# If you want to run with Neo4j, make sure to start the server before running the experiment and stop it afterward. For example:
# ./neo4j_server/bin/neo4j start
# python run_experiment.py ...args...
# ./neo4j_server/bin/neo4j stop
