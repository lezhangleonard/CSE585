#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:a40:1
#SBATCH --account=ece567w26_class
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source ~/.bashrc
conda activate experiment_env
module load cuda
# Add tokens here (if required)
export HF_TOKEN=""
export GEMINI_API_KEY=""
export MODEL_HOME="/home/dimashu/hf_models/Qwen2.5-1.5B"

mkdir -p logs

echo "Starting vLLM server..."
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_HOME \
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

python run_experiment_v2.py --mode synthetic

kill $VLLM_PID
echo "Job complete."

# echo "All workloads generated."
# ./neo4j_server/bin/neo4j start
# python run_experiment.py --workload workloads/w_20_hot_0.8.json --store memory
# ./neo4j_server/bin/neo4j stop
