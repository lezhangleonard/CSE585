## 1. Acquire Great Lakes Interactive GPU Session
Some dependencies will not install without a CUDA environment (only exists on compute nodes)
```bash
# On Great Lakes
srun --account=<account_name> --partition=gpu --gres=gpu:1 --mem 16G --time=01:00:00 --pty bash -i
```

## 2. Environment Setup

Create and activate a dedicated virtual environment to manage your dependencies.

```bash
# Using Conda
module load python3.10-anaconda
conda create -n experiment_env python=3.10 -y
conda activate experiment_env
```

## 3. Install Dependencies

Install the required Python packages using the provided `requirements.txt` file.

```bash
pip install -r requirements.txt
```

## 4. Neo4j Installation

This project requires **Neo4j 5.26**.

1. **Download:** Obtain the Neo4j 5.26 Community or Enterprise edition.
2. **Installation Path:** Install or extract the server into the `/neo4j_server` directory within the project root.
3. **Start Server:**
   ```bash
   wget https://dist.neo4j.org/neo4j-community-5.26.0-unix.tar.gz
   ./neo4j_server/bin/neo4j start
   ```

## 5. Qwen Model Setup

The experiment uses the Qwen2.5-7b-instruct model from Hugging Face.

```bash
# Log into Hugging Face and download the model
hf auth login
hf download Qwen/Qwen2.5-1.5B
```
Save the download path and update the line in run_test.sh:
```bash
export MODEL_PATH=<qwen_model_path>
```

## 6. Usage

### A. Generate Workloads
Use `workload_gen.py` to create the JSON workload files.

```bash
python workload_gen.py --n <number_of_lines> --hot <hot_ratio> --out <output_file.json>
```
* **Example:** `python workload_gen.py --n 1000 --hot 0.2 --out workload_1.json`
* You can also use `gen.sh`. There are already existing workloads in workloads/ to be used.

### B. Run Experiments
Execute the main experiment script.

```bash
bash run_test.sh
```

This script runs the workloads with the following configuration:

- `--backend vllm`  
  Chooses the inference backend: `vllm` uses the local vLLM server for generation, while `hf` would use Hugging Face’s `transformers` directly.

- `--store memory`  
  Uses an in‑memory key‑value store for the knowledge graph. Using `neo4j` instead would persist state in a Neo4j graph database.

- `--visualize_dag` (disabled by omission)  
  Controls whether to generate and save DAG visualizations during execution.

- `--verbose` (disabled by omission)  
  Enables extra logging from the extraction and executor threads.

- `--resolve_pronouns` (disabled by omission)  
  Enables pronoun coreference resolution in the fact‑extraction pipeline. When omitted, sentences are processed without explicit pronoun resolution.

- `--workloads ...`  
  Specifies the JSON workload files (e.g., `workloads/real/w_5_hot_0.1.json` and `workloads/synthetic/w_5_hot_0.1.json`) that list the raw text inputs to process.

Make sure `vllm` is installed in the conda/Pip environment used by `run_test.sh` and that `MODEL_PATH` points to your model directory before running.
