# Evaluation Methodology and Setup

This document details the experimental framework used to evaluate the trade-offs between execution speed (throughput) and data integrity (correctness)

## 1. Experimental Environment

### Hardware Configuration
The experiments were executed on a high-performance compute node with the following specifications:
* **GPU:** 1x NVIDIA A40 (48GB VRAM)
* **CPU:** 4 Cores
* **Memory:** 32GB RAM
* **Scheduler:** SLURM

### Software Stack
* **LLM Engine:** [vLLM](https://github.com/vllm-project/vllm) hosted locally via an OpenAI-compatible API.
* **Model:** `Qwen2.5-7b-instruct`
* **Storage:** Thread-safe `InMemoryStore` (State snapshotting enabled for Batch/DAG executors).
* **Dependency Logic:** NetworkX-based Directed Acyclic Graph (DAG) generation.

## 2. Experimental Matrix

We conducted a total of **64 independent runs** to stress-test the system across varied levels of data contention and scale.

| Parameter | Values |
| :--- | :--- |
| **Workload Types** | `Synthetic` (Abstract IDs), `Real` (Natural Entities) |
| **Workload Sizes ($N$)** | 50, 100, 200, 500 updates |
| **Hot Ratios** | 0.1, 0.5, 0.8, 0.95 (Frequency of "hot" subject access) |
| **Batch Ratios ($BR$)** | 4 ($0.25N$), 8 ($0.125N$) |
| **Executors** | `Sequential` (Baseline), `Batch` (Naive Parallel), `DAG` (Conflict-Aware) |

## 3. Key Metrics Defined

### 3.1 Correctness (State Integrity)
Correctness is measured using the Jaccard Similarity between the final state of a parallel executor and the sequential ground truth.
$$Correctness = \frac{|State_{Ref} \cap State_{Cand}|}{|State_{Ref} \cup State_{Cand}|}$$
Where $State_{Ref}$ is the result of the `SequentialExecutor`.

### 3.2 Throughput
The total number of Knowledge Graph updates processed per second.
$$Throughput = \frac{Total Updates}{Total Execution Time (sec)}$$

### 3.3 Parallel Efficiency (DAG Only)
* **Average Parallel Width:** The average number of updates executed simultaneously per reasoning layer.
* **Critical Path Length:** The number of sequential layers required to clear a batch.

## 4. Execution Workflow

1. **Extraction:** Raw text is passed to the Extraction LLM to generate $(S, P, O)$ triples. Pronoun resolution is disabled for this benchmark to isolate reasoning performance.
2. **Batching:** Updates are grouped into chunks based on the `Batch Ratio`.
3. **Execution:**
   - **Sequential:** Processes updates 1-by-1.
   - **Batch:** Sends all updates in a chunk to the LLM simultaneously using a single state snapshot.
   - **DAG:** Analyzes subject-level dependencies, builds a dependency graph, and executes non-conflicting updates in parallel layers.
4. **Logging:** Stats are serialized to `stats.json` and full transaction traces to `transactions.jsonl` for post-hoc analysis.