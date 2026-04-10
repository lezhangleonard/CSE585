import os
import json
import argparse
import gc
import torch
import time
from vllm.distributed.parallel_state import destroy_model_parallel


from extractor import MockExtract
from reasoning_engine import ReasoningEngine
from llm_reasoning_engine import LLMReasoningEngine
from store import InMemoryStore, Neo4jStore
from sequence_executor import SequentialExecutor
from batch_executor import BatchExecutor
from dag_executor import DAGExecutor


def write_stats(path, stats: dict):
    with open(path, "w") as f:
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")


def write_transactions(path, transactions: list):
    with open(path, "w") as f:
        for txn in transactions:
            f.write("-----\n")
            for k, v in txn.items():
                f.write(f"{k}: {v}\n")

def create_run_dir(executor_type, workload_path):
    workload_name = os.path.splitext(os.path.basename(workload_path))[0]
    run_dir = os.path.join("runs", workload_name, executor_type)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def write_final_state(path, final_state: dict):
    with open(path, "w") as f:
        for (s, p), o in final_state.items():
            f.write(f"({s}, {p}) -> {o}\n")


def log(result, run_dir):
    write_stats(os.path.join(run_dir, "stats.txt"), result["stats"])
    write_transactions(
        os.path.join(run_dir, "transactions.txt"),
        result["transactions"]
    )
    write_final_state(
        os.path.join(run_dir, "final_state.txt"),
        result["final_state"]
    )


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", type=str, required=True)
    parser.add_argument("--engine", type=str, default="deterministic", choices=["deterministic", "llm"])
    parser.add_argument("--store", type=str, default="memory", choices=["memory", "neo4j"])
    parser.add_argument("--uri", type=str, default="bolt://localhost:7687")
    parser.add_argument("--user", type=str, default="neo4j")
    parser.add_argument("--password", type=str, default="mypwdmypwd")
    parser.add_argument("--batch_size", type=str, default=5)
    parser.add_argument("--visualize", action="store_true", help="Generate DAG visualization for the first batch")

    args = parser.parse_args()

    with open(args.workload, "r") as f:
        workload = json.load(f)

    extractor = MockExtract()

    if args.store == "memory":
        store = InMemoryStore()
    else:
        store = Neo4jStore(args.uri, args.user, args.password)

    engine0 = LLMReasoningEngine()

    print("Starting Sequential...")
    seq_dir = create_run_dir("sequential", args.workload)
    seq_executor = SequentialExecutor(
        extractor=extractor,
        store=store,
        reasoning_engine=engine0
    )

    seq_result = seq_executor.run(workload)
    ground_truth = seq_result["final_state"]

    # batch_dir = create_run_dir("batch", args.workload)
    # batch_executor = BatchExecutor(
    #     extractor=extractor,
    #     store=store,
    #     reasoning_engine=reasoning_engine,
    #     batch_size=args.batch_size
    # )

    engine0.reset()
    print("Starting DAG...")
    dag_dir = create_run_dir("dag", args.workload)
    dag_executor = DAGExecutor(
        extractor=extractor,
        store=store,
        reasoning_engine=engine0,
        batch_size=args.batch_size
    )

    # batch_result = batch_executor.run(workload)
    # batch_final = batch_result["final_state"]

    
    dag_result = dag_executor.run(workload, visualize=args.visualize, dag_dir=dag_dir)
    dag_final = dag_result["final_state"]
    
    gt_items = set(ground_truth.items())
    # batch_items = set(batch_final.items())
    dag_items = set(dag_final.items())

    # batch_correct = len(gt_items & batch_items)
    # batch_total = len(gt_items | batch_items)
    # batch_correctness = batch_correct / batch_total if batch_total > 0 else 1.0
    # batch_result["stats"]["correctness"] = batch_correctness

    dag_correct = len(gt_items & dag_items)
    dag_total = len(gt_items | dag_items)
    dag_correctness = dag_correct / dag_total if dag_total > 0 else 1.0
    dag_result["stats"]["correctness"] = dag_correctness

    log(seq_result, seq_dir)
    # log(batch_result, batch_dir)
    log(dag_result, dag_dir)

    print(f"\n{'='*60}")
    print(f"EXPERIMENT COMPLETE: {args.workload}")
    print(f"{'='*60}")
    print(f"Sequantial Executor: Correctness={1:.4f}, Parallel Width={1}, Throughput={seq_result['stats']['throughput_updates_per_sec']:.4f}")
    # print(f"Batch Executor: Correctness={batch_correctness:.4f}, Parallel Width={args.batch_size}, Throughput={batch_result['stats']['throughput_updates_per_sec']:.4f}")
    print(f"DAG Executor:   Correctness={dag_correctness:.4f}, Parallel Width={dag_result['stats']['avg_parallel_width']:.2f}, Throughput={dag_result['stats']['throughput_updates_per_sec']:.4f}")
    print(f"{'='*60}\n")

    


if __name__ == "__main__":
    main()