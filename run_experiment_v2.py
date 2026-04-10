import os
import json
import argparse
import threading
import queue

from updated_agent_extractor import SimpleAgenticPipeline 
from llm_reasoning_engine import LLMReasoningEngine
from store import InMemoryStore, Neo4jStore
from dag_executor import DAGExecutor
from vllm import LLM

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
    write_transactions(os.path.join(run_dir, "transactions.txt"), result["transactions"])
    write_final_state(os.path.join(run_dir, "final_state.txt"), result["final_state"])


SENTINEL = object()

def extraction_worker(raw_texts, extractor, batch_size, data_queue):
    print("[Producer] Extraction Thread Started.")
    
    for i in range(0, len(raw_texts), batch_size):
        chunk = raw_texts[i : i + batch_size]
        structured_chunk = extractor.process_stream(chunk)

        for fact in structured_chunk:
            print(f"S: {fact.s} | P: {fact.p} | O: {fact.o}")


        data_queue.put(structured_chunk)
        print(f"[Producer] Extracted batch {i//batch_size + 1}. Pushed to queue.")
    data_queue.put(SENTINEL)
    print("[Producer] Extraction Complete.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", type=str, required=True)
    parser.add_argument("--store", type=str, default="memory", choices=["memory", "neo4j"])
    parser.add_argument("--uri", type=str, default="bolt://localhost:7687")
    parser.add_argument("--user", type=str, default="neo4j")
    parser.add_argument("--password", type=str, default="mypwdmypwd")
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    shared_llm = LLM(
        model="/gpfs/accounts/engin_root/engin1/arshiv/hf_models/qwen2.5-1.5b",
        trust_remote_code=True, 
        gpu_memory_utilization=0.9, 
        max_model_len=4096, 
        enforce_eager=True, 
        max_num_seqs=16, 
        max_num_batched_tokens=8192
    )



    # 1. Load Raw Data
    with open(args.workload, "r") as f:
        raw_texts = json.load(f)

    print(f"\n{'='*60}")
    print(f"INITIALIZING PIPELINE (High VRAM Usage)")
    print(f"{'='*60}")
    
    extractor = SimpleAgenticPipeline(shared_llm)
    engine = LLMReasoningEngine(shared_llm)
    if args.store == "memory":
        store = InMemoryStore()
    else:
        store = Neo4jStore(args.uri, args.user, args.password)

    dag_dir = create_run_dir("dag", args.workload)
    dag_executor = DAGExecutor(
        extractor=None,
        store=store,
        reasoning_engine=engine,
        batch_size=args.batch_size
    )

    pq = queue.Queue(maxsize=3)

    producer = threading.Thread(
        target=extraction_worker, 
        args=(raw_texts, extractor, args.batch_size, pq)
    )
    producer.start()


    print("\n[Consumer] DAG Execution Engine Ready. Waiting for first batch...")

    dag_result = dag_executor.run_stream(
        data_queue=pq, 
        sentinel=SENTINEL, 
        dag_dir=dag_dir, 
        visualize=args.visualize
    )

    producer.join()
    log(dag_result, dag_dir)

    print(f"\n{'='*60}")
    print(f"EXPERIMENT COMPLETE: {args.workload}")
    print(f"{'='*60}")
    print(f"DAG Executor: Correctness=N/A (Ground truth skipped), "
          f"Parallel Width={dag_result['stats']['avg_parallel_width']:.2f}, "
          f"Throughput={dag_result['stats']['throughput_updates_per_sec']:.4f} up/sec")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()