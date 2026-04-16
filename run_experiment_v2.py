import os
import json
import threading
import queue
import copy
import torch
import argparse

from updated_agent_extractor import SimpleAgenticPipeline
from llm_reasoning_engine import LLMReasoningEngine
from store import InMemoryStore, Neo4jStore
from sequence_executor import SequentialExecutor
from batch_executor import BatchExecutor
from dag_executor import DAGExecutor
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/scratch/engin_root/engin1/arshiv/ml/hf_models/qwen2.5-7b-instruct"
BACKEND = "hf"  # "hf" or "vllm"
STORE_TYPE = "memory"  # "memory" or "neo4j"
VISUALIZE_DAG = True
VERBOSE = False
RESOLVE_PRONOUNS = False


SENTINEL = object()


class LLMInterface:
    def generate(self, prompts, sampling_params=None, use_tqdm=False):
        raise NotImplementedError


class HFBackend(LLMInterface):
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.stop_sequences = ["<|eot_id|>"]

    def _apply_stop(self, text):
        cut_positions = []
        for stop in self.stop_sequences:
            pos = text.find(stop)
            if pos != -1:
                cut_positions.append(pos)
        if cut_positions:
            return text[:min(cut_positions)]
        return text

    def generate(self, prompts, sampling_params=None, use_tqdm=False):
        max_new_tokens = 256
        if sampling_params is not None and hasattr(sampling_params, "max_tokens"):
            max_new_tokens = sampling_params.max_tokens

        texts = [
            self.tokenizer.apply_chat_template(
                p,
                tokenize=False,
                add_generation_prompt=True
            )
            for p in prompts
        ]

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        class HFOutput:
            def __init__(self, text):
                self.outputs = [type("Out", (), {"text": text})()]

        wrapped = []
        for i in range(len(prompts)):
            prompt_len = inputs["attention_mask"][i].sum().item()
            gen_tokens = outputs[i][prompt_len:]

            if len(gen_tokens) == 0:
                text = ""
            else:
                text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)

            text = self._apply_stop(text)
            wrapped.append(HFOutput(text))

        return wrapped

class VLLMBackend(LLMInterface):
    def __init__(self, model_path):
        self.llm = LLM(
            model=model_path,
            trust_remote_code=True,
            gpu_memory_utilization=0.6,
            max_model_len=4096,
            max_num_seqs=8,
            max_num_batched_tokens=4096,
            enforce_eager=False,
        )

    def generate(self, prompts, sampling_params=None, use_tqdm=False):
        if sampling_params is None:
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=256,
                stop=["<|eot_id|>"],
            )

        texts = [
            self.llm.get_tokenizer().apply_chat_template(
                p,
                tokenize=False,
                add_generation_prompt=True
            )
            for p in prompts
        ]

        return self.llm.generate(texts, sampling_params, use_tqdm=use_tqdm)


def build_llm():
    if BACKEND == "hf":
        return HFBackend(MODEL_PATH)
    if BACKEND == "vllm":
        return VLLMBackend(MODEL_PATH)
    raise ValueError(f"Unknown backend: {BACKEND}")


def create_store():
    if STORE_TYPE == "memory":
        return InMemoryStore()
    return Neo4jStore("bolt://localhost:7687", "neo4j", "mypwdmypwd")


def create_run_dir(executor_type, workload_path, batch_ratio, RUN_TYPE):
    workload_name = os.path.splitext(os.path.basename(workload_path))[0]
    run_dir = os.path.join("eval_runs", RUN_TYPE, f"br_{batch_ratio}", workload_name, executor_type)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def write_text_stats(path, stats: dict):
    with open(path, "w") as f:
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def write_transactions_txt(path, transactions: list):
    with open(path, "w") as f:
        for txn in transactions:
            f.write("-----\n")
            for k, v in txn.items():
                f.write(f"{k}: {v}\n")


def write_jsonl(path, rows: list):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def write_final_state_txt(path, final_state: dict):
    with open(path, "w") as f:
        for (s, p), o in final_state.items():
            f.write(f"({s}, {p}) -> {o}\n")


def serialize_final_state(final_state: dict):
    return [{"subject": s, "predicate": p, "object": o} for (s, p), o in final_state.items()]


def serialize_extracted_facts(extracted_facts):
    return [
        {
            "id": fact.id,
            "subject": fact.s,
            "predicate": fact.p,
            "object": fact.o,
            "source_sentence": fact.source_sentence,
            "resolved_sentence": fact.resolved_sentence,
            "simplified_sentence": fact.simplified_sentence,
        }
        for fact in extracted_facts
    ]


def compute_final_state_metrics(reference_state: dict, candidate_state: dict):
    ref_items = set(reference_state.items())
    cand_items = set(candidate_state.items())
    intersection = len(ref_items & cand_items)
    union = len(ref_items | cand_items)
    return {
        "final_state_exact_match": ref_items == cand_items,
        "correctness": (intersection / union) if union > 0 else 1.0,
        "final_state_intersection": intersection,
        "final_state_union": union,
        "reference_final_state_size": len(reference_state),
        "candidate_final_state_size": len(candidate_state),
    }


def extraction_worker(raw_texts, extractor, batch_size, data_queue, extracted_fact_sink=None, verbose=False):
    if verbose:
        print("[Producer] Extraction thread started.")
    for i in range(0, len(raw_texts), batch_size):
        chunk = raw_texts[i:i + batch_size]
        if verbose:
            print(f"[Producer] Extracting chunk starting at index {i}, size={len(chunk)}")
        structured_chunk = extractor.process_stream(chunk)
        if extracted_fact_sink is not None:
            extracted_fact_sink.extend(copy.deepcopy(structured_chunk))
        if verbose:
            for fact in structured_chunk:
                print(f"S: {fact.s} | P: {fact.p} | O: {fact.o}")
            print(f"[Producer] Queue size before put: {data_queue.qsize()}")
        data_queue.put(structured_chunk)
        if verbose:
            print(f"[Producer] Pushed batch {(i // batch_size) + 1} to queue. Queue size now: {data_queue.qsize()}")
    data_queue.put(SENTINEL)
    if verbose:
        print("[Producer] Extraction complete. Sentinel pushed.")


def run_single_executor(executor_name, executor, raw_texts, extractor, batch_size, run_dir):
    pq = queue.Queue(maxsize=3)
    extracted_facts = []
    producer = threading.Thread(
        target=extraction_worker,
        args=(raw_texts, extractor, batch_size, pq, extracted_facts, VERBOSE),
    )
    producer.start()

    if executor_name == "dag":
        result = executor.run_stream(pq, SENTINEL, dag_dir=run_dir, visualize=VISUALIZE_DAG)
    elif executor_name == "batch":
        result = executor.run_stream(pq, SENTINEL, batch_dir=run_dir, visualize=False)
    else:
        result = executor.run_stream(pq, SENTINEL, seq_dir=run_dir, visualize=False)

    producer.join()
    return result, extracted_facts


def persist_run(result, run_dir, extracted_facts, run_config):
    write_text_stats(os.path.join(run_dir, "stats.txt"), result["stats"])
    write_json(os.path.join(run_dir, "stats.json"), result["stats"])
    write_transactions_txt(os.path.join(run_dir, "transactions.txt"), result["transactions"])
    write_jsonl(os.path.join(run_dir, "transactions.jsonl"), result["transactions"])
    write_final_state_txt(os.path.join(run_dir, "final_state.txt"), result["final_state"])
    write_json(os.path.join(run_dir, "final_state.json"), serialize_final_state(result["final_state"]))
    write_jsonl(os.path.join(run_dir, "extracted_facts.jsonl"), serialize_extracted_facts(extracted_facts))
    write_json(os.path.join(run_dir, "run_config.json"), run_config)
    if "batch_summaries" in result:
        write_json(os.path.join(run_dir, "batch_summaries.json"), result["batch_summaries"])


def run_workload(shared_llm, workload_path, RUN_TYPE):
    with open(workload_path, "r") as f:
        raw_texts = json.load(f)


    workload_size = len(raw_texts)
    batches = [4, 8, 16]

    for bs in batches:
        batch_size = 1 + (workload_size // bs)
        print(f"\n{'=' * 60}")
        print(f"RUNNING WORKLOAD: {workload_path}")
        print(f"backend={BACKEND}, batch_size={batch_size}, store={STORE_TYPE}")
        print(f"{'=' * 60}")

        executor_order = ["sequential", "batch", "dag"]
        results = {}
        reference_final_state = None

        for executor_name in executor_order:
            extractor = SimpleAgenticPipeline(shared_llm, debug=VERBOSE, resolve_pronouns=RESOLVE_PRONOUNS)
            engine = LLMReasoningEngine(shared_llm, debug=VERBOSE)
            store = create_store()

            if executor_name == "sequential":
                executor = SequentialExecutor(extractor=extractor, store=store, reasoning_engine=engine)
            elif executor_name == "batch":
                executor = BatchExecutor(extractor=extractor, store=store, reasoning_engine=engine, batch_size=batch_size)
            else:
                executor = DAGExecutor(extractor=extractor, store=store, reasoning_engine=engine, batch_size=batch_size)

            run_dir = create_run_dir(executor_name, workload_path, bs, RUN_TYPE)
            run_config = {
                "workload": workload_path,
                "backend": BACKEND,
                "store": STORE_TYPE,
                "batch_size": batch_size,
                "executor": executor_name,
                "visualize_dag": VISUALIZE_DAG,
                "resolve_pronouns": RESOLVE_PRONOUNS,
            }

            print(f"Starting {executor_name}...")
            result, extracted_facts = run_single_executor(
                executor_name=executor_name,
                executor=executor,
                raw_texts=raw_texts,
                extractor=extractor,
                batch_size=batch_size,
                run_dir=run_dir,
            )

            if executor_name == "sequential":
                reference_final_state = result["final_state"]
                result["stats"]["correctness"] = 1.0
                result["stats"]["final_state_exact_match"] = True
                result["stats"]["reference_final_state_size"] = len(reference_final_state)
                result["stats"]["candidate_final_state_size"] = len(reference_final_state)
                result["stats"]["final_state_intersection"] = len(reference_final_state)
                result["stats"]["final_state_union"] = len(reference_final_state)
            else:
                result["stats"].update(compute_final_state_metrics(reference_final_state, result["final_state"]))

            persist_run(result, run_dir, extracted_facts, run_config)
            results[executor_name] = result

            print(
                f"{executor_name}: correctness={result['stats']['correctness']:.4f}, "
                f"throughput={result['stats']['throughput_updates_per_sec']:.4f}, "
                f"mutations={result['stats']['num_mutations']}, "
                f"no_ops={result['stats']['num_no_ops']}"
            )

    return results


def main():
    print(f"\n{'=' * 60}")
    print("LOADING MODEL ONCE")
    print(f"backend={BACKEND}, model={MODEL_PATH}")
    print(f"{'=' * 60}")
    shared_llm = build_llm()

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["synthetic", "real"], required=True)
    args = parser.parse_args()

    WORKLOADS = [
        f"workloads/{args.mode}/w_5_hot_0.1.json",
        f"workloads/{args.mode}/w_5_hot_0.5.json",
        f"workloads/{args.mode}/w_5_hot_0.8.json",
        f"workloads/{args.mode}/w_5_hot_0.95.json",
        f"workloads/{args.mode}/w_20_hot_0.1.json",
        f"workloads/{args.mode}/w_20_hot_0.5.json",
        f"workloads/{args.mode}/w_20_hot_0.8.json",
        f"workloads/{args.mode}/w_20_hot_0.95.json",
        f"workloads/{args.mode}/w_100_hot_0.1.json",
        f"workloads/{args.mode}/w_100_hot_0.5.json",
        f"workloads/{args.mode}/w_100_hot_0.8.json",
        f"workloads/{args.mode}/w_100_hot_0.95.json",
        f"workloads/{args.mode}/w_500_hot_0.1.json",
        f"workloads/{args.mode}/w_500_hot_0.5.json",
        f"workloads/{args.mode}/w_500_hot_0.8.json",
        f"workloads/{args.mode}/w_500_hot_0.95.json",
        f"workloads/{args.mode}/w_1000_hot_0.1.json",
        f"workloads/{args.mode}/w_1000_hot_0.5.json",
        f"workloads/{args.mode}/w_1000_hot_0.8.json",
        f"workloads/{args.mode}/w_1000_hot_0.95.json",
    ]

    all_results = {}
    for workload_path in WORKLOADS:
        all_results[workload_path] = run_workload(shared_llm, workload_path, args.mode)

    print(f"\n{'=' * 60}")
    print("ALL WORKLOADS COMPLETE")
    print(f"num_workloads={len(WORKLOADS)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
