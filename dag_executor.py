import concurrent.futures
from typing import List
from abstract_executor import AbstractExecutor
from dependency_analyzer import DependencyAnalyzer
from execution_planner import ExecutionPlanner
import os
import queue

class DAGExecutor(AbstractExecutor):
    def __init__(self, extractor, store, reasoning_engine, batch_size: int = 50):
        super().__init__(None, store, reasoning_engine)
        self.batch_size = batch_size
        self.analyzer = DependencyAnalyzer()
        self.planner = ExecutionPlanner()

    def run():
        raise NotImplementedError

    def run_stream(self, data_queue:queue.Queue, sentinel: List[dict], dag_dir, visualize: bool = False):
        self.store.clear()
        self.total_updates = 0
        self.num_no_ops = 0
        self.num_mutations = 0
        self.transaction_log = []
        total_layers = 0
        batch_idx = 0
        
        self._start_timer()

        while True:
            batch_updates = data_queue.get()

            if batch_updates is sentinel:
                break

            edges = self.analyzer.analyze(batch_updates)
            layers = self.planner.plan_layers(len(batch_updates), edges)
            total_layers += len(layers)

            if visualize and batch_idx == 0:
                self.planner.visualize_dag(len(batch_updates), edges, os.path.join(dag_dir, "dag_viz.png"))

            for layer in layers:
                snapshot_before = self.store.snapshot()
                layer_updates = [batch_updates[idx] for idx in layer]
                
                layer_updates_dicts = []
                for idx, fact in zip(layer, layer_updates):
                    layer_updates_dicts.append({
                        "id": (batch_idx * self.batch_size) + idx, 
                        "subject": fact.s,
                        "predicate": fact.p,
                        "object": fact.o
                    })

                batch_decisions = self.reasoning_engine.reason_batch(snapshot_before, layer_updates_dicts)

                for decision in batch_decisions:
                    self.store.apply(decision)

                snapshot_after = self.store.snapshot()
                for decision in batch_decisions:
                    key = (decision.subject, decision.predicate)
                    self._record_decision(decision, snapshot_before.get(key), snapshot_after.get(key))
                    
                    self.total_updates += 1
                    if decision.action == "NO_OP":
                        self.num_no_ops += 1
                    else:
                        self.num_mutations += 1

            batch_idx += 1
            data_queue.task_done()

        self._stop_timer()
        stats = self._compute_stats()
        stats["avg_parallel_width"] = self.total_updates / total_layers if total_layers > 0 else 0
        stats["executor_type"] = "dag_stream"
        
        return {
            "stats": stats,
            "transactions": self.transaction_log,
            "final_state": self.store.get_final_state()
        }