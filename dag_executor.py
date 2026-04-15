import os
import queue
from abstract_executor import AbstractExecutor
from dependency_analyzer import DependencyAnalyzer
from execution_planner import ExecutionPlanner


class DAGExecutor(AbstractExecutor):
    def __init__(self, extractor, store, reasoning_engine, batch_size: int = 50):
        super().__init__(extractor, store, reasoning_engine)
        self.batch_size = batch_size
        self.analyzer = DependencyAnalyzer()
        self.planner = ExecutionPlanner()

    def run(self):
        raise NotImplementedError

    def run_stream(self, data_queue: queue.Queue, sentinel, dag_dir=None, visualize: bool = False):
        self.store.clear()
        self.total_updates = 0
        self.num_no_ops = 0
        self.num_mutations = 0
        self.transaction_log = []
        self._start_timer()

        total_layers = 0
        total_edges = 0
        total_layer_width = 0
        max_layer_width = 0
        num_batches = 0
        batch_idx = 0
        batch_summaries = []

        while True:
            batch_updates = data_queue.get()
            if batch_updates is sentinel:
                break

            edges = self.analyzer.analyze(batch_updates)
            layers = self.planner.plan_layers(len(batch_updates), edges)
            total_layers += len(layers)
            total_edges += len(edges)

            layer_widths = [len(layer) for layer in layers]
            total_layer_width += sum(layer_widths)
            max_layer_width = max(max_layer_width, max(layer_widths) if layer_widths else 0)

            batch_summaries.append({
                "batch_index": batch_idx,
                "batch_size": len(batch_updates),
                "num_edges": len(edges),
                "num_layers": len(layers),
                "layer_widths": layer_widths,
            })

            if visualize and batch_idx == 0 and dag_dir is not None:
                self.planner.visualize_dag(len(batch_updates), edges, os.path.join(dag_dir, "dag_viz.png"))

            for layer_idx, layer in enumerate(layers):
                snapshot_before = self.store.snapshot()
                layer_updates = [batch_updates[idx] for idx in layer]
                layer_updates_dicts = []
                for idx, fact in zip(layer, layer_updates):
                    layer_updates_dicts.append({
                        "id": (batch_idx * self.batch_size) + idx,
                        "subject": fact.s,
                        "predicate": fact.p,
                        "object": fact.o,
                    })

                batch_decisions = self.reasoning_engine.reason_batch(snapshot_before, layer_updates_dicts)
                for decision in batch_decisions:
                    self.store.apply(decision)

                snapshot_after = self.store.snapshot()
                for decision in batch_decisions:
                    key = (decision.subject, decision.predicate)
                    self._record_decision(
                        decision,
                        snapshot_before.get(key),
                        snapshot_after.get(key),
                        batch_index=batch_idx,
                        layer_index=layer_idx,
                    )
                    self.total_updates += 1
                    if decision.action == "NO_OP":
                        self.num_no_ops += 1
                    else:
                        self.num_mutations += 1

            num_batches += 1
            batch_idx += 1
            data_queue.task_done()

        self._stop_timer()
        stats = self._compute_stats()
        stats["executor_type"] = "dag_stream"
        stats["batch_size"] = self.batch_size
        stats["num_batches"] = num_batches
        stats["num_layers_total"] = total_layers
        stats["num_dependency_edges_total"] = total_edges
        stats["max_layer_width"] = max_layer_width
        stats["avg_layer_width"] = (total_layer_width / total_layers) if total_layers > 0 else 0.0
        stats["avg_parallel_width"] = (self.total_updates / total_layers) if total_layers > 0 else 0.0

        return {
            "stats": stats,
            "transactions": self.transaction_log,
            "final_state": self.store.get_final_state(),
            "batch_summaries": batch_summaries,
        }