import queue
from abstract_executor import AbstractExecutor


class BatchExecutor(AbstractExecutor):
    def __init__(self, extractor, store, reasoning_engine, batch_size: int = 5):
        super().__init__(extractor, store, reasoning_engine)
        self.batch_size = batch_size

    def run(self):
        raise NotImplementedError

    def run_stream(self, data_queue: queue.Queue, sentinel, batch_dir=None, visualize: bool = False):
        self.store.clear()
        self.total_updates = 0
        self.num_no_ops = 0
        self.num_mutations = 0
        self.transaction_log = []
        self._start_timer()

        batch_idx = 0
        num_batches = 0
        max_batch_size_seen = 0

        while True:
            batch_updates = data_queue.get()
            if batch_updates is sentinel:
                break

            snapshot_before = self.store.snapshot()
            batch_updates_dicts = []
            for idx, fact in enumerate(batch_updates):
                batch_updates_dicts.append({
                    "id": (batch_idx * self.batch_size) + idx,
                    "subject": fact.s,
                    "predicate": fact.p,
                    "object": fact.o,
                })

            batch_decisions = self.reasoning_engine.reason_batch(snapshot_before, batch_updates_dicts)
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
                    layer_index=0,
                )
                self.total_updates += 1
                if decision.action == "NO_OP":
                    self.num_no_ops += 1
                else:
                    self.num_mutations += 1

            num_batches += 1
            max_batch_size_seen = max(max_batch_size_seen, len(batch_updates))
            batch_idx += 1
            data_queue.task_done()

        self._stop_timer()
        stats = self._compute_stats()
        stats["executor_type"] = "batch_stream"
        stats["batch_size"] = self.batch_size
        stats["concurrency"] = self.batch_size
        stats["avg_parallel_width"] = float(self.batch_size)
        stats["num_batches"] = num_batches
        stats["max_batch_size_seen"] = max_batch_size_seen

        return {
            "stats": stats,
            "transactions": self.transaction_log,
            "final_state": self.store.get_final_state(),
        }