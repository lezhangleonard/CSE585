import queue
from abstract_executor import AbstractExecutor


class SequentialExecutor(AbstractExecutor):
    def run(self):
        raise NotImplementedError

    def run_stream(self, data_queue: queue.Queue, sentinel, seq_dir=None, visualize: bool = False):
        self.store.clear()
        self.total_updates = 0
        self.num_no_ops = 0
        self.num_mutations = 0
        self.transaction_log = []
        self._start_timer()

        num_batches = 0
        global_update_idx = 0

        while True:
            batch_updates = data_queue.get()
            if batch_updates is sentinel:
                break

            for fact in batch_updates:
                update = {
                    "id": global_update_idx,
                    "subject": fact.s,
                    "predicate": fact.p,
                    "object": fact.o,
                }
                snapshot_before = self.store.snapshot()
                decision = self.reasoning_engine.reason(snapshot_before, update)
                self.store.apply(decision)
                snapshot_after = self.store.snapshot()
                key = (decision.subject, decision.predicate)
                self._record_decision(
                    decision,
                    snapshot_before.get(key),
                    snapshot_after.get(key),
                    batch_index=num_batches,
                    layer_index=0,
                )

                self.total_updates += 1
                if decision.action == "NO_OP":
                    self.num_no_ops += 1
                else:
                    self.num_mutations += 1
                global_update_idx += 1

            num_batches += 1
            data_queue.task_done()

        self._stop_timer()
        stats = self._compute_stats()
        stats["executor_type"] = "sequential_stream"
        stats["concurrency"] = 1
        stats["avg_parallel_width"] = 1.0
        stats["num_batches"] = num_batches

        return {
            "stats": stats,
            "transactions": self.transaction_log,
            "final_state": self.store.get_final_state(),
        }