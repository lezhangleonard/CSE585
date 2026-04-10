
from typing import List
from abstract_executor import AbstractExecutor


class BatchExecutor(AbstractExecutor):

    def __init__(self, extractor, store, reasoning_engine, batch_size: int = 5):
        super().__init__(extractor, store, reasoning_engine)
        self.batch_size = batch_size

    def run(self, workload: List[dict]):
        """
        Naive batching:
        - Snapshot once per batch
        - All updates reason on that snapshot
        - Apply immediately after reasoning (using stale snapshot)
        """

        # Reset store and metrics
        self.store.clear()
        self.total_updates = 0
        self.num_no_ops = 0
        self.num_mutations = 0
        self.transaction_log = []

        self._start_timer()

        for i in range(0, len(workload), self.batch_size):

            batch = workload[i:i + self.batch_size]

            # One snapshot for entire batch
            snapshot = self.store.snapshot()

            for item in batch:
                structured = self.extractor.extract(item)
                self._process_update(snapshot, structured)

        self._stop_timer()

        stats = self._compute_stats()
        stats["executor_type"] = "batch"
        stats["batch_size"] = self.batch_size
        stats["concurrency"] = self.batch_size
        stats["correctness"] = None

        return {
            "stats": stats,
            "transactions": self.transaction_log,
            "final_state": self.store.get_final_state()
        }