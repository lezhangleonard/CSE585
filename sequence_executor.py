from typing import List
from abstract_executor import AbstractExecutor


class SequentialExecutor(AbstractExecutor):

    def run(self, workload: List[dict]):
        """
        Executes workload strictly sequentially.
        Snapshot is taken per update.
        This defines canonical ground truth.
        """

        # Reset store and metrics
        self.store.clear()
        self.total_updates = 0
        self.num_no_ops = 0
        self.num_mutations = 0
        self.transaction_log = []

        # Start timing
        self._start_timer()

        for item in workload:
            # Extract structured update
            structured = self.extractor.extract(item)
            snapshot = self.store.snapshot()
            self._process_update(snapshot, structured)

        self._stop_timer()

        stats = self._compute_stats()

        # Sequential is canonical correctness
        stats["correctness"] = 1.0
        stats["concurrency"] = 1
        stats["executor_type"] = "sequential"

        return {
            "stats": stats,
            "transactions": self.transaction_log,
            "final_state": self.store.get_final_state()
        }