import time
from abc import ABC, abstractmethod
from typing import List, Dict
from reasoning_engine import Decision, ReasoningEngine

class AbstractExecutor(ABC):
    def __init__(self, extractor, store, reasoning_engine):
        self.extractor = extractor
        self.store = store
        self.reasoning_engine = reasoning_engine

        # Runtime metrics
        self.total_updates = 0
        self.num_no_ops = 0
        self.num_mutations = 0
        self.start_time = None
        self.end_time = None

        # Transaction trace
        self.transaction_log: List[Dict] = []
    
    def _process_update(self, snapshot: dict, structured_update: dict):
        """
        Performs read–decide–write for a single update.
        Snapshot must be supplied by subclass.
        """

        key = (structured_update["subject"], structured_update["predicate"])
        before_value = snapshot.get(key)

        decision = self.reasoning_engine.reason(snapshot, structured_update)

        self.store.apply(decision)

        # After apply, read actual store state
        after_snapshot = self.store.snapshot()
        after_value = after_snapshot.get(key)

        self._record_decision(decision, before_value, after_value)

        self.total_updates += 1

        if decision.action == "NO_OP":
            self.num_no_ops += 1
        else:
            self.num_mutations += 1

        return decision


    def _record_decision(self, decision: Decision, before_value, after_value):

        self.transaction_log.append({
            "update_id": decision.update_id,
            "subject": decision.subject,
            "predicate": decision.predicate,
            "template_type": decision.template_type,
            "requested_object": decision.requested_object,
            "previous_object": decision.previous_object,
            "before_value": before_value,
            "action": decision.action,
            "after_value": after_value
        })
    
    def _start_timer(self):
        self.start_time = time.perf_counter()

    def _stop_timer(self):
        self.end_time = time.perf_counter()
    
    def _compute_stats(self) -> dict:
        total_time = self.end_time - self.start_time
        


        total_inference_time = self.reasoning_engine.total_inference_time
        avg_time_per_inference = (total_inference_time / self.reasoning_engine.total_decisions if self.reasoning_engine.total_decisions > 0 else 0)

        overhead_time = total_time - total_inference_time

        throughput = (
            self.total_updates / total_time
            if total_time > 0 else 0
        )

        return {
            "total_updates": self.total_updates,
            "num_no_ops": self.num_no_ops,
            "num_mutations": self.num_mutations,
            "total_time_sec": total_time,
            "pure_inference_time": total_inference_time,
            "system_overhead_sec": overhead_time,
            "throughput_updates_per_sec": throughput,
            "total_decisions": self.reasoning_engine.total_decisions ,
            "avg_time_per_inference": avg_time_per_inference
        }
    
    @abstractmethod
    def run(self, workload: List[dict]):
        """
        Subclasses implement execution strategy.
        """
        pass