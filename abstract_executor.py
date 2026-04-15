import time
from abc import ABC, abstractmethod
from typing import List, Dict


class AbstractExecutor(ABC):
    def __init__(self, extractor, store, reasoning_engine):
        self.extractor = extractor
        self.store = store
        self.reasoning_engine = reasoning_engine

        self.total_updates = 0
        self.num_no_ops = 0
        self.num_mutations = 0
        self.start_time = None
        self.end_time = None
        self.transaction_log: List[Dict] = []

    def _record_decision(self, decision, before_value, after_value, **extra):
        record = {
            "update_id": decision.update_id,
            "subject": decision.subject,
            "predicate": decision.predicate,
            "requested_object": decision.requested_object,
            "previous_object": decision.previous_object,
            "before_value": before_value,
            "action": decision.action,
            "new_value": decision.new_value,
            "after_value": after_value,
        }
        record.update(extra)
        self.transaction_log.append(record)

    def _start_timer(self):
        self.start_time = time.perf_counter()

    def _stop_timer(self):
        self.end_time = time.perf_counter()

    def _compute_stats(self) -> dict:
        total_time = self.end_time - self.start_time if self.start_time is not None and self.end_time is not None else 0.0
        reasoner_time = getattr(self.reasoning_engine, "total_inference_time", 0.0)
        reasoner_total_decisions = getattr(self.reasoning_engine, "total_decisions", 0)
        reasoner_avg_time = reasoner_time / reasoner_total_decisions if reasoner_total_decisions > 0 else 0.0
        extractor_time = getattr(self.extractor, "total_inference_time", 0.0) if self.extractor is not None else 0.0
        throughput = self.total_updates / total_time if total_time > 0 else 0.0

        return {
            "total_updates": self.total_updates,
            "num_no_ops": self.num_no_ops,
            "num_mutations": self.num_mutations,
            "total_time_sec": total_time,
            "throughput_updates_per_sec": throughput,
            "final_state_size": len(self.store.get_final_state()),
            "reasoner_llm_time_sec": reasoner_time,
            "reasoner_total_decisions": reasoner_total_decisions,
            "reasoner_avg_time_per_decision_sec": reasoner_avg_time,
            "reasoner_num_llm_calls": getattr(self.reasoning_engine, "num_llm_calls", 0),
            "reasoner_json_parse_failures": getattr(self.reasoning_engine, "json_parse_failures", 0),
            "reasoner_invalid_action_count": getattr(self.reasoning_engine, "invalid_action_count", 0),
            "reasoner_fallback_decisions": getattr(self.reasoning_engine, "fallback_decisions", 0),
            "extractor_llm_time_sec": extractor_time,
            "extractor_num_llm_calls": getattr(self.extractor, "num_llm_calls", 0) if self.extractor is not None else 0,
            "extractor_json_parse_failures": getattr(self.extractor, "json_parse_failures", 0) if self.extractor is not None else 0,
            "extractor_num_sentences_processed": getattr(self.extractor, "num_sentences_processed", 0) if self.extractor is not None else 0,
        }

    @abstractmethod
    def run(self, workload: List[dict]):
        pass