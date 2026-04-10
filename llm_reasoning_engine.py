import json
import time
from typing import Dict, Tuple, Optional, List
from vllm import LLM, SamplingParams
from reasoning_engine import Decision

class LLMReasoningEngine:
    def __init__(self, llm):
        self.llm = llm
        self.sampling_params = SamplingParams(temperature=0, max_tokens=256, stop=["}", "<|eot_id|>", "\n\n"])
        self.total_inference_time = 0.0
        self.total_decisions = 0

    def reset(self):
        self.total_inference_time = 0.0
        self.total_decisions = 0
        try:
            self.llm.llm_engine.free_unused_paged_cache() 
        except AttributeError:
            pass
    
    def _get_subject_context(self, snapshot: Dict[Tuple[str, str], str], subject: str) -> str:
        """Extracts 1-hop neighborhood for the subject."""
        relevant = [f"- {s} {p} {o}" for (s, p), o in snapshot.items() if s == subject]
        return "\n".join(relevant) if relevant else "No current knowledge."
    
    def _format_prompt(self, context_str: str, update: Dict) -> str:
        s, p, o, t = update["subject"], update["predicate"], update["object"], update["template_type"]
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
                You are the Semantic Memory Controller for an AI Agent. Your only job is to evaluate updates and return JSON. 
                Do not write code. Do not explain yourself outside of the JSON.

                Current Memory Context:
                {context_str}
                <|eot_id|><|start_header_id|>user<|end_header_id|>
                Incoming Fact:
                ({s}, {p}, {o})

                Guidelines:
                - If the incoming fact is already known (even if phrased differently), return 'NO_OP'.
                - If the incoming fact contradicts, updates, or adds entirely new information, return 'APPLY'.

                Return JSON: {{"action": "APPLY" | "NO_OP", "reasoning": "..."}}
                <|eot_id|><|start_header_id|>assistant<|end_header_id|>
                {{"""
    
    def reason(self, snapshot: Dict, update: Dict) -> Decision:
        return self.reason_batch(snapshot, [update])[0]
    
    def reason_batch(self, snapshot: Dict, updates: List[Dict]) -> List[Decision]:
        start_time = time.perf_counter()
        
        # Build all prompts
        prompts = []
        current_values = []
        for up in updates:
            context = self._get_subject_context(snapshot, up["subject"])
            prompts.append(self._format_prompt(context, up))
            current_values.append(snapshot.get((up["subject"], up["predicate"])))

        # Execute Batch Inference
        outputs = self.llm.generate(prompts, self.sampling_params)
        
        # Parse all outputs
        decisions = [
            self._parse_vllm_output(outputs[i], updates[i], current_values[i])
            for i in range(len(updates))
        ]

        self.total_inference_time += (time.perf_counter() - start_time)
        self.total_decisions += len(updates)
        return decisions
    
    def _parse_vllm_output(self, vllm_output, update: Dict, current_val: Optional[str]) -> Decision:
        generated_text = vllm_output.outputs[0].text
        full_json_str = "{" + generated_text
        if not full_json_str.endswith("}"):
            full_json_str += "}"

        try:
            res = json.loads(full_json_str)
            action = res.get("action", "NO_OP")
            if action not in ["APPLY", "NO_OP"]:
                action = "NO_OP"
        except Exception as e:
            action = "NO_OP"

        new_value = current_val
        if action == "APPLY":
            new_value = update["object"]

        return Decision(
            update_id=update["id"],
            subject=update["subject"],
            predicate=update["predicate"],
            template_type=update["template_type"],
            requested_object=update["object"],
            previous_object=current_val,
            action=action,
            new_value=new_value
        )
    
    
    

   

# def test_single_update():
#     engine = LLMReasoningEngineV2()
    
#     # Simulate an empty memory
#     snapshot = {} 
    
#     # Update: Trying to START a relation
#     update = {
#         "id": 1,
#         "subject": "Shivam",
#         "predicate": "lives_in",
#         "object": "Ann Arbor",
#         "template_type": "START"
#     }
    
#     print("Testing START on empty memory...")
#     decision = engine.reason(snapshot, update)
#     print(f"Action: {decision.action} | Reasoning: {decision.new_value}")

#     # Now simulate memory HAS the fact
#     snapshot[("Shivam", "lives_in")] = "Ann Arbor"
    
#     # Update: Trying to START the same thing (should be NO_OP)
#     print("\nTesting START on existing memory (Conflict)...")
#     decision_conflict = engine.reason(snapshot, update)
#     print(f"Action: {decision_conflict.action} | New Value: {decision_conflict.new_value}")

# if __name__ == "__main__":
#     test_single_update()

