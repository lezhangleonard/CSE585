import json
import time
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from vllm import SamplingParams


@dataclass(frozen=True)
class Decision:
    update_id: int
    subject: str
    predicate: str
    requested_object: Optional[str]
    previous_object: Optional[str]
    action: str
    new_value: Optional[str]


class LLMReasoningEngine:
    def __init__(self, llm, debug: bool = False, debug_limit: int = 6):
        self.llm = llm
        self.debug = debug
        self.debug_limit = debug_limit
        self.debug_count = 0

        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=96,
            stop=["<|eot_id|>", "<|end_of_text|>", "\n\n"]
        )

        self.total_inference_time = 0.0
        self.total_decisions = 0
        self.num_llm_calls = 0
        self.json_parse_failures = 0
        self.invalid_action_count = 0
        self.fallback_decisions = 0

    def reset(self):
        self.total_inference_time = 0.0
        self.total_decisions = 0
        self.num_llm_calls = 0
        self.json_parse_failures = 0
        self.invalid_action_count = 0
        self.fallback_decisions = 0
        self.debug_count = 0
        try:
            self.llm.llm_engine.free_unused_paged_cache()
        except AttributeError:
            pass

    def _get_subject_context(self, snapshot: Dict[Tuple[str, str], str], subject: str) -> str:
        relevant = [
            f"- ({s}, {p}) -> {o}"
            for (s, p), o in snapshot.items()
            if s == subject
        ]
        return "\n".join(relevant) if relevant else "NONE"

    def _format_prompt(self, snapshot: Dict[Tuple[str, str], str], update: Dict) -> str:
        s = update["subject"]
        p = update["predicate"]
        o = update["object"]
        current_value = snapshot.get((s, p))
        current_value_str = current_value if current_value is not None else "NONE"
        context_str = self._get_subject_context(snapshot, s)

        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a memory update classifier.

Return exactly one valid JSON object with these keys only:
- \"action\"
- \"reasoning\"

Allowed values for \"action\": \"APPLY\" or \"NO_OP\".

Decision rule:
- Return \"NO_OP\" only if the current value for the same (subject, predicate) already equals the incoming object.
- Return \"APPLY\" if no current value exists for that (subject, predicate).
- Return \"APPLY\" if the incoming object differs from the current value.

Do not output markdown.
Do not output code fences.
Do not output any text before or after the JSON object.
Keep \"reasoning\" short.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Current pair:
subject={s}
predicate={p}
current_value={current_value_str}

Incoming update:
subject={s}
predicate={p}
object={o}

Other known facts for this subject:
{context_str}

Return one JSON object only.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    def reason(self, snapshot: Dict, update: Dict) -> Decision:
        return self.reason_batch(snapshot, [update])[0]

    def reason_batch(self, snapshot: Dict, updates: List[Dict]) -> List[Decision]:
        start_time = time.perf_counter()
        prompts = [self._format_prompt(snapshot, up) for up in updates]
        current_values = [snapshot.get((up["subject"], up["predicate"])) for up in updates]

        if self.debug and self.debug_count < self.debug_limit:
            for up, current_val, prompt in zip(updates, current_values, prompts):
                print(f"[Reasoning] update={up['id']} current={current_val}")
                print(prompt)
                print("-" * 80)

        self.num_llm_calls += 1
        outputs = self.llm.generate(prompts, self.sampling_params, use_tqdm=False)
        decisions = []

        for i in range(len(updates)):
            decision = self._parse_vllm_output(outputs[i], updates[i], current_values[i])
            decisions.append(decision)

            if self.debug and self.debug_count < self.debug_limit:
                raw = outputs[i].outputs[0].text
                print(f"[Reasoning Output] update={updates[i]['id']}")
                print(f"  raw={repr(raw)}")
                print(f"  decision=({decision.action}, prev={decision.previous_object}, new={decision.new_value})")
                print("=" * 80)
                self.debug_count += 1

        self.total_inference_time += (time.perf_counter() - start_time)
        self.total_decisions += len(updates)
        return decisions

    def _extract_json_object(self, text: str) -> Optional[dict]:
        text = text.strip()
        if not text:
            return None

        candidates = [text]

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and start <= end:
            candidates.append(text[start:end + 1])

        if not text.startswith("{"):
            candidates.append("{" + text + "}")
            candidates.append("{" + text)
        if not text.endswith("}"):
            candidates.append(text + "}")

        seen = set()
        for cand in candidates:
            cand = cand.strip()
            if not cand or cand in seen:
                continue
            seen.add(cand)
            try:
                return json.loads(cand)
            except Exception:
                continue
        return None

    def _fallback_action(self, current_val: Optional[str], incoming_obj: str) -> str:
        if current_val is None:
            return "APPLY"
        if str(current_val).strip() == str(incoming_obj).strip():
            return "NO_OP"
        return "APPLY"

    def _parse_vllm_output(self, vllm_output, update: Dict, current_val: Optional[str]) -> Decision:
        generated_text = vllm_output.outputs[0].text.strip()
        parsed = self._extract_json_object(generated_text)

        used_fallback = False
        if parsed is not None:
            action = str(parsed.get("action", "")).strip().upper()
            if action not in ["APPLY", "NO_OP"]:
                self.invalid_action_count += 1
                action = self._fallback_action(current_val, update["object"])
                used_fallback = True
        else:
            self.json_parse_failures += 1
            print("[Reasoning Parse Error] Could not parse model output as JSON")
            print("[Reasoning Raw]", repr(generated_text))
            action = self._fallback_action(current_val, update["object"])
            used_fallback = True

        if used_fallback:
            self.fallback_decisions += 1

        new_value = current_val
        if action == "APPLY":
            new_value = update["object"]

        return Decision(
            update_id=update["id"],
            subject=update["subject"],
            predicate=update["predicate"],
            requested_object=update["object"],
            previous_object=current_val,
            action=action,
            new_value=new_value
        )