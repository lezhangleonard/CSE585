import json
import time
from dataclasses import dataclass
from typing import List, Dict, Optional
from vllm import SamplingParams


@dataclass
class ExtractedFact:
    id: int
    s: str
    p: str
    o: str
    source_sentence: str
    resolved_sentence: str
    simplified_sentence: str


class SimpleAgenticPipeline:
    def __init__(self, llm, context_window: int = 5, resolve_pronouns: bool = True, debug: bool = False):
        self.llm = llm
        self.context_window = context_window
        self.resolve_pronouns = resolve_pronouns
        self.debug = debug
        self.resolved_memory = []

        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=160,
            stop=["<|im_end|>", "<|eot_id|>", "<|end_of_text|>"]
        )

        self.total_inference_time = 0.0
        self.num_llm_calls = 0
        self.json_parse_failures = 0
        self.num_sentences_processed = 0
        self.num_resolution_calls = 0
        self.num_resolution_skipped = 0

    def reset(self):
        self.resolved_memory = []
        self.total_inference_time = 0.0
        self.num_llm_calls = 0
        self.json_parse_failures = 0
        self.num_sentences_processed = 0
        self.num_resolution_calls = 0
        self.num_resolution_skipped = 0

    def _build_resolve_prompt(self, text: str) -> str:
        system_instruction = """You are a precise NLP system for pronoun resolution.
Resolve pronouns in the target sentence using the supplied context.
Return exactly one valid JSON object with keys:
- \"description_of_changes\"
- \"resolved_target_sentence\"
Do not output any text before or after the JSON object."""
        return f"""<|im_start|>system
{system_instruction}<|im_end|>
<|im_start|>user
Input: {json.dumps(text)}
Output:<|im_end|>
<|im_start|>assistant
"""

    def _build_extract_prompt(self, text: str) -> str:
        system_instruction = """You are an information extraction system for a knowledge graph.
Return exactly one valid JSON object with keys:
- \"simplified_sentence\"
- \"subject\"
- \"predicate\"
- \"object\"
Rules:
1. Simplify to the core relation.
2. Extract subject, predicate, object from the simplified sentence.
3. Keep important qualifiers inside the object.
4. If input starts with \"Demo: \" followed by a synthetic triple, extract it verbatim without rewriting.
Do not output any text before or after the JSON object."""
        return f"""<|im_start|>system
{system_instruction}<|im_end|>
<|im_start|>user
Input: {json.dumps(text)}
Output:<|im_end|>
<|im_start|>assistant
"""

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

    def _generate_json(self, prompt: str) -> dict:
        start_time = time.perf_counter()
        self.num_llm_calls += 1
        outputs = self.llm.generate([prompt], self.sampling_params, use_tqdm=False)
        self.total_inference_time += time.perf_counter() - start_time

        response_text = outputs[0].outputs[0].text.strip()
        parsed = self._extract_json_object(response_text)
        if parsed is None:
            self.json_parse_failures += 1
            print(f"[Extractor Parse Warning] Failed to parse JSON: {repr(response_text)}")
            return {}
        return parsed

    def process_stream(self, sentences: List[Dict]) -> List[ExtractedFact]:
        extracted_facts = []

        for item in sentences:
            target_id = item.get("id")
            target = item.get("text", "")

            context = " ".join(self.resolved_memory[-self.context_window:])
            text_input = f"Context:\n{context}\n\nTarget sentence:\n{target}\n"

            if self.resolve_pronouns:
                resolve_prompt = self._build_resolve_prompt(text_input)
                resolve_data = self._generate_json(resolve_prompt)
                resolved_sentence = resolve_data.get("resolved_target_sentence", target)
                self.num_resolution_calls += 1
            else:
                resolved_sentence = target
                self.num_resolution_skipped += 1

            self.resolved_memory.append(resolved_sentence)

            extract_prompt = self._build_extract_prompt(resolved_sentence)
            extract_data = self._generate_json(extract_prompt)

            fact = ExtractedFact(
                id=target_id,
                s=extract_data.get("subject", ""),
                p=extract_data.get("predicate", ""),
                o=extract_data.get("object", ""),
                source_sentence=target,
                resolved_sentence=resolved_sentence,
                simplified_sentence=extract_data.get("simplified_sentence", "")
            )
            extracted_facts.append(fact)
            self.num_sentences_processed += 1

            if self.debug:
                print(f"[Extractor] id={fact.id}")
                print(f"  source:    {fact.source_sentence}")
                print(f"  resolved:  {fact.resolved_sentence}")
                print(f"  simplified:{fact.simplified_sentence}")
                print(f"  triple:    ({fact.s}, {fact.p}, {fact.o})")

        return extracted_facts