import json
from dataclasses import dataclass
from typing import List, Dict
import argparse
from vllm import SamplingParams

@dataclass
class ExtractedFact:
    """Structured output format matching the agentic framework."""
    id: int
    s: str
    p: str
    o: str
    source_sentence: str
    resolved_sentence: str
    simplified_sentence: str

class SimpleAgenticPipeline:
    def __init__(self, llm, context_window: int = 5):
        self.llm = llm
        self.context_window = context_window
        self.resolved_memory = []
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=150,
            stop=["}", "<|im_end|>"] # Stop generating immediately after JSON closes
        )

    def _build_resolve_prompt(self, text: str) -> list:
        system_instruction = """You are a precise NLP system for pronoun resolution.
Task:
- Resolve ALL pronouns in the target sentence using the provided context.
- Pronouns include: she, he, it, they, her, his, its, their, etc.
- You MUST replace every pronoun with its explicit referent noun.
- Do NOT leave any pronouns unresolved.

Rules:
- Use ONLY entities from the context.
- If multiple candidates exist, choose the most likely one.
- Preserve the original sentence meaning.

You MUST return ONLY valid JSON:
{
  "description_of_changes": "...",
  "resolved_target_sentence": "..."
}"""
        return f"""<|im_start|>system
        {system_instruction}<|im_end|>
        <|im_start|>user
        Input: "{text}"
        Output:<|im_end|>
        <|im_start|>assistant
        {{"""

    def _build_extract_prompt(self, text: str) -> list:
        system_instruction = """You are an expert Information Extraction system for a Knowledge Graph.
            Your task is to take a complex sentence, simplify it into its core relational meaning, and then extract the triple.

            Follow these strict steps:
            1. Simplify the original sentence to its most basic relationship.
            2. Extract the subject, predicate, and object ONLY from that simplified sentence.
            3. If the sentence contains additional context like locations, dates, or institutions, combine them into the 'object' string instead of deleting them.
            4. SPECIAL OVERRIDE: If the input sentence is strictly in the format "Demo: [Subject] [Predicate] [Object]" (e.g., "Demo: Sub_2 Pred_1 Obj_1"), bypass all simplification rules. Directly extract those exact three words as the subject, predicate, and object.

            You MUST return exactly this JSON structure and nothing else:
            {
              "simplified_sentence": "...",
              "subject": "...",
              "predicate": "...",
              "object": "..."
            }

            ---
            EXAMPLE 1 (THIS IS ONLY AN EXAMPLE):
            Input: "Demo: Sub_2 Pred_1 Obj_1."
            Output:
            {
              "simplified_sentence": "Demo: Sub_2 Pred_1 Obj_1.",
              "subject": "Sub_2",
              "predicate": "Pred_1",
              "object": "Obj_1"
            }"""
        return f"""<|im_start|>system
        {system_instruction}<|im_end|>
        <|im_start|>user
        Input: "{text}"
        Output:<|im_end|>
        <|im_start|>assistant
        {{"""

    def _generate_json(self, prompt: str) -> dict:
        """Utility wrapper to handle vLLM inference and JSON cleaning."""
        outputs = self.llm.generate([prompt], self.sampling_params, use_tqdm=False)
        
        response_text = outputs[0].outputs[0].text.strip()

        full_json_str = "{" + response_text
        if not full_json_str.endswith("}"):
            full_json_str += "}"

        try:
            return json.loads(full_json_str)
        except json.JSONDecodeError:
            print(f"Warning: Failed to parse JSON.\n{full_json_str}")
            return {}

    def process_stream(self, sentences: List[Dict]) -> List[ExtractedFact]:
        """Main pipeline loop bridging memory, resolution, and extraction."""
        extracted_facts = []

        for item in sentences:
            target_id = item.get("id")
            target = item.get("text", "")
            
            context = " ".join(self.resolved_memory[-self.context_window:])
            text_input = f"Context:\n{context}\n\nTarget sentence:\n{target}\n"

            resolve_prompt = self._build_resolve_prompt(text_input)
            resolve_data = self._generate_json(resolve_prompt)
            resolved_sentence = resolve_data.get("resolved_target_sentence", target)

            self.resolved_memory.append(resolved_sentence)

            extract_prompt = self._build_extract_prompt(resolved_sentence)
            extract_data = self._generate_json(extract_prompt)

            fact = ExtractedFact(
                id = target_id,
                s=extract_data.get("subject", ""),
                p=extract_data.get("predicate", ""),
                o=extract_data.get("object", ""),
                source_sentence=target,
                resolved_sentence=resolved_sentence,
                simplified_sentence=extract_data.get("simplified_sentence", "")
            )
            extracted_facts.append(fact)

        return extracted_facts

# --- Usage Example ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", type=str, required=True)
    args = parser.parse_args()

    # 1. Load Raw Data
    with open(args.workload, "r") as f:
        raw_texts = json.load(f)
    
    # Initialize pipeline
    extractor = SimpleAgenticPipeline()
    
    # Process stream
    results = extractor.process_stream(raw_texts)
    
    # Clean Dataclass output
    for fact in results:
        print(f"S: {fact.s} | P: {fact.p} | O: {fact.o}")