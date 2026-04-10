import json
import re
from abc import ABC, abstractmethod

class TripleExtract(ABC):

    @abstractmethod
    def extract(self, item: dict) -> dict:
        pass

class MockExtract(TripleExtract):

    def extract(self, item: dict) -> dict:

        template_type = item["metadata"]["template_type"]
        subject = item["metadata"]["subject"]
        predicate = item["metadata"]["predicate"]
        obj = item["metadata"]["object"]
        
        return {
            "id": item["id"],
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "template_type": template_type
        }

class LlamaExtract(TripleExtract):

    def __init__(self, client, model_name: str):
        self.client = client
        self.model_name = model_name

    def extract(self, item: dict) -> dict:

        prompt = f"""
Extract a knowledge graph triple from the sentence below.
Return STRICT JSON with keys:
"subject", "predicate", "object", "template_type"

Sentence:
"{item["text"]}"

Return JSON only.
"""

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        content = response.choices[0].message.content.strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON returned by LLM:\n{content}")



def build_extractor(mode="mock", client=None, model_name=None):
    """
    Factory method to switch between extractors.
    """

    if mode == "mock":
        return MockExtract()

    elif mode == "llama":
        if client is None or model_name is None:
            raise ValueError("Llama mode requires client and model_name.")
        return LlamaExtract(client, model_name)

    else:
        raise ValueError(f"Unknown extractor mode: {mode}")