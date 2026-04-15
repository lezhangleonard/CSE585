from typing import List, Dict, Tuple

class DependencyAnalyzer:
    """
    Identifies structural dependencies within a batch of updates.
    A dependency exists if two updates target the same (subject) key.
    """
    def analyze(self, updates: List[dict]) -> List[Tuple[int, int]]:
        edges = []
        # Maps subject -> index of the last update that touched ANY of its predicates
        last_subject_touch: Dict[str, int] = {}

        for current_idx, update in enumerate(updates):
            subj = update.s
            
            # If ANY previous update in this batch touched this person/thing,
            # we MUST wait for it to finish so the LLM has fresh context.
            if subj in last_subject_touch:
                prev_idx = last_subject_touch[subj]
                edges.append((prev_idx, current_idx))
                
            last_subject_touch[subj] = current_idx
            
        return edges