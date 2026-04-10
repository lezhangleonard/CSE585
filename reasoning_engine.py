from dataclasses import dataclass
from typing import Optional, Dict, Tuple


@dataclass(frozen=True)
class Decision:
    """
    Immutable decision returned by the reasoning engine.
    """

    update_id: int
    subject: str
    predicate: str
    template_type: str

    requested_object: Optional[str]
    previous_object: Optional[str]

    action: str               # "APPLY" | "NO_OP"
    new_value: Optional[str]  # object or None


class ReasoningEngine:
    """
    Pure semantic lifecycle engine.

    Takes:
        - snapshot: Dict[(subject, predicate)] -> active_object
        - structured update dict

    Returns:
        - Decision object

    No side effects.
    No store access.
    No logging.
    """

    def reason(
        self,
        snapshot: Dict[Tuple[str, str], Optional[str]],
        update: Dict
    ) -> Decision:

        uid = update["id"]
        s = update["subject"]
        p = update["predicate"]
        o = update["object"]
        t = update["template_type"]

        current = snapshot.get((s, p))

        if t == "START":
            if current is None:
                return Decision(
                    update_id=uid,
                    subject=s,
                    predicate=p,
                    template_type=t,
                    requested_object=o,
                    previous_object=current,
                    action="APPLY",
                    new_value=o
                )
            else:
                return Decision(
                    update_id=uid,
                    subject=s,
                    predicate=p,
                    template_type=t,
                    requested_object=o,
                    previous_object=current,
                    action="NO_OP",
                    new_value=current
                )

        # --- SWITCH ---
        elif t == "SWITCH":
            if current is not None and current != o:
                return Decision(
                    update_id=uid,
                    subject=s,
                    predicate=p,
                    template_type=t,
                    requested_object=o,
                    previous_object=current,
                    action="APPLY",
                    new_value=o
                )
            else:
                return Decision(
                    update_id=uid,
                    subject=s,
                    predicate=p,
                    template_type=t,
                    requested_object=o,
                    previous_object=current,
                    action="NO_OP",
                    new_value=current
                )

        # --- END ---
        elif t == "END":
            if current is not None:
                return Decision(
                    update_id=uid,
                    subject=s,
                    predicate=p,
                    template_type=t,
                    requested_object=o,
                    previous_object=current,
                    action="APPLY",
                    new_value=None
                )
            else:
                return Decision(
                    update_id=uid,
                    subject=s,
                    predicate=p,
                    template_type=t,
                    requested_object=o,
                    previous_object=current,
                    action="NO_OP",
                    new_value=None
                )

        else:
            raise ValueError(f"Unknown template_type: {t}")