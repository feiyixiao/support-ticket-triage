"""Escalation gate: pure, deterministic rules, no LLM. Escalate to a human when ANY reason fires.

The confidence threshold and the similarity floor are first guesses. They are not tuned or
validated; `eval/run_eval.py` reports how the gate behaves so they can be tuned on a dev split."""
from triage.config import settings


def decide(state: dict) -> dict:
    reasons: list[str] = []
    c = state.get("classification")
    if c is None:
        reasons.append("classifier_unavailable")
    else:
        if c["confidence"] < settings.conf_threshold:
            reasons.append("low_classifier_confidence")
        if c["urgency"] == "critical":
            reasons.append("critical_urgency")
    retrieved = state.get("retrieved") or []
    if not retrieved or retrieved[0]["score"] < settings.min_similarity:
        reasons.append("no_similar_case")
    d = state.get("draft")
    if d is None:
        reasons.append("draft_unavailable")
    else:
        valid = {r["id"] for r in retrieved}
        if not d["cited_case_ids"]:
            reasons.append("no_citation")
        elif not set(d["cited_case_ids"]) <= valid:
            reasons.append("invalid_citation")  # cited a case that was not retrieved: fabricated
        if d["needs_more_info"]:
            reasons.append("needs_more_info")
    return {"escalate": bool(reasons), "reasons": reasons}
