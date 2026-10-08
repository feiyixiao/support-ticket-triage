"""5-stage LangGraph pipeline:

  transcribe(+diarize) -> classify -> retrieve -> draft -> escalate

A failing LLM stage records itself in `degraded` and the pipeline continues; the escalation
gate then sends the ticket to a human, so an outage never produces an unreviewed answer."""
from typing import Annotated, TypedDict
import operator

from langgraph.graph import END, StateGraph

from triage import asr, gate
from triage.config import settings
from triage.llm import LLMUnavailable, structured_call
from triage.retrieval import load_index
from triage.schemas import Classification, Draft

_index = None

CLASSIFY_SYSTEM = (
    "You triage customer-support tickets. Return intent (billing, technical_issue, account_access, "
    "shipping, cancellation, feature_request, other) and urgency (low, normal, high, critical). "
    "critical = production outage, suspected account takeover or fraud, or a legal threat; high = money "
    "or access blocked; normal = a normal problem; low = question or wish. "
    "`confidence` is the probability that BOTH intent and urgency are correct: be calibrated, and use "
    "values below 0.6 for vague tickets. Return JSON.")

DRAFT_SYSTEM = (
    "You draft a support reply using ONLY the resolved cases below. If none of them applies, set "
    "needs_more_info=true and ask one clarifying question. List the ids of the cases you used in "
    "cited_case_ids; never cite an id that is not listed. At most 5 sentences. Return JSON.\n\nCases:\n{cases}")


class TriageState(TypedDict, total=False):
    text: str
    audio_path: str
    utterances: list[dict]
    diarization: str
    transcript: str
    classification: dict
    retrieved: list[dict]
    draft: dict
    escalation: dict
    degraded: Annotated[list[str], operator.add]


async def transcribe_node(s: TriageState) -> dict:
    if s.get("audio_path"):
        try:
            utts = await asr.transcribe(s["audio_path"])
        except Exception:
            return {"degraded": ["transcribe"], "utterances": [], "transcript": ""}
        utts, status = asr.diarize(s["audio_path"], utts)
    else:
        utts, status = asr.parse_text(s.get("text", "")), "n/a"
    return {"utterances": utts, "diarization": status,
            "transcript": " ".join(u["text"] for u in utts)}


async def classify_node(s: TriageState) -> dict:
    if not s.get("transcript"):
        return {"degraded": ["classify"]}
    try:
        c = await structured_call(CLASSIFY_SYSTEM, s["transcript"], Classification)
        return {"classification": c.model_dump()}
    except LLMUnavailable:
        return {"degraded": ["classify"]}


async def retrieve_node(s: TriageState) -> dict:
    global _index
    _index = _index or load_index()
    return {"retrieved": _index.search(s.get("transcript", ""), settings.top_k)}


async def draft_node(s: TriageState) -> dict:
    cases = "\n".join(f"[{r['id']}] problem: {r['problem']} | resolution: {r['resolution']}"
                      for r in s.get("retrieved", []))
    if not s.get("transcript"):
        return {"degraded": ["draft"]}
    try:
        d = await structured_call(DRAFT_SYSTEM.format(cases=cases or "(none)"), s["transcript"], Draft)
        return {"draft": d.model_dump()}
    except LLMUnavailable:
        return {"degraded": ["draft"]}


async def escalate_node(s: TriageState) -> dict:
    return {"escalation": gate.decide(s)}


def build_graph():
    g = StateGraph(TriageState)
    for name, fn in [("transcribe", transcribe_node), ("classify", classify_node),
                     ("retrieve", retrieve_node), ("draft", draft_node), ("escalate", escalate_node)]:
        g.add_node(name, fn)
    g.set_entry_point("transcribe")
    for a, b in [("transcribe", "classify"), ("classify", "retrieve"),
                 ("retrieve", "draft"), ("draft", "escalate")]:
        g.add_edge(a, b)
    g.add_edge("escalate", END)
    return g.compile()


graph = build_graph()


async def run_ticket(text: str = "", audio_path: str = "") -> dict:
    return await graph.ainvoke({"text": text, "audio_path": audio_path, "degraded": []})
