import pytest

from triage import gate, graph
from triage.llm import LLMUnavailable
from triage.retrieval import load_index
from triage.schemas import Classification, Draft


def _fake(cls=None, draft=None, fail=()):
    async def fake(system, user, schema, temperature=0.0):
        if schema is Classification:
            if "classify" in fail:
                raise LLMUnavailable("down")
            return cls or Classification(intent="billing", urgency="normal", confidence=0.9, reason="x")
        if "draft" in fail:
            raise LLMUnavailable("down")
        return draft or Draft(reply="Refunded.", cited_case_ids=["c01"], needs_more_info=False)
    return fake


def test_retrieval_finds_the_matching_case():
    top = load_index().search("I was charged twice for the same invoice", 3)
    assert top[0]["id"] == "c01" and top[0]["score"] > 0.2


def test_retrieval_scores_unrelated_text_low():
    assert load_index().search("zebra giraffe holiday photos", 1)[0]["score"] < 0.2


def _state(conf=0.9, urgency="normal", score=0.5, cited=("c01",), more=False, draft=True):
    return {"classification": {"intent": "billing", "urgency": urgency, "confidence": conf, "reason": ""},
            "retrieved": [{"id": "c01", "score": score}],
            "draft": {"reply": "", "cited_case_ids": list(cited), "needs_more_info": more} if draft else None}


def test_gate_passes_a_clean_ticket():
    assert gate.decide(_state()) == {"escalate": False, "reasons": []}


@pytest.mark.parametrize("kw,reason", [
    ({"conf": 0.4}, "low_classifier_confidence"),
    ({"urgency": "critical"}, "critical_urgency"),
    ({"score": 0.05}, "no_similar_case"),
    ({"cited": ()}, "no_citation"),
    ({"cited": ("c99",)}, "invalid_citation"),
    ({"more": True}, "needs_more_info"),
    ({"draft": False}, "draft_unavailable"),
])
def test_gate_reasons(kw, reason):
    r = gate.decide(_state(**kw))
    assert r["escalate"] and reason in r["reasons"]


def test_gate_escalates_when_classifier_is_down():
    s = _state(); s["classification"] = None
    assert "classifier_unavailable" in gate.decide(s)["reasons"]


async def test_pipeline_happy_path(monkeypatch):
    monkeypatch.setattr(graph, "structured_call", _fake())
    r = await graph.run_ticket("I was charged twice for the same invoice, please refund one.")
    assert r["classification"]["intent"] == "billing"
    assert r["escalation"] == {"escalate": False, "reasons": []}
    assert r["degraded"] == []


async def test_pipeline_degrades_and_escalates_when_llm_is_down(monkeypatch):
    monkeypatch.setattr(graph, "structured_call", _fake(fail=("classify", "draft")))
    r = await graph.run_ticket("I was charged twice for the same invoice.")
    assert set(r["degraded"]) == {"classify", "draft"}
    assert r["escalation"]["escalate"]


async def test_fabricated_citation_is_escalated(monkeypatch):
    bad = Draft(reply="Done.", cited_case_ids=["c42"], needs_more_info=False)
    monkeypatch.setattr(graph, "structured_call", _fake(draft=bad))
    r = await graph.run_ticket("I was charged twice for the same invoice.")
    assert "invalid_citation" in r["escalation"]["reasons"]


async def test_speaker_lines_are_parsed(monkeypatch):
    monkeypatch.setattr(graph, "structured_call", _fake())
    r = await graph.run_ticket("Agent: Hello, how can I help?\nCustomer: I was charged twice.")
    assert [u["speaker"] for u in r["utterances"]] == ["agent", "customer"]
