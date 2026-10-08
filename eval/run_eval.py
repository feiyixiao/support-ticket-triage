"""Offline eval on data/eval_tickets.jsonl (24 SYNTHETIC tickets, one annotator).

    python -m eval.run_eval            # needs GROQ_API_KEY; ~50 LLM calls

Reports intent/urgency accuracy, escalation precision/recall, and how well the classifier's own
confidence predicts correctness (reliability bins, ECE, AUROC). n is tiny: read these as a smoke
test and a way to find failure cases, not as performance claims."""
import asyncio
import json
from datetime import datetime
from pathlib import Path

from triage.graph import run_ticket

HERE = Path(__file__).parent


def ece(conf, ok, bins=3):
    n = len(conf)
    out = 0.0
    for b in range(bins):
        idx = [i for i, c in enumerate(conf) if min(int(c * bins), bins - 1) == b]
        if idx:
            out += len(idx) / n * abs(sum(ok[i] for i in idx) / len(idx) - sum(conf[i] for i in idx) / len(idx))
    return out


def auroc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    return sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))


async def main():
    items = [json.loads(l) for l in open(HERE.parent / "data/eval_tickets.jsonl")]
    sem = asyncio.Semaphore(2)  # free tier: 8k tokens/min per model

    async def one(it):
        async with sem:
            return await run_ticket(it["text"])

    runs = await asyncio.gather(*(one(it) for it in items))
    rows, conf, ok = [], [], []
    tp = fp = fn = tn = 0
    for it, r in zip(items, runs):
        c = r.get("classification")
        esc = r["escalation"]["escalate"]
        row = {"id": it["id"], "text": it["text"], "gold": it, "pred": c, "escalate": esc,
               "reasons": r["escalation"]["reasons"], "degraded": r.get("degraded", [])}
        rows.append(row)
        if c:
            both = c["intent"] == it["intent"] and c["urgency"] == it["urgency"]
            conf.append(c["confidence"]); ok.append(both)
        tp += esc and it["should_escalate"]; fp += esc and not it["should_escalate"]
        fn += (not esc) and it["should_escalate"]; tn += (not esc) and not it["should_escalate"]
    n = len(items)
    cls = [r for r in rows if r["pred"]]
    summary = {
        "n": n, "classified": len(cls),
        "intent_acc": sum(r["pred"]["intent"] == r["gold"]["intent"] for r in cls) / max(len(cls), 1),
        "urgency_acc": sum(r["pred"]["urgency"] == r["gold"]["urgency"] for r in cls) / max(len(cls), 1),
        "escalation": {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                       "precision": tp / (tp + fp) if tp + fp else 0.0,
                       "recall": tp / (tp + fn) if tp + fn else 0.0,
                       "escalation_rate": (tp + fp) / n},
        "confidence": {"ece_3bins": ece(conf, ok) if conf else None,
                       "auroc_conf_vs_correct": auroc(conf, ok) if conf else None,
                       "mean_conf": sum(conf) / len(conf) if conf else None,
                       "acc_both": sum(ok) / len(ok) if ok else None},
        "missed_escalations": [r["id"] for r in rows if not r["escalate"] and r["gold"]["should_escalate"]],
        "false_escalations": [r["id"] for r in rows if r["escalate"] and not r["gold"]["should_escalate"]],
    }
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    path = out / f"{datetime.now():%Y-%m-%dT%H%M%S}.json"
    json.dump({"summary": summary, "rows": rows}, open(path, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(summary, indent=1)); print("->", path)

if __name__ == "__main__":
    asyncio.run(main())
