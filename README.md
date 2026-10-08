# Support Ticket Triage & Resolution Assistant (minimal rebuild)

Rebuilt on 2026-10-07 after an earlier version was lost. It matches the CV line: a 5-stage LangGraph pipeline
(transcription + speaker diarization, intent / urgency classification, retrieval over resolved cases,
response drafting, confidence-based escalation to a human). It is a **minimal** version, built in one session.

```
text / audio -> [1 transcribe (+diarize)] -> [2 classify] -> [3 retrieve] -> [4 draft] -> [5 escalate gate]
```

| Stage | What it does | Status |
|---|---|---|
| 1 transcribe | Groq Whisper (`verbose_json` segments) for audio; typed text with `Customer:` / `Agent:` lines is parsed | text path tested; **audio path not run on real audio** |
| 1b diarize | pyannote 3.1 by max time overlap, only if `pyannote.audio` + `HF_TOKEN` exist; otherwise labels stay `unknown` | **not installed, not tested**; labels never affect classification |
| 2 classify | JSON-schema + Pydantic: intent, urgency, self-reported `confidence`, via a model fallback chain | tested (offline + live) |
| 3 retrieve | pure-Python TF-IDF cosine over 12 **synthetic** resolved cases (`data/resolved_cases.jsonl`) | tested; no ChromaDB / FAISS in this version |
| 4 draft | reply from retrieved cases only; must list `cited_case_ids` | tested |
| 5 escalate | deterministic rules, no LLM: low confidence, critical urgency, no similar case, no/invalid citation, needs more info, any stage down | tested; thresholds are **untuned first guesses** |

A failing LLM stage records itself in `degraded`; the gate then escalates, so an outage never yields an unreviewed answer.

```bash
cp .env.example .env            # GROQ_API_KEY
pip install -r requirements.txt
python run.py --text "I was charged twice for my March invoice."
pytest -q                       # 15 offline tests
python -m eval.run_eval         # ~50 LLM calls, writes eval/results/<ts>.json
```

## What was measured (2026-10-07, smoke run, not a result to quote)

24 **synthetic** tickets I wrote with gold labels (raw output: `eval/results/2026-10-07T170540.json`) (one annotator, no real customers), `openai/gpt-oss-20b` first in the chain.

| | |
|---|---|
| intent accuracy | 0.875 (21/24) |
| urgency accuracy | 0.625 (15/24); many gold urgencies are judgement calls (e.g. a double charge: normal or high?) |
| escalation (gold = needs a human) | recall 1.0 (10/10), precision 0.71 (4 false escalations) |
| confidence | mean 0.88 vs 0.54 of tickets with both labels correct; ECE (3 bins) 0.34; AUROC 0.59 |

Reading: the self-reported confidence is **overconfident and barely separates right from wrong** here, so the escalation
recall comes from the other rules (no citation, no similar case, critical urgency), not from the confidence threshold.
With n = 24 and one annotator this is a smoke test and a list of failure cases (`eval/results/*.json`), not a performance claim.

## Known limitations
- Synthetic data only. No real tickets, no second annotator, no held-out split; the gate thresholds were not tuned.
- The draft can claim actions nobody took ("I've verified both invoice IDs" in the live smoke run). The gate checks
  citations, not whether the reply promises something the system cannot do.
- `no_citation` / `needs_more_info` fire together on out-of-scope tickets, so the false-escalation count is partly the gate being conservative.
- English only (TF-IDF stop words, prompts). No persistence, no API, no UI, no tracing, not deployed.
- Audio and diarization are written but unverified.

## Next
1. Tune `CONF_THRESHOLD` / `MIN_SIMILARITY` on a dev split, evaluate once on a held-out split (same protocol as `speech-language-agent/eval/PROTOCOL.md`).
2. Replace verbalized confidence by sampling consistency and compare ECE / AUROC.
3. Check draft claims against the retrieved resolution text instead of trusting the generator.
4. Record 5-10 real audio samples to verify transcription and diarization.
