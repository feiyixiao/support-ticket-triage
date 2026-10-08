"""Stage 1 helpers: Whisper transcription (Groq) and optional speaker diarization (pyannote).

Diarization labels are attached to utterances for display only; classification uses the whole
transcript, so a wrong speaker label cannot change the triage decision."""
import os
import re
from pathlib import Path

from triage.config import settings


def parse_text(text: str) -> list[dict]:
    """Typed tickets / transcripts. Lines like 'Customer: ...' / 'Agent: ...' keep their speaker."""
    utts = []
    for line in text.strip().splitlines():
        m = re.match(r"^\s*(customer|agent|kunde|support)\s*:\s*(.*)$", line, re.I)
        utts.append({"speaker": m.group(1).lower() if m else "unknown",
                     "text": (m.group(2) if m else line).strip(), "start": None, "end": None})
    return [u for u in utts if u["text"]]


async def transcribe(audio_path: str) -> list[dict]:
    from triage.llm import get_client
    p = Path(audio_path)
    r = await get_client().audio.transcriptions.create(
        file=(p.name, p.read_bytes()), model=settings.asr_model, response_format="verbose_json")
    segs = getattr(r, "segments", None) or []
    if not segs:
        return [{"speaker": "unknown", "text": r.text.strip(), "start": None, "end": None}]
    return [{"speaker": "unknown", "text": s["text"].strip(), "start": s["start"], "end": s["end"]}
            for s in segs]


def diarize(audio_path: str, utts: list[dict]) -> tuple[list[dict], str]:
    """Assign SPEAKER_xx by maximum time overlap. Returns (utterances, status).
    status is 'pyannote' or 'none' (pyannote / HF_TOKEN not available, labels stay 'unknown')."""
    token = os.getenv("HF_TOKEN")
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        return utts, "none"
    if not token:
        return utts, "none"
    pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
    turns = [(t.start, t.end, spk) for t, _, spk in pipe(audio_path).itertracks(yield_label=True)]
    out = []
    for u in utts:
        best, best_ov = "unknown", 0.0
        for s, e, spk in turns:
            ov = min(e, u["end"] or 0) - max(s, u["start"] or 0)
            if ov > best_ov:
                best, best_ov = spk, ov
        out.append({**u, "speaker": best})
    return out, "pyannote"
