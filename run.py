"""python run.py --text "I was charged twice"   |   python run.py --audio call.wav"""
import argparse
import asyncio
import json

from triage.graph import run_ticket

ap = argparse.ArgumentParser()
ap.add_argument("--text", default="")
ap.add_argument("--audio", default="")
a = ap.parse_args()
r = asyncio.run(run_ticket(a.text, a.audio))
print(json.dumps({k: r.get(k) for k in ("classification", "retrieved", "draft", "escalation", "degraded")},
                 ensure_ascii=False, indent=2))
