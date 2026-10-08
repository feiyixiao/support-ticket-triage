"""Retrieval over resolved cases: pure-Python TF-IDF + cosine (no extra dependency).

Swap for ChromaDB / FAISS embeddings by replacing `CaseIndex`; the pipeline only needs
`search(text, k) -> [{"id", "intent", "problem", "resolution", "score"}]`."""
import json
import math
import re
from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "resolved_cases.jsonl"
STOP = set("a an the and or but of to in on for with is are was were be been it its i my me we our you your "
           "this that these those at by from as not no do does did can could please have has had will would "
           "there then so if when how what why".split())


def tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-zäöüß0-9]+", text.lower()) if t not in STOP and len(t) > 1]


class CaseIndex:
    def __init__(self, cases: list[dict]):
        self.cases = cases
        docs = [Counter(tokens(c["problem"] + " " + c["resolution"])) for c in cases]
        df = Counter(t for d in docs for t in d)
        n = len(docs)
        self.idf = {t: math.log((1 + n) / (1 + c)) + 1 for t, c in df.items()}
        self.vecs = [self._vec(d) for d in docs]

    def _vec(self, counts: Counter) -> dict[str, float]:
        v = {t: (1 + math.log(c)) * self.idf[t] for t, c in counts.items() if t in self.idf}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    def search(self, text: str, k: int = 3) -> list[dict]:
        q = self._vec(Counter(tokens(text)))
        scored = [(sum(w * v.get(t, 0.0) for t, w in q.items()), c) for v, c in zip(self.vecs, self.cases)]
        scored.sort(key=lambda x: -x[0])
        return [{**c, "score": round(s, 3)} for s, c in scored[:k]]


def load_index(path: Path = DATA) -> CaseIndex:
    return CaseIndex([json.loads(line) for line in open(path) if line.strip()])
