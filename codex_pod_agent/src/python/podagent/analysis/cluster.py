from __future__ import annotations

from typing import Dict, List, Tuple


def _tokens(text: str) -> set[str]:
    import re

    text = text.lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff ]+", " ", text)
    toks = {t for t in text.split() if len(t) >= 2}
    return toks


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def cluster_items(items: List[Dict], threshold: float = 0.55) -> List[List[Dict]]:
    reps: List[Tuple[set[str], List[Dict]]] = []
    for it in items:
        text = ((it.get("title") or "") + " " + (it.get("summary") or ""))[:2000]
        tok = _tokens(text)
        placed = False
        for i, (rep_tok, group) in enumerate(reps):
            if jaccard(tok, rep_tok) >= threshold:
                group.append(it)
                reps[i] = (rep_tok | tok, group)
                placed = True
                break
        if not placed:
            reps.append((tok, [it]))
    return [g for _, g in reps]

