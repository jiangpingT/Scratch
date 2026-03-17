from __future__ import annotations

import math
import time
from typing import Dict, List, Tuple


KEYWORDS = {
    "agent": 1.4,
    "agents": 1.4,
    "ai agent": 1.6,
    "llm": 1.3,
    "large language model": 1.5,
    "robot": 1.2,
    "robotics": 1.2,
    "tool": 1.1,
    "loop": 1.0,
}

SOURCE_WEIGHT = {
    "arXiv": 1.1,
}


def _kw_score(text: str) -> float:
    t = text.lower()
    return sum(w for k, w in KEYWORDS.items() if k in t)


def _recency_score(ts: float, now: float | None = None) -> float:
    now = now or time.time()
    days = max(0.0, (now - ts) / 86400.0)
    return 1.5 * math.exp(-0.3 * days)


def score_item(item: Dict) -> float:
    title = (item.get("title") or "") + " " + (item.get("summary") or "")
    kw = _kw_score(title)
    rec = _recency_score(float(item.get("published") or time.time()))
    src = SOURCE_WEIGHT.get(item.get("source") or "", 1.0)
    return kw + rec + 0.1 * src


def rank_items(items: List[Dict], topk: int = 10) -> List[Tuple[float, Dict]]:
    scored = [(score_item(it), it) for it in items]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:topk]

