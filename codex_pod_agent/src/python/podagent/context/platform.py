from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List

import requests  # type: ignore

from .store import EpisodeContext


API_URL = os.getenv("CLAUDE_PLATFORM_API_URL", "").rstrip("/")
API_KEY = os.getenv("CLAUDE_PLATFORM_API_KEY", "")


def enabled() -> bool:
    return bool(API_URL and API_KEY)


def _headers() -> dict:
    return {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


def pull_recent(limit: int = 3) -> List[EpisodeContext]:
    if not enabled():
        return []
    url = f"{API_URL}/context/recent?limit={int(limit)}"
    r = requests.get(url, headers=_headers(), timeout=10)
    r.raise_for_status()
    data = r.json() or []
    out: List[EpisodeContext] = []
    for d in data:
        out.append(EpisodeContext(date=d.get("date", ""), bullets=d.get("bullets", ""), script=d.get("script", ""), links=d.get("links", []) or []))
    return out


def push_episode(ctx: EpisodeContext) -> None:
    if not enabled():
        return
    url = f"{API_URL}/context/episode"
    body = {"date": ctx.date, "bullets": ctx.bullets, "script": ctx.script, "links": ctx.links}
    r = requests.post(url, headers=_headers(), data=json.dumps(body), timeout=10)
    r.raise_for_status()

