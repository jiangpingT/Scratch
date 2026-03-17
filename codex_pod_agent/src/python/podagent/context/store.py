from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .. import config


CTX_DIR = Path("sessions")
CTX_FILE = CTX_DIR / "context.jsonl"


@dataclass
class EpisodeContext:
    date: str
    bullets: str
    script: str
    links: List[str]


def _ensure_files() -> None:
    CTX_DIR.mkdir(parents=True, exist_ok=True)
    if not CTX_FILE.exists():
        CTX_FILE.touch()


def load_history(limit: int = 3) -> List[EpisodeContext]:
    _ensure_files()
    items: List[EpisodeContext] = []
    with open(CTX_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                items.append(
                    EpisodeContext(
                        date=obj.get("date", ""),
                        bullets=obj.get("bullets", ""),
                        script=obj.get("script", ""),
                        links=obj.get("links", []) or [],
                    )
                )
            except Exception:
                continue
    return items[-limit:]


def save_episode(ctx: EpisodeContext) -> None:
    _ensure_files()
    with open(CTX_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(ctx), ensure_ascii=False) + "\n")


def dedup_items_by_links(items: List[Dict], seen_links: Iterable[str]) -> List[Dict]:
    seen = {l for l in seen_links if l}
    out: List[Dict] = []
    for it in items:
        link = (it.get("link") or "").strip()
        if link and link in seen:
            continue
        out.append(it)
    return out

