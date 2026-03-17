from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List


@dataclass
class ArxivItem:
    title: str
    link: str
    summary: str
    published: float
    source: str


def fetch_latest(limit: int = 8) -> List[ArxivItem]:
    try:
        import feedparser  # type: ignore
    except Exception as e:
        raise RuntimeError("feedparser 未安装：pip install feedparser") from e

    query = (
        "http://export.arxiv.org/api/query?"
        "search_query=cat:cs.AI+OR+cat:cs.CL+OR+cat:cs.LG+OR+cat:cs.RO+AND+all:(agent+OR+robot+OR+\"large+language+model\")&"
        "sortBy=lastUpdatedDate&sortOrder=descending&max_results=20"
    )
    feed = feedparser.parse(query)
    items: List[ArxivItem] = []
    now = time.time()
    for e in feed.entries:
        ts = now
        if hasattr(e, "published_parsed") and e.published_parsed:
            ts = time.mktime(e.published_parsed)
        items.append(
            ArxivItem(
                title=getattr(e, "title", ""),
                link=getattr(e, "link", ""),
                summary=getattr(e, "summary", ""),
                published=ts,
                source="arXiv",
            )
        )
    items.sort(key=lambda x: x.published, reverse=True)
    return items[:limit]

