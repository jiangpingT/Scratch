from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List


FEEDS = [
    "https://www.alignmentforum.org/feed.xml",
    "https://www.lesswrong.com/feed.xml",
    "https://openai.com/blog/rss.xml",
    "https://www.anthropic.com/news/rss.xml",
    "https://deepmind.google/discover/blog/feed.xml",
]


@dataclass
class FeedItem:
    title: str
    link: str
    summary: str
    published: float
    source: str


def fetch_latest(limit: int = 10) -> List[FeedItem]:
    try:
        import feedparser  # type: ignore
    except Exception as e:
        raise RuntimeError("feedparser 未安装：pip install feedparser") from e

    items: List[FeedItem] = []
    for url in FEEDS:
        feed = feedparser.parse(url)
        for e in feed.entries[: limit // len(FEEDS) + 5]:
            published = None
            if hasattr(e, "published_parsed") and e.published_parsed:
                published = time.mktime(e.published_parsed)
            elif hasattr(e, "updated_parsed") and e.updated_parsed:
                published = time.mktime(e.updated_parsed)
            else:
                published = time.time()
            items.append(
                FeedItem(
                    title=getattr(e, "title", ""),
                    link=getattr(e, "link", ""),
                    summary=getattr(e, "summary", "") or getattr(e, "description", ""),
                    published=published,
                    source=url,
                )
            )
    items.sort(key=lambda x: x.published, reverse=True)
    return items[:limit]

