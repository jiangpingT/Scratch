from __future__ import annotations

import os
from datetime import datetime
from email.utils import formatdate
from pathlib import Path
from typing import List

from .. import config


def list_episodes() -> List[Path]:
    d = Path(config.OUTPUT_DIR)
    if not d.exists():
        return []
    return sorted([p for p in d.glob("*.mp3")], key=lambda p: p.stat().st_mtime, reverse=True)


def build_feed() -> str:
    items = list_episodes()
    now_rfc = formatdate(usegmt=True)
    title = config.PODCAST_FEED_TITLE
    author = config.PODCAST_FEED_AUTHOR
    base = config.PODCAST_BASE_URL.rstrip("/")
    xml_items: List[str] = []
    for p in items:
        fname = p.name
        url = f"{base}/{fname}" if base else fname
        size = p.stat().st_size
        pubdate = formatdate(p.stat().st_mtime, usegmt=True)
        xml_items.append(
            f"""
            <item>
              <title>{fname}</title>
              <link>{url}</link>
              <guid isPermaLink="false">{fname}</guid>
              <pubDate>{pubdate}</pubDate>
              <enclosure url="{url}" length="{size}" type="audio/mpeg"/>
              <author>{author}</author>
              <description>Auto-generated daily AI podcast</description>
            </item>
            """.strip()
        )

    xml = f"""
    <?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
      <channel>
        <title>{title}</title>
        <link>{base}</link>
        <description>Daily AI papers and meta-thoughts</description>
        <language>zh-cn</language>
        <lastBuildDate>{now_rfc}</lastBuildDate>
        {''.join(xml_items)}
      </channel>
    </rss>
    """.strip()
    return xml


def main() -> None:
    out_dir = Path(config.OUTPUT_DIR).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    feed_xml = build_feed()
    feed_path = out_dir / "feed.xml"
    feed_path.write_text(feed_xml, encoding="utf-8")
    print(f"Wrote feed: {feed_path}")


if __name__ == "__main__":
    main()

