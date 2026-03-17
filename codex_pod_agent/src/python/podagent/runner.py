from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from . import config
from .fetchers.arxiv import fetch_latest as fetch_arxiv
from .fetchers.rss import fetch_latest as fetch_rss
from .llm.claude_agent import summarize_items, generate_duo_script
from .llm.claude_loop import summarize_items_with_loop, generate_duo_script_with_loop
from .llm import dryrun as dry_llm
from .context.manager import load_history, save_episode
from .context.store import dedup_items_by_links, EpisodeContext
from .verify import verify_bullets, verify_script
from .podcast.render import render_script_to_mp3


def collect_sources() -> List[Dict]:
    arxiv = fetch_arxiv(limit=8)
    rss = fetch_rss(limit=10)
    merged: List[Dict] = []
    for x in arxiv:
        merged.append({
            "title": x.title,
            "link": x.link,
            "summary": x.summary,
            "published": x.published,
            "source": x.source,
        })
    for x in rss:
        merged.append({
            "title": x.title,
            "link": x.link,
            "summary": x.summary,
            "published": x.published,
            "source": x.source,
        })
    # 基于历史已播出的链接做去重，减少重复
    history = load_history(limit=config.DEDUP_HISTORY)
    seen_links = []
    for h in history:
        seen_links.extend(h.links)
    return dedup_items_by_links(merged, seen_links)


def run_once() -> Path:
    items = collect_sources()
    if config.DRY_RUN:
        bullets = dry_llm.summarize_items(items)
        script = dry_llm.generate_duo_script(bullets)
    else:
        try:
            if config.CLAUDE_USE_LOOP:
                bullets = summarize_items_with_loop(items)
                # Verify bullets
                try:
                    bullets = verify_bullets(bullets)
                except Exception:
                    pass
                script = generate_duo_script_with_loop(bullets)
                # Verify script
                try:
                    script = verify_script(script)
                except Exception:
                    pass
            else:
                bullets = summarize_items(items)
                try:
                    bullets = verify_bullets(bullets)
                except Exception:
                    pass
                script = generate_duo_script(bullets)
                try:
                    script = verify_script(script)
                except Exception:
                    pass
        except Exception as e:
            print(f"LLM 生成失败，回退离线模式: {e}")
            bullets = dry_llm.summarize_items(items)
            script = dry_llm.generate_duo_script(bullets)
    out_dir = Path(config.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    mp3_path = out_dir / f"{config.today_str()}-daily-ai-duo.mp3"
    render_script_to_mp3(script, str(mp3_path))
    # 保存上下文以供后续去重与脉络
    links = [it.get("link", "") for it in items if it.get("link")]
    save_episode(EpisodeContext(date=config.today_str(), bullets=bullets, script=script, links=links))
    return mp3_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="仅运行一次")
    args = parser.parse_args()
    if args.once:
        out = run_once()
        print(f"Generated: {out}")
    else:
        print("建议使用系统计划任务(如cron)在每日09:00运行：\n  make build")


if __name__ == "__main__":
    main()
