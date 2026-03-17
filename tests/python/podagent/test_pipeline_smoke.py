from pathlib import Path

from podagent import config
from podagent.llm import dryrun
from podagent.podcast.dialogue import parse_script_to_turns
from podagent.feed.generate_feed import build_feed


def test_dryrun_script_and_feed(tmp_path):
    # 构造少量伪造条目，避免外网与API依赖
    items = [
        {
            "title": "Agent tool-use improves LLM planning",
            "link": "https://example.com/a1",
            "summary": "We explore loop-based agent planning for complex tasks.",
            "published": 0,
            "source": "arXiv",
        },
        {
            "title": "Robotics with foundation models",
            "link": "https://example.com/r1",
            "summary": "Large models enable rapid policy adaptation in robots.",
            "published": 0,
            "source": "blog",
        },
    ]

    bullets = dryrun.summarize_items(items)
    assert "- " in bullets

    script = dryrun.generate_duo_script(bullets)
    turns = parse_script_to_turns(script)
    assert len(turns) >= 4
    assert set(s for s, _ in turns).issubset({"A", "B"})

    # 构造假音频与feed
    out_dir = tmp_path / "episodes"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "2025-01-01-daily-ai-duo.mp3").write_bytes(b"FAKE-MP3")

    old_output = config.OUTPUT_DIR
    try:
        # 将输出目录重定向到临时目录
        config.OUTPUT_DIR = str(out_dir)
        xml = build_feed()
        assert "2025-01-01-daily-ai-duo.mp3" in xml
        assert "<rss" in xml and "</rss>" in xml
    finally:
        config.OUTPUT_DIR = old_output

