from __future__ import annotations

from typing import Dict, List


def summarize_items(items: List[Dict]) -> str:
    lines = []
    for it in items[:6]:
        title = it.get("title", "").strip()
        link = it.get("link", "").strip()
        lines.append(f"- {title}（详见：{link}）")
    if not lines:
        lines = ["- 今日暂无新内容，回顾近期方法与趋势。"]
    return "\n".join(lines)


def generate_duo_script(bullets: str, duration_min: int = 5) -> str:
    intro = "A: 早上好，欢迎收听我们的AI每日简报。\nB: 今天我们聚焦大模型、Agent与机器人。"
    body = []
    for line in bullets.splitlines():
        if not line.strip():
            continue
        topic = line.lstrip("- ")
        body.append(f"A: 今天的要点：{topic}")
        body.append("B: 这个方向的实际影响在于落地效率与安全性。")
    outro = "A: 以上就是今天的主要内容。\nB: 明天见，记得订阅我们的播客。"
    return "\n".join([intro, *body, outro])

