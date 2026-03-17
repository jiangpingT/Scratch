from __future__ import annotations

from typing import List, Tuple


def sanitize_text(text: str) -> str:
    # 去除常见符号与 Markdown 痕迹，避免TTS读出“井号/星号”等
    to_strip = ["#", "*", "•", "_", "`"]
    for ch in to_strip:
        text = text.replace(ch, "")
    # 去掉多余的空白与无意义前缀
    text = text.strip(" -\t")
    return " ".join(text.split())


def parse_script_to_turns(script: str) -> List[Tuple[str, str]]:
    lines = [l.strip() for l in script.splitlines() if l.strip()]
    turns: List[Tuple[str, str]] = []
    for l in lines:
        if l.startswith("A:") or l.startswith("A："):
            content = l.split(":", 1)[-1].strip() if ":" in l else l[2:].strip()
            content = sanitize_text(content)
            if content:
                turns.append(("A", content))
        elif l.startswith("B:") or l.startswith("B："):
            content = l.split(":", 1)[-1].strip() if ":" in l else l[2:].strip()
            content = sanitize_text(content)
            if content:
                turns.append(("B", content))
    if not turns:
        for i, l in enumerate(lines):
            speaker = "A" if i % 2 == 0 else "B"
            content = sanitize_text(l)
            if content:
                turns.append((speaker, content))
    return turns
