from __future__ import annotations

from pathlib import Path

from .dialogue import parse_script_to_turns
from .tts import synthesize_dialogue


def render_script_to_mp3(script: str, out_path: str) -> str:
    turns = parse_script_to_turns(script)
    Path(Path(out_path).parent).mkdir(parents=True, exist_ok=True)
    return synthesize_dialogue(turns, out_path)

