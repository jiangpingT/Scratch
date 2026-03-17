from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore


@dataclass
class LoopStrategy:
    topk: int = 10
    cluster_threshold: float = 0.55
    max_tool_steps_summary: int = 4
    max_tool_steps_script: int = 2


def load_strategy(path: str | None = None) -> LoopStrategy:
    # 默认读取仓库内的配置文件
    p = Path(path or "codex_pod_agent/config/loop_strategy.yaml").resolve()
    if not p.exists():
        return LoopStrategy()
    data: dict[str, Any] = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return LoopStrategy(
        topk=int(data.get("topk", 10)),
        cluster_threshold=float(data.get("cluster_threshold", 0.55)),
        max_tool_steps_summary=int(data.get("max_tool_steps_summary", 4)),
        max_tool_steps_script=int(data.get("max_tool_steps_script", 2)),
    )

