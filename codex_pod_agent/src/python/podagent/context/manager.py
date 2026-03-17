from __future__ import annotations

from typing import Iterable, List

from .store import EpisodeContext, load_history as load_local, save_episode as save_local
from .platform import enabled as platform_enabled, pull_recent, push_episode


def load_history(limit: int = 3) -> List[EpisodeContext]:
    local = load_local(limit)
    if platform_enabled():
        try:
            remote = pull_recent(limit)
            # 简单合并（以本地顺序为主，追加远端中本地缺失的）
            seen = {(e.date, e.script[:50]) for e in local}
            for r in remote:
                key = (r.date, r.script[:50])
                if key not in seen:
                    local.append(r)
        except Exception:
            pass
    return local[-limit:]


def save_episode(ctx: EpisodeContext) -> None:
    save_local(ctx)
    if platform_enabled():
        try:
            push_episode(ctx)
        except Exception:
            # 远端失败不影响本地保存
            pass

